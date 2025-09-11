import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class DefenseTypes:
    NoDefense = 'NoDefense'
    Krum = 'Krum'
    TrimmedMean = 'TrimmedMean'
    Bulyan = 'Bulyan'
    Median = 'Median'
    Fang = 'Fang'
    Agr_filter = 'Agr_filter'
    Agr_xai = 'Agr_xai'

    def __str__(self):
        return self.value

def no_defense(users_grads, users_count, corrupted_count):
    return np.mean(users_grads, axis=0)

def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = torch.norm(users_grads[i] - users_grads[j])
    return distances

def krum(users_grads, users_count, corrupted_count, distances=None,return_index=False, debug=False):
    if not return_index:
        assert users_count >= 2*corrupted_count + 1,('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    minimal_error_index = -1

    if distances is None:
        distances = _krum_create_distances(users_grads)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    if return_index:
        return minimal_error_index
    else:
        return users_grads[minimal_error_index]

# def trimmed_mean(users_grads, users_count, corrupted_count):
#     number_to_consider = int(users_grads.shape[0] - corrupted_count) - 1
#     current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)
#
#     for i, param_across_users in enumerate(users_grads.T):
#         med = np.median(param_across_users)
#         good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
#         current_grads[i] = np.mean(good_vals) + med
#     return current_grads


def trimmed_mean(users_grads, corrupted_count):
    # 计算需要考虑的值的数量
    number_to_consider = users_grads.size(0) - int(users_grads.size(0)*corrupted_count)
    med=torch.median(users_grads[:,:],dim=0)[0]
    indices=torch.topk(torch.abs(users_grads-med),k=number_to_consider,largest=False,dim=0)[1]
    good_vals = torch.gather(users_grads,dim=0,index=indices)
    current_grads=torch.mean(good_vals,dim=0)

    return current_grads


def bulyan(users_grads, users_count, corrupted_count):
    assert users_count >= 4*corrupted_count + 3
    set_size = users_count - 2*corrupted_count
    selection_set = []

    distances = _krum_create_distances(users_grads)
    while len(selection_set) < set_size:
        currently_selected = krum(users_grads, users_count - len(selection_set), corrupted_count, distances, True)
        selection_set.append(users_grads[currently_selected])

        # remove the selected from next iterations:
        distances.pop(currently_selected)
        for remaining_user in distances.keys():
            distances[remaining_user].pop(currently_selected)

    return trimmed_mean(torch.stack(selection_set), 0.2)

def median(users_grads):
    med=torch.median(users_grads[:,:],dim=0)[0]
    return med

def fang(grad, corrupted_count, model, testLoader, pooling=False, n_H=1024, output_size=10, kernel=3, mode="combined"):
    base_param = get_flatten_parameters(model)
    acc_rec = torch.zeros(grad.size(0))
    loss_rec = torch.zeros(grad.size(0))
    replica = grad.clone()
    loss_function = nn.CrossEntropyLoss()
    optimizer= torch.optim.SGD(model.parameters(), lr=0.01)
    if pooling:
        grad = fang_pooling(grad, n_H, output_size, kernel_size=kernel)
    for i in range(grad.size(0)):
        load_flattened_parameters(grad[i],model)
        acc, loss = back_prop(model, testLoader, loss_function, optimizer)
        acc_rec[i] = acc
        loss_rec[i] = loss
    k_selection = int(round(grad.size(0) * (1 - corrupted_count)))
    ERR = torch.topk(acc_rec, k_selection).indices
    if mode == "err":
        return replica[ERR]
    LRR = torch.topk(loss_rec, k_selection, largest=False).indices
    if mode == "lrr":
        return replica[LRR]
    # If it's merged ERR and LRR, then compute the intersection
    union = torch.cat([ERR, LRR])
    uniques, counts = union.unique(return_counts=True)
    final_idx = uniques[counts > 1]

    return torch.mean(replica[final_idx], dim=0)


def fang_pooling(grad, n_H, output_size, kernel_size=3):
    """
    Pooling method designed for Fang AGR and others who need to restore the input size. See Alg. 1 and Fig. 2
    in the original paper
    :param grad: the input collection of gradients
    :param n_H: the size of hidden layer
    :param output_size: The size of output (to avoid vector size issue)
    :param kernel_size: The size of the pooling kernel
    :return: The gradients with all less-activated nodes replaced with 0
    """
    # Remove the residual part of parameters, since we experiment with 1-hidden-layer fully-connected NN, the overall
    # parameters are input_features * hidden_layer_size + hidden_layer_size + hidden_layer_size * output_size +
    # output_size. If cut off output_size, then it should be able to divide by hidden_layer_size. Apply pooling on the
    # reshaped gradients is equivalent to apply layer_basis pooling
    residual = grad[:, -output_size:]
    grad = grad[:, :-output_size]
    grad = grad.reshape(grad.size(0), 1, -1, n_H)
    size1 = grad.size()
    pool = torch.nn.MaxPool2d(kernel_size=kernel_size, return_indices=True)
    unpool = torch.nn.MaxUnpool2d(kernel_size=kernel_size)
    grad, idx = pool(grad)
    grad = unpool(grad, idx, output_size=size1)
    grad = grad.reshape(grad.size(0), -1)
    grad = torch.hstack([grad, residual])
    return grad

def normal_pooling(grad, n_H, output_size, kernel_size=3):
    """
    Apply pooling to given collected gradients
    :param grad: collected gradients
    :param n_H: the size of hidden layer
    :param output_size: The size of output class (to avoid vector size issue)
    :param kernel_size: The size of the pooling kernel
    :return: The gradients with all less-activated nodes dropped
    """
    pool = torch.nn.MaxPool2d(kernel_size, stride=kernel_size)
    grad = grad[:, :-output_size]
    grad = grad.reshape(grad.size(0), 1, -1, n_H)
    grad = pool(grad)
    grad = grad.reshape(grad.size(0), -1)
    return grad

def agr_filter(grad, malicious_factor=0.2, k=10,  pooling=False, n_H=64, output_size=10, mode="merge", normalize=True):
    replica = grad.clone()
    if normalize:
        grad = torch.nn.functional.normalize(grad)
    if pooling:
        # Apply pooling to the collected gradients using
        grad = normal_pooling(grad, n_H, output_size)
    selection_size = int(round(grad.size(0) * (1 - malicious_factor)))
    selected = torch.zeros(grad.size(0), dtype=torch.bool)
    if mode in ["merge", "dense"]:
        dist_matrix = torch.cdist(grad, grad)
        k_nearest = torch.topk(dist_matrix, k=k, largest=False, dim=1)
        neighbour_dist = torch.zeros(grad.size(0))
        for i in range(grad.size(0)):
            idx = k_nearest.indices[i]
            neighbour = dist_matrix[idx][:, idx]
            neighbour_dist[i] = neighbour.sum()
        dense_selected = torch.topk(neighbour_dist, largest=False, k=selection_size).indices
        if mode == "dense":
            return replica[dense_selected]
    if mode in ["merge", "cosine"]:
        cos_matrix = cosine_distance_torch(grad)
        k_nearest = torch.topk(cos_matrix, k=k, dim=1)
        neighbour_dist = torch.zeros(grad.size(0))
        for i in range(grad.size(0)):
            idx = k_nearest.indices[i]
            neighbour = cos_matrix[idx][:, idx]
            neighbour_dist[i] = neighbour.sum()
        cos_selected = torch.topk(neighbour_dist, k=selection_size).indices
        if mode == "cosine":
            return replica[cos_selected]
    if mode == "merge":
        union = torch.cat([dense_selected, cos_selected])
        uniques, count = union.unique(return_counts=True)
        selected = uniques[count > 1]
    return torch.mean(replica[selected], dim=0)

def agr_xai(grad, corrupted_count, model, testLoader, mode="merge", restore_size=False, top_p=2):
    """
        自定义实现AGR-XAI放大过程。
        :param models: 客户端模型列表（与客户端梯度一一对应）
        :param gradients: 每个客户端的接收到的梯度列表
        :param dataset_loader: 干净数据集的 DataLoader
        :param top_p: 选择的最重要特征图数量
        :param restore_size: 是否填充未选中的梯度为0
        :return: 放大后的梯度集合 G_amp
        """
    G_amp0 = []  # 最终放大的梯度集合
    G_amp1 = []  # 最终放大的梯度集合

    for g_i in grad:
        # 用客户端的梯度更新模型    #     g_i_amp = g_i.clone()  # 复制原始梯度
        load_flattened_parameters(g_i,model)  # 加载模型参数
        model.eval()  # 设置为评估模式

        # 初始化放大的梯度
        g_i_amp0 = []
        g_i_amp1 = []

        # 对干净数据集进行前向传播并计算梯度
        feature_maps = []  # 用于存储卷积层的特征图
        gradients = []  # 用于存储反向传播得到的梯度

        # 注册钩子以捕获特征图和梯度
        def forward_hook(module, input, output):
            feature_maps.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # # 假设我们对最后一层卷积层感兴趣，注册钩子
        # target_layer = model.base.conv2[0]  # 替换为实际目标层
        # forward_handle = target_layer.register_forward_hook(forward_hook)
        # backward_handle = target_layer.register_backward_hook(backward_hook)

        # 注册钩子
        target_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        handles = []
        for layer in target_layers:
            handles.append(layer.register_forward_hook(forward_hook))
            handles.append(layer.register_backward_hook(backward_hook))

        # 前向传播
        for i, (data, y) in enumerate(testLoader):
            data = data.cuda() if torch.cuda.is_available() else data
            output = model(data)

            # # 针对每个样本的最大置信度类别
            # target_scores = output.gather(1, output.argmax(dim=1, keepdim=True)).squeeze()
            #
            # # 对所有样本的目标分数求均值，确保为标量
            # target_scores = target_scores.mean()
            target_scores = output.mean()
            target_scores.backward(retain_graph=True)

            # 获取特征图和梯度
            A_k = feature_maps[-1]  # 最后一层的特征图
            dL_dAk0 = gradients[-2]  # 对特征图的梯度
            dL_dAk1 = gradients[-1]  # 对特征图的梯度

            # 计算α_k（特征图的重要性权重）
            alpha_k0 = dL_dAk0.mean(dim=(2, 3))  # 在空间维度上求平均
            alpha_k1 = dL_dAk1.mean(dim=(2, 3))  # 在空间维度上求平均

            # 按权重排序，选择top_p个特征图
            topk_indices0 = alpha_k0.topk(top_p).indices
            topk_indices1 = alpha_k1.topk(top_p).indices

            g_i_amp0.append(topk_indices0.unique())
            g_i_amp1.append(topk_indices1.unique())

        # # 注销钩子
        # forward_handle.remove()
        # backward_handle.remove()
        # 移除钩子
        for handle in handles:
            handle.remove()

        # 将放大的梯度添加到集合中
        G_amp0.append(torch.cat(g_i_amp0).unique())
        G_amp1.append(torch.cat(g_i_amp0).unique())
    G_amp0 = torch.cat(G_amp0).unique()
    G_amp1 = torch.cat(G_amp1).unique()

    """
    param_index = 0
    for param in model.parameters():
        param_size = torch.numel(param)
        if target_layers[1].weight.shape == param.shape:

        if target_layers[0].weight.shape == param.shape:


        new_param_vector = current_grads[param_index:param_index + param_size]
        new_param_tensor = new_param_vector.view(param.shape)

        param.data = new_param_tensor.clone()

        param_index += param_size
    """


    return G_amp0
    # """
    #     :param grad: List of gradients from different clients
    #     :param model: The model instance
    #     :param data_loader: DataLoader for server-side clean dataset D
    #     :param top_p: The number of top feature maps to select
    #     :param restore_size: Whether to fill the dropped gradients with 0s
    #     :return: Amplified grad collection G_amp
    #     """
    # G_amp = []  # 最终放大的梯度集合
    # cam = GradCAM(model=model, target_layers=[model.base.conv2[0]])
    # cam1 = GradCAM(model, model.modules())
    # cam2 = GradCAM(model=model, target_layers=[module for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))])
    #
    # for g_i in grad:
    #     g_i_amp = g_i.clone()  # 复制原始梯度
    #     load_flattened_parameters(g_i,model)  # 加载模型参数
    #
    #     # 遍历数据集并计算Grad-CAM权重
    #     for i, (x, y) in enumerate(testLoader):
    #         x = x.cuda() if torch.cuda.is_available() else x
    #         y = y.cuda() if torch.cuda.is_available() else y  # 如果在GPU上，确保y也在GPU上
    #
    #         # 计算Grad-CAM权重，使用真实标签y作为target_category
    #         cam_output = cam(input_tensor=x)
    #
    #         # 计算 α_k
    #         alpha_k = cam_output.mean(dim=(2, 3))  # 在空间维度上平均
    #
    #         # 按权重排序，选择 top_p 个特征图
    #         topk_indices = alpha_k.topk(top_p).indices
    #         g_i_amp = torch.zeros_like(g_i)  # 初始化放大的梯度
    #
    #         # 索引选择的梯度
    #         for idx in topk_indices:
    #             g_i_amp[idx] = g_i[idx]
    #
    #     # 如果 restore_size 为True，则填充剩余的梯度为0
    #     if restore_size:
    #         g_i_amp = torch.where(g_i == 0, torch.tensor(0.0), g_i_amp)
    #
    #     # 将放大的梯度添加到集合中
    #     G_amp.append(g_i_amp)
    #
    # if restore_size:
    #     return fang(torch.stack(G_amp), corrupted_count, model, testLoader, mode=mode)
    # return agr_filter(torch.stack(G_amp), corrupted_count, mode=mode)



def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """
    Calculate cosine distance of two vectors if x2 is given, otherwise calculate the pair-wise cosine distance of
    each line inside x1
    :param x1: the first given vector
    :param x2: the second given vector
    :param eps: the dummy minimal value to avoid division-by-zero
    :return: the pair-wise cosine similarity
    """
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def back_prop(model, testLoader, loss_function, optimizer):
    param = get_flatten_parameters(model)
    total_loss = 0.0
    count_batches = 0
    acc = 0
    num = 0
    for i, (x, y) in enumerate(testLoader):
        if type(x) == type([]):
            x[0] = x[0].to('cuda')
        else:
            x = x.to('cuda')
        y = y.to('cuda')
        output = model(x)
        acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        num += y.shape[0]

        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count_batches += 1
    total_loss /= count_batches
    acc /= num
    return acc, total_loss

def get_flatten_parameters(model):
    model_parameters = [param.view(-1) for param in model.parameters()]
    model_parameters_flattened = torch.cat(model_parameters)
    return model_parameters_flattened

def load_flattened_parameters(current_grads, model):
    param_index = 0
    for param in model.parameters():
        param_size = torch.numel(param)

        new_param_vector = current_grads[param_index:param_index + param_size]
        new_param_tensor = new_param_vector.view(param.shape)

        param.data = new_param_tensor.clone()

        param_index += param_size

defend = {DefenseTypes.Krum: krum,
          DefenseTypes.TrimmedMean: trimmed_mean, DefenseTypes.NoDefense: no_defense,
          DefenseTypes.Bulyan: bulyan, DefenseTypes.Median: median, DefenseTypes.Fang: fang,
          DefenseTypes.Agr_filter: agr_filter, DefenseTypes.Agr_xai: agr_xai}
