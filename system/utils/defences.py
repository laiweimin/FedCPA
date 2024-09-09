import numpy as np
from collections import defaultdict
import torch

class DefenseTypes:
    NoDefense = 'NoDefense'
    Krum = 'Krum'
    TrimmedMean = 'TrimmedMean'
    Bulyan = 'Bulyan'
    Median = 'Median'

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

defend = {DefenseTypes.Krum: krum,
          DefenseTypes.TrimmedMean: trimmed_mean, DefenseTypes.NoDefense: no_defense,
          DefenseTypes.Bulyan: bulyan, DefenseTypes.Median: median}
