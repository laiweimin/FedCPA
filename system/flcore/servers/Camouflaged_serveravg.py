# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import copy
import os
import random
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
import json
import numpy as np
from flcore.clients.clientavg import clientAVG, Camouflage_clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from utils.data_utils import read_client_data
import torch
from utils import defences

class Camouflaged_FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        self.camouflage_clients=list(np.random.choice(range(args.num_clients), int(args.num_clients*args.camouflage_ratio), replace=False))
        [target_class, poison_class] = np.random.choice(self.global_model.head.out_features, replace=False, size=2)
        self.target_class = target_class
        self.poison_class = poison_class

        # select target images with witches brew
        camou_test_dataset=read_client_data(self.dataset, self.camouflage_clients[0], is_train=False)
        poison_class_index = [i for i, item in enumerate(camou_test_dataset) if item[1] == poison_class]
        target_images_index=np.random.choice(poison_class_index, args.camouflage_images_count, replace=False)
        self.target_images=torch.stack([camou_test_dataset[i][0] for i in target_images_index])

        self.set_camouflage_clients(Camouflage_clientAVG, self.camouflage_clients, target_class, poison_class, self.target_images)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        if args.camouflage_type=="CrossClient":
            print("Camouflage type: CrossClient")
            self.clients[self.camouflage_clients[0]].camouflage = 0
            print("Poison client: ", self.camouflage_clients[0])
            print("Camouflage client: ", self.camouflage_clients[1])
        else:
            print("Camouflage type: WithinClient")
            print("Camouflage clients: ", self.camouflage_clients)
        print("Target class: {}, Poison class: {}".format(self.target_class, self.poison_class))

        # self.load_model()
        self.Budget = []

        self.unlearning = args.unlearning
        self.unlearning_type = args.unlearning_type



    def train(self):
        self.save_init_global_model()
        self.Init_golbal_model=copy.deepcopy(self.global_model)
        print("save_init_global_model")
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # test poison
            if i%self.eval_gap == 0:
                self.global_model.eval()
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(target_class=self.target_class, poison_class=self.poison_class)
                target_images_prediction = self.global_model(self.target_images.cuda()).argmax(1)
                print(
                    f"Poison class: {self.poison_class}, Target class: {self.target_class}, Poisoned image prediction: {target_images_prediction}")

            for client in self.selected_clients:
                client.train(i)


            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            if self.unlearning == "FedRecovery":
                self.aggregate_parameters_with_DP(i)
            else:
                self.aggregate_parameters(i)

            self.Budget.append(time.time() - s_t)
            print(f"\n-------------Round number: {i}-------------")
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.unlearning == "NoUnlearning":
            print("Not unlearning")
        else:
            #Init unlearning
            if self.unlearning_type=="UnlearningData":
                for UnlearningId in self.camouflage_clients:
                    self.clients[UnlearningId].unlearning_data()
            elif self.unlearning_type=="UnlearningClient":
                print(f"Remove Camouflaged Client: {self.camouflage_clients[1]}")
                self.clients.remove(self.clients[self.camouflage_clients[1]])

            #Start unlearning
            if self.unlearning == "Retrain":
                print("Retrain")
                self.global_model=copy.deepcopy(self.Init_golbal_model)

                for i in range(self.global_rounds + 1):
                    s_t = time.time()
                    self.selected_clients = self.select_clients()
                    self.send_models()

                    # test poison
                    if i % self.eval_gap == 0:
                        self.global_model.eval()
                        print(f"\n-------------Round number: {i}-------------")
                        print("\nEvaluate global model")
                        self.evaluate(target_class=self.target_class, poison_class=self.poison_class)
                        target_images_prediction = self.global_model(self.target_images.cuda()).argmax(1)
                        print(
                            f"Poison class: {self.poison_class}, Target class: {self.target_class}, Poisoned image prediction: {target_images_prediction}")

                    for client in self.selected_clients:
                        client.train(i)

                    self.receive_models()
                    if self.dlg_eval and i % self.dlg_gap == 0:
                        self.call_dlg(i)
                    self.aggregate_parameters()

                    self.Budget.append(time.time() - s_t)
                    print(f"\n-------------Retrain Round number: {i}-------------")
                    print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

                    if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                        break

                print("\nBest accuracy.")
                # self.print_(max(self.rs_test_acc), max(
                #     self.rs_train_acc), min(self.rs_train_loss))
                print(max(self.rs_test_acc))
                print("\nAverage time cost per round.")
                print(sum(self.Budget[1:]) / len(self.Budget[1:]))

                self.save_results_with_unlearing()
                self.save_global_model_with_unlearning()

            if self.unlearning == "FedForgotten":
                print("FedForgotten")
                self.global_model = copy.deepcopy(self.Init_golbal_model)
                for client in self.clients:
                    client.set_FedForgotten()
                for i in range(self.global_rounds + 1):
                    s_t = time.time()
                    self.selected_clients = self.select_clients()
                    self.send_models()

                    # test poison
                    if i % self.eval_gap == 0:
                        self.global_model.eval()
                        print(f"\n-------------Round number: {i}-------------")
                        print("\nEvaluate global model")
                        self.evaluate(target_class=self.target_class, poison_class=self.poison_class)
                        target_images_prediction = self.global_model(self.target_images.cuda()).argmax(1)
                        print(
                            f"Poison class: {self.poison_class}, Target class: {self.target_class}, Poisoned image prediction: {target_images_prediction}")

                    for client in self.selected_clients:
                        client.FedForgotten(i)

                    self.receive_models()
                    if self.dlg_eval and i % self.dlg_gap == 0:
                        self.call_dlg(i)
                    self.aggregate_parameters()

                    self.Budget.append(time.time() - s_t)
                    print(f"\n-------------FedForgotten Round number: {i}-------------")
                    print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

                    if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                        break

                print("\nBest accuracy.")
                # self.print_(max(self.rs_test_acc), max(
                #     self.rs_train_acc), min(self.rs_train_loss))
                print(max(self.rs_test_acc))
                print("\nAverage time cost per round.")
                print(sum(self.Budget[1:]) / len(self.Budget[1:]))

                self.save_results_with_unlearing()
                self.save_global_model_with_unlearning()

            if self.unlearning == "FedRecovery":
                print("FedRecovery")
                gradients_of_global_model = []
                gradients_of_local_model = []

                for i in range(self.global_rounds+1):
                    gradients_of_global_model.append(self.global_flattened_parameters_logs[i+1] - self.global_flattened_parameters_logs[i])
                    gradients_of_local_model.append(self.flattened_parameters_logs[i] - self.global_flattened_parameters_logs[i])

                squared_norms = torch.tensor([torch.norm(grad) ** 2 for grad in gradients_of_global_model]).cuda()
                sum_squared_norms = torch.sum(squared_norms)
                p_i = squared_norms / sum_squared_norms
                gradient_residuals=[]

                if self.unlearning_type == "UnlearningData":
                    for i in range(self.global_rounds+1):
                        sum_gradients = torch.sum(gradients_of_local_model[i], dim=0) - gradients_of_local_model[i][
                            self.camouflage_clients[0]] - gradients_of_local_model[i][self.camouflage_clients[1]]
                        average_gradients = sum_gradients / (self.num_clients - 2)
                        gradient_residuals_i = 1 / self.num_clients * (
                                    average_gradients - gradients_of_local_model[i][self.camouflage_clients[0]] / 2 -
                                    gradients_of_local_model[i][self.camouflage_clients[1]] / 2)
                        gradient_residuals.append(gradient_residuals_i)

                    noise = torch.normal(0, self.last_sigma * self.last_sigma, size=self.global_flattened_parameters_logs[-1].shape).cuda()

                    Recovery_parameters=self.global_flattened_parameters_logs[-1] - torch.sum(torch.stack(gradient_residuals) * p_i[:, None], dim=0)

                    Recovery_parameters = Recovery_parameters + noise

                    param_index = 0
                    for server_param in self.global_model.parameters():
                        param_size = torch.numel(server_param)
                        new_param_vector = Recovery_parameters[param_index:param_index + param_size]
                        new_param_tensor = new_param_vector.view(server_param.shape)
                        server_param.data = new_param_tensor.clone()
                        param_index += param_size
                    for i in range(20):
                        self.selected_clients = [self.clients[self.camouflage_clients[0]], self.clients[self.camouflage_clients[1]]]
                        self.send_models()
                        for client in self.selected_clients:
                            client.train(self.global_rounds-1)
                        uploaded_models=[]
                        for client in self.selected_clients:
                            uploaded_models.append(client.model)
                        # self.receive_models()
                        flattened_parameters = []
                        for model in uploaded_models:
                            model_parameters = [param.view(-1) for param in model.parameters()]
                            model_parameters_flattened = torch.cat(model_parameters)
                            flattened_parameters.append(model_parameters_flattened)
                        flattened_parameters = torch.stack(flattened_parameters)
                        Recovery_parameters = Recovery_parameters *4 / 5 + torch.mean(flattened_parameters, dim=0) / 5
                        param_index = 0
                        for server_param in self.global_model.parameters():
                            param_size = torch.numel(server_param)
                            new_param_vector = Recovery_parameters[param_index:param_index + param_size]
                            new_param_tensor = new_param_vector.view(server_param.shape)
                            server_param.data = new_param_tensor.clone()
                            param_index += param_size

                elif self.unlearning_type=="UnlearningClient":
                    for i in range(self.global_rounds+1):
                        sum_gradients = torch.sum(gradients_of_local_model[i], dim=0) - gradients_of_local_model[i][self.camouflage_clients[1]]
                        average_gradients = sum_gradients / (self.num_clients - 1)
                        gradient_residuals_i = 1 / self.num_clients * (
                                    average_gradients  - gradients_of_local_model[i][self.camouflage_clients[1]] )
                        gradient_residuals.append(gradient_residuals_i)

                    Recovery_parameters = self.global_flattened_parameters_logs[-1] - torch.sum(
                        torch.stack(gradient_residuals) * p_i[:, None], dim=0)

                    noise = torch.normal(0, self.last_sigma * self.last_sigma,
                                         size=self.global_flattened_parameters_logs[-1].shape).cuda()

                    Recovery_parameters = Recovery_parameters + noise

                    param_index = 0
                    for server_param in self.global_model.parameters():
                        param_size = torch.numel(server_param)
                        new_param_vector = Recovery_parameters[param_index:param_index + param_size]
                        new_param_tensor = new_param_vector.view(server_param.shape)
                        server_param.data = new_param_tensor.clone()
                        param_index += param_size

                print(f"\n-------------FedRecovery Evaluate -------------")
                self.global_model.eval()
                self.evaluate(target_class=self.target_class, poison_class=self.poison_class)
                target_images_prediction = self.global_model(self.target_images.cuda()).argmax(1)
                print(
                    f"Poison class: {self.poison_class}, Target class: {self.target_class}, Poisoned image prediction: {target_images_prediction}")

                self.save_results_with_unlearing()
                self.save_global_model_with_unlearning()

            if self.unlearning == "FedEraser":
                print("FedEraser")
                for client in self.clients:
                    if client.id == self.camouflage_clients[0] or client.id == self.camouflage_clients[1]:
                        continue
                    client.local_epochs = client.local_epochs // 2
                unlearn_global_models = list()
                unlearn_global_models.append(self.global_flattened_parameters_logs[0])
                curr_model = self.global_flattened_parameters_logs[0] * 0
                for i in range(len(self.flattened_parameters_logs[0])):
                    if self.unlearning_type=="UnlearningData" and i in self.camouflage_clients:
                        continue
                    if self.unlearning_type=="UnlearningClient" and i == self.camouflage_clients[1]:
                        continue
                    if self.unlearning_type == "UnlearningData":
                        curr_model += (self.flattened_parameters_logs[0][i] / (self.num_clients - len(self.camouflage_clients)))
                    elif self.unlearning_type == "UnlearningClient":
                        curr_model += (self.flattened_parameters_logs[0][i] / (self.num_clients - 1))
                unlearn_global_models.append(curr_model)
                for i in range(1, self.global_rounds + 1):
                    # set global_model = unlearn_global_models[i]
                    param_index = 0
                    for server_param in self.global_model.parameters():
                        param_size = torch.numel(server_param)
                        new_param_vector = unlearn_global_models[-1][param_index:param_index + param_size]
                        new_param_tensor = new_param_vector.view(server_param.shape)
                        server_param.data = new_param_tensor.clone()
                        param_index += param_size

                    self.selected_clients = []
                    for c in self.clients:
                        if self.unlearning_type == "UnlearningData":
                            if c.id == self.camouflage_clients[0] or c.id == self.camouflage_clients[1]:
                                continue
                            else:
                                self.selected_clients.append(self.clients[c.id])
                        elif c.id > self.camouflage_clients[1]:
                            self.selected_clients.append(self.clients[c.id - 1])
                        else:
                            self.selected_clients.append(self.clients[c.id])
                    self.send_models()
                    for client in self.selected_clients:
                        client.train(self.global_rounds)
                    uploaded_models = []
                    for client in self.selected_clients:
                        uploaded_models.append(client.model)
                    flattened_parameters = []
                    for model in uploaded_models:
                        model_parameters = [param.view(-1) for param in model.parameters()]
                        model_parameters_flattened = torch.cat(model_parameters)
                        flattened_parameters.append(model_parameters_flattened)
                    flattened_parameters = torch.stack(flattened_parameters)
                    newGlobalM = unlearn_global_models[-1]
                    oldGlobalM = self.global_flattened_parameters_logs[i+1]

                    if self.args.defense == 'TrimmedMean':
                        newClientM = defences.trimmed_mean(flattened_parameters, 0.2)
                    elif self.args.defense == 'Krum':
                        newClientM = defences.krum(flattened_parameters, len(flattened_parameters), 2)
                    elif self.args.defense == 'Bulyan':
                        newClientM = defences.bulyan(flattened_parameters, len(flattened_parameters), 2)
                    elif self.args.defense == 'Median':
                        newClientM = defences.median(flattened_parameters)
                    else:
                        newClientM = torch.mean(flattened_parameters, dim=0)

                    oldClientM = newClientM * 0
                    for j in range(len(self.flattened_parameters_logs[i])):
                        if j == self.camouflage_clients[0] or j == self.camouflage_clients[1]:
                            continue
                        oldClientM += (self.flattened_parameters_logs[i][j] / (
                                    self.num_clients - len(self.camouflage_clients)))
                    old_param_update = oldClientM - oldGlobalM
                    new_param_update = newClientM - newGlobalM
                    step_length = torch.norm(old_param_update)
                    step_direction = new_param_update / torch.norm(new_param_update)
                    return_model_state = newGlobalM + step_length * step_direction
                    if self.unlearning_type == "UnlearningData":
                        param_index = 0
                        for server_param in self.global_model.parameters():
                            param_size = torch.numel(server_param)
                            new_param_vector = return_model_state[param_index:param_index + param_size]
                            new_param_tensor = new_param_vector.view(server_param.shape)
                            server_param.data = new_param_tensor.clone()
                            param_index += param_size

                        self.selected_clients = [self.clients[self.camouflage_clients[0]],
                                                 self.clients[self.camouflage_clients[1]]]
                        self.send_models()
                        for client in self.selected_clients:
                            client.train(self.global_rounds - 1)
                        uploaded_models = []
                        for client in self.selected_clients:
                            uploaded_models.append(client.model)
                        # self.receive_models()
                        flattened_parameters = []
                        for model in uploaded_models:
                            model_parameters = [param.view(-1) for param in model.parameters()]
                            model_parameters_flattened = torch.cat(model_parameters)
                            flattened_parameters.append(model_parameters_flattened)
                        flattened_parameters = torch.stack(flattened_parameters)
                        return_model_state = return_model_state * 4 / 5 + torch.mean(flattened_parameters, dim=0) / 5

                    unlearn_global_models.append(return_model_state.clone())

                param_index = 0
                for server_param in self.global_model.parameters():
                    param_size = torch.numel(server_param)
                    new_param_vector = unlearn_global_models[-1][param_index:param_index + param_size]
                    new_param_tensor = new_param_vector.view(server_param.shape)
                    server_param.data = new_param_tensor.clone()
                    param_index += param_size

                # test poison
                self.global_model.eval()
                target_images_prediction = self.global_model(self.target_images.cuda()).argmax(1)
                print("\nEvaluate global model")
                self.evaluate(target_class=self.target_class, poison_class=self.poison_class)
                print(f"Poison class: {self.poison_class}, Target class: {self.target_class}, Poisoned image prediction: {target_images_prediction}")

                print("\nBest accuracy.")
                # self.print_(max(self.rs_test_acc), max(
                #     self.rs_train_acc), min(self.rs_train_loss))
                print(max(self.rs_test_acc))
                print("\nAverage time cost per round.")
                print(sum(self.Budget[1:]) / len(self.Budget[1:]))

                self.save_results_with_unlearing()
                self.save_global_model_with_unlearning()

            if self.unlearning == "FedRecover":
                print("FedRecover")
                param_index = 0
                bs=2
                recover_golbal_model_logs = []
                exact_clients_model_delta_logs = []
                exact_global_model_delta_logs = []
                recover_golbal_model_logs.append(self.global_flattened_parameters_logs[0])

                for i in range(self.global_rounds+1):
                    s_t = time.time()
                    param_index = 0
                    for server_param in self.global_model.parameters():
                        param_size = torch.numel(server_param)
                        new_param_vector = recover_golbal_model_logs[-1][param_index:param_index + param_size]
                        new_param_tensor = new_param_vector.view(server_param.shape)
                        server_param.data = new_param_tensor.clone()
                        param_index += param_size
                    self.send_models()
                    if i <= 3 or i % 5 == 0 or i >= self.global_rounds-2:
                        self.selected_clients = self.select_clients()
                        exact_clients_model_delta=[]
                        exact_clients_model = []
                        for client in self.selected_clients:
                            client.train(i)
                            model_parameters = [param.view(-1) for param in client.model.parameters()]
                            model_parameters_flattened = torch.cat(model_parameters)
                            exact_clients_model.append(model_parameters_flattened)
                            exact_clients_model_delta.append(recover_golbal_model_logs[-1] - model_parameters_flattened)
                        exact_clients_model = torch.stack(exact_clients_model)
                        exact_clients_model_delta= torch.stack(exact_clients_model_delta)

                        exact_clients_model_delta_logs.append(exact_clients_model_delta.clone())

                        if self.args.defense == 'TrimmedMean':
                            mean_estimated_update = defences.trimmed_mean(exact_clients_model, 0.2)
                        elif self.args.defense == 'Krum':
                            mean_estimated_update = defences.krum(exact_clients_model, len(exact_clients_model), 2)
                        elif self.args.defense == 'Bulyan':
                            mean_estimated_update = defences.bulyan(exact_clients_model, len(exact_clients_model),2)
                        elif self.args.defense == 'Median':
                            mean_estimated_update = defences.median(exact_clients_model)
                        else:
                            mean_estimated_update = torch.mean(exact_clients_model, dim=0)
                        recover_golbal_model_logs.append(mean_estimated_update.clone())
                        exact_global_model_delta_logs.append(recover_golbal_model_logs[-1] - recover_golbal_model_logs[-2])
                    else:
                        Delta_W = torch.stack(exact_global_model_delta_logs[-2:])
                        estimated_update=[]
                        for c in self.clients:
                            if self.unlearning_type=="UnlearningData" and c.id in self.camouflage_clients:
                                c.train(i)
                                model_parameters = [param.view(-1) for param in c.model.parameters()]
                                model_parameters_flattened = torch.cat(model_parameters)
                                estimated_client_update = recover_golbal_model_logs[-1] - model_parameters_flattened
                                print(f"Epoch{i}, CamouClient {c.id} is retrined")
                                estimated_update.append(estimated_client_update.clone())
                            if self.unlearning_type=="UnlearningClient" and c.id == self.camouflage_clients[1]:
                                continue
                            else:
                                if c.id > self.camouflage_clients[1]:
                                    Delta_G = torch.stack((exact_clients_model_delta_logs[-2:][0][c.id-1],exact_clients_model_delta_logs[-2:][1][c.id-1]),dim=0)
                                else:
                                    Delta_G = torch.stack((exact_clients_model_delta_logs[-2:][0][c.id],exact_clients_model_delta_logs[-2:][1][c.id]),dim=0)
                                A = torch.matmul(Delta_W, Delta_G.T)
                                D = torch.diag(torch.diag(A))
                                L = torch.tril(A, diagonal=-1)
                                sigma = torch.dot(Delta_G[-2].T, Delta_W[-2]) / torch.dot(Delta_W[-2].T, Delta_W[-2])
                                M1 = torch.cat((-D, L.T), dim=1)
                                M2 = torch.cat((L, sigma * torch.matmul(Delta_W, Delta_W.T)), dim=1)
                                M = torch.cat((M1, M2), dim=0)
                                v=recover_golbal_model_logs[-1] - self.global_flattened_parameters_logs[i]
                                b = torch.cat((torch.matmul(Delta_G, v), sigma * torch.matmul(Delta_W, v)))
                                p = torch.matmul(torch.inverse(M),b.unsqueeze(1))
                                H_v = sigma * v - torch.matmul(torch.cat((Delta_G, sigma * Delta_W), dim=0).T, p).squeeze()
                                original_client_update = self.global_flattened_parameters_logs[i]-self.flattened_parameters_logs[i][c.id]
                                estimated_client_update = original_client_update+H_v

                                alpha = 1e-5
                                num_elements = len(estimated_client_update)
                                k = int((1-alpha) * num_elements)
                                tau = torch.kthvalue(original_client_update.abs(), k).values
                                print(f"============tau:{tau}===norm:{torch.norm(estimated_client_update, p=float('inf'))}================")
                                if torch.norm(estimated_client_update, p=float('inf')) > tau:
                                    c.train(i)
                                    model_parameters = [param.view(-1) for param in c.model.parameters()]
                                    model_parameters_flattened = torch.cat(model_parameters)
                                    estimated_client_update = recover_golbal_model_logs[-1] - model_parameters_flattened
                                    print(f"Epoch{i}, Client {c.id} is retrined")
                                estimated_update.append(estimated_client_update.clone())
                        estimated_update = torch.stack(estimated_update)
                        if self.args.defense == 'TrimmedMean':
                            mean_estimated_update = defences.trimmed_mean(estimated_update, 0.2)
                        elif self.args.defense == 'Krum':
                            mean_estimated_update = defences.krum(estimated_update, len(estimated_update), 2)
                        elif self.args.defense == 'Bulyan':
                            mean_estimated_update = defences.bulyan(estimated_update, len(estimated_update),2)
                        elif self.args.defense == 'Median':
                            mean_estimated_update = defences.median(estimated_update)
                        else:
                            mean_estimated_update = torch.mean(estimated_update, dim=0)
                        recover_golbal_model_logs.append(recover_golbal_model_logs[-1]-mean_estimated_update)

                    if i % self.eval_gap == 0:
                        self.global_model.eval()
                        print(f"\n-------------Round number: {i}-------------")
                        print("\nEvaluate global model")
                        self.evaluate(target_class=self.target_class, poison_class=self.poison_class)
                        target_images_prediction = self.global_model(self.target_images.cuda()).argmax(1)
                        print(
                            f"Poison class: {self.poison_class}, Target class: {self.target_class}, Poisoned image prediction: {target_images_prediction}")

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
