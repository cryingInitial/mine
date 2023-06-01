# When we make a new one, we should inherit the Finetune class.
import logging
import copy
from operator import attrgetter
import time
import datetime
import random
import numpy as np
import torch
import pickle
import math

import torch.nn as nn
import torch.nn.functional as F

from methods.cl_manager import CLManagerBase, MemoryBase
from utils.data_loader import cutmix_data, MultiProcessLoader
from utils import autograd_hacks
logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")


class Ours(CLManagerBase):
    def __init__(
            self, train_datalist, test_datalist, device, **kwargs
    ):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        
        # for ours
        self.T = kwargs["temperature"]
        self.corr_warm_up = kwargs["corr_warm_up"]
        self.target_layer = kwargs["target_layer"]
        self.selected_num = 512
        self.corr_map = {}
        self.count_decay_ratio = kwargs["count_decay_ratio"]
        self.k_coeff = kwargs["k_coeff"]
        
        super().__init__(
            train_datalist, test_datalist, device, **kwargs
        )
        self.past_dist_dict = {}
        self.class_std_list = []
        self.features = None
        self.sample_std_list = []
        self.sma_class_loss = {}
        self.normalized_dict = {}
        self.freeze_idx = []
        self.add_new_class_time = []
        self.ver = kwargs["version"]
        self.avg_prob = kwargs["avg_prob"]
        self.weight_option = kwargs["weight_option"]
        self.weight_method = kwargs["weight_method"]
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.prev_weight_list = None
        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]
        self.ema_ratio = kwargs['ema_ratio']
        self.weight_ema_ratio = kwargs["weight_ema_ratio"]
        self.use_batch_cutmix = kwargs["use_batch_cutmix"]
        self.device = device
        self.klass_warmup = kwargs["klass_warmup"]
        self.loss_balancing_option = kwargs["loss_balancing_option"]
        self.grad_cls_score_mavg = {}
        self.corr_map_list = []
        self.sample_count_list = []
        self.labels_list=[]
        self._supported_layers = ['Linear', 'Conv2d']
        self.freeze_warmup = 500
        self.grad_dict = {}

        # for gradient subsampling
        
        # self.grad_mavg_base = {n: torch.zeros_like(p) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_mavgsq_base = {n: torch.zeros_like(p) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_mvar_base = {n: torch.zeros_like(p) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_cls_score_mavg_base = {n: 0 for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_criterion_base = {n: 0 for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_dict_base = {n: [] for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.selected_num = 512
        # print("keys")
        # print(self.grad_mavg_base.keys())
        # self.selected_mask = {}
        # for key in self.grad_mavg_base.keys():
        #     a = self.grad_mavg_base[key].flatten()
        #     selected_indices = torch.randperm(len(a))[:self.selected_num]
        #     self.selected_mask[key] = selected_indices
        #     self.grad_mavg_base[key] = torch.zeros(self.selected_num).to(self.device)
        #     self.grad_mavgsq_base[key] = torch.zeros(self.selected_num).to(self.device)
        #     self.grad_mvar_base[key] = torch.zeros(self.selected_num).to(self.device)
        #
        # print("self.selected_mask.keys()")
        # print(self.selected_mask.keys())

        self.last_grad_mean = 0.0

        self.grad_mavg = []
        self.grad_mavgsq = []
        self.grad_mvar = []
        self.grad_criterion = []
        
        self.grad_ema_ratio = 0.01

        # Information based freezing
        self.unfreeze_rate = kwargs["unfreeze_rate"]
        # Information based freezing
        self.fisher_ema_ratio = 0.01
        if self.model_name == 'resnet18':
            self.num_blocks = 9
            self.fisher = [0.0 for _ in range(9)]
        else:
            raise NotImplementedError("Layer blocks for Fisher Information calculation not defined")
        self.cumulative_fisher = []
        self.frozen = False

        self.cumulative_fisher = []

        self.klass_train_warmup = kwargs["klass_train_warmup"]

        self.recent_ratio = kwargs["recent_ratio"]
        self.gpu_transform = kwargs["gpu_transform"]
        self.use_kornia = kwargs["use_kornia"]
        self.use_amp =  False #kwargs["use_amp"]
        self.cls_weight_decay = kwargs["cls_weight_decay"]
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # for name, p in self.model.named_parameters():
        #     print(name, p.shape)


    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = OurMemory(self.memory_size, self.T, self.count_decay_ratio, self.k_coeff, self.device)

        self.grad_score_per_layer = None

        if self.target_layer == "whole_conv2":
            self.target_layers = ["group1.blocks.block0.conv2.block.0.weight", "group1.blocks.block1.conv2.block.0.weight", "group2.blocks.block0.conv2.block.0.weight", "group2.blocks.block1.conv2.block.0.weight", "group3.blocks.block0.conv2.block.0.weight", "group3.blocks.block1.conv2.block.0.weight", "group4.blocks.block0.conv2.block.0.weight", "group4.blocks.block1.conv2.block.0.weight"]
        elif self.target_layer == "last_conv2":
            self.target_layers = ["group1.blocks.block1.conv2.block.0.weight", "group2.blocks.block1.conv2.block.0.weight", "group3.blocks.block1.conv2.block.0.weight", "group4.blocks.block1.conv2.block.0.weight"]

        autograd_hacks.add_hooks(self.model)
        self.selected_mask = {}
        self.grad_mavg_base = {n: torch.zeros_like(p) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        self.param_count = {n: int((p.numel()) * 0.0005) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        #print("self.param_count")
        #print(self.param_count)
        for key in self.grad_mavg_base.keys():
            a = self.grad_mavg_base[key].flatten()
            #selected_indices = torch.randperm(len(a))[:self.selected_num]
            selected_indices = torch.randperm(len(a))[:self.param_count[key]]
            self.selected_mask[key] = selected_indices
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.sim_matrix = torch.zeros([0, 0]).to(self.device)
        self.self_cls_sim = 0.0
        self.other_cls_sim = 0.0

        self.waiting_batch = []
        for i in range(self.future_steps):
            self.load_batch()

    def generate_waiting_batch(self, iterations, similarity_matrix=None):
        for i in range(iterations):
            self.waiting_batch.append(self.temp_future_batch + self.memory.retrieval(self.memory_batch_size, similarity_matrix=similarity_matrix))

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            self.future_add_new_class()
        self.update_memory(sample, self.future_sample_num)
        self.future_num_updates += self.online_iter

        if  self.future_num_updates >= 1:
            if self.future_sample_num >= self.corr_warm_up:
                self.generate_waiting_batch(int(self.future_num_updates), self.sim_matrix)
            else:
                self.generate_waiting_batch(int(self.future_num_updates))
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def _layer_type(self, layer: nn.Module) -> str:
        return layer.__class__.__name__

    def prev_check(self, idx):
        result = True
        for i in range(idx):
            if i not in self.freeze_idx:
                result = False
                break
        return result

    def unfreeze_layers(self):
        self.frozen = False
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def freeze_layers(self):
        if len(self.freeze_idx) > 0:
            self.frozen = True
        for i in self.freeze_idx:
            if i==0:
                # freeze initial block
                for name, param in self.model.named_parameters():
                    if "initial" in name:
                        param.requires_grad = False
                continue
            self.freeze_layer((i-1)//2, (i-1)%2)

    def freeze_layer(self, layer_index, block_index=None):
        # group(i)가 들어간 layer 모두 freeze
        if self.target_layer == "last_conv2":
            group_name = "group" + str(layer_index)
        elif self.target_layer == "whole_conv2":
            group_name = "group" + str(layer_index) + ".blocks.block"+str(block_index)

        # print("freeze", group_name)
        for name, param in self.model.named_parameters():
            if group_name in name:
                param.requires_grad = False

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample)
            self.writer.add_scalar(f"train/add_new_class", 1, sample_num)
            self.add_new_class_time.append(sample_num)
            # print("seed", self.rnd_seed, "dd_new_class_time")
            # print(self.add_new_class_time)
        else:
            self.writer.add_scalar(f"train/add_new_class", 0, sample_num)
        
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()


    def future_add_new_class(self):
        # n = self.sim_matrix.size(0)
        # if n > 0:
        #     prev_sim_matrix = copy.deepcopy(self.sim_matrix)
        #     diagonal_avg = torch.diag(self.sim_matrix).mean()
        #     if n > 1:
        #         off_diagonal_avg = (torch.mean(self.sim_matrix) - diagonal_avg/n)*n/(n-1)
        #     else:
        #         off_diagonal_avg = 0
        #     self.sim_matrix = torch.ones([n+1, n+1]).to(self.device)*off_diagonal_avg
        #     self.sim_matrix[n, n] = diagonal_avg
        #     self.sim_matrix[0:n, 0:n] = prev_sim_matrix
        # else:
        #     self.sim_matrix = torch.zeros([1, 1]).to(self.device)
        self.sim_matrix = torch.ones([len(self.memory.cls_list), len(self.memory.cls_list)]).to(self.device)*self.other_cls_sim
        self.sim_matrix.fill_diagonal_(self.self_cls_sim)
        ### for calculating similarity ###
        # len_key = len(self.corr_map.keys())
        # if len_key > 1:
        #     total_corr = 0.0
        #     total_corr_count = 0
        #     self_corr_count = 0
        #     for i in range(len_key):
        #         for j in range(i+1, len_key):
        #             if self.corr_map[i][j] is not None:
        #                 total_corr += self.corr_map[i][j]
        #                 total_corr_count += 1
        #     if total_corr_count >= 1:
        #         self.initial_corr = total_corr / (total_corr_count)
        #     else:
        #         self.initial_corr = 0.0
        #     self_corr = 0.0
        #     for i in range(len_key):
        #         if self.corr_map[i][i] is not None:
        #             self_corr += self.corr_map[i][i]
        #             self_corr_count += 1
        #     if self_corr_count >= 1:
        #         self_corr_avg = total_corr / (self_corr_count+1e-10)
        #     else:
        #         self_corr_avg = 0.5
        # else:
        #     self.initial_corr = None
        #
        # for i in range(len_key):
        #     # 모든 class의 avg_corr로 initialize
        #     self.corr_map[i][len_key] = self.initial_corr
        #
        # # 자기 자신은 1로 initialize
        # self.corr_map[len_key] = {}
        # if len_key > 1:
        #     self.corr_map[len_key][len_key] = self_corr_avg
        # else:
        #     self.corr_map[len_key][len_key] = None


    def add_new_class(self, class_name, sample=None):
        # print("!!add_new_class seed", self.rnd_seed)
        self.cls_dict[class_name] = len(self.exposed_classes)
        
        # self.grad_cls_score_mavg[len(self.exposed_classes)] = copy.deepcopy(self.grad_cls_score_mavg_base)
        # self.grad_dict[len(self.exposed_classes)] = copy.deepcopy(self.grad_dict_base)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        prev_bias = copy.deepcopy(self.model.fc.bias.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
                self.model.fc.bias[:self.num_learned_class - 1] = prev_bias
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

        autograd_hacks.remove_hooks(self.model)
        autograd_hacks.add_hooks(self.model)
        
        # for unfreezing model

        # initialize with mean
        # if len(self.grad_mavg) >= 2:
        #     self.grad_mavg_base = {key: torch.mean(torch.stack([self.grad_mavg[kls][key] for kls in range(len(self.grad_mavg))]), dim=0) for key in self.grad_mavg_base.keys()}
        #     self.grad_mavgsq_base = {key: torch.mean(torch.stack([self.grad_mavgsq[kls][key] for kls in range(len(self.grad_mavg))]), dim=0) for key in self.grad_mavgsq_base.keys()}
        #     self.grad_mvar_base = {key: torch.mean(torch.stack([self.grad_mvar[kls][key] for kls in range(len(self.grad_mavg))]), dim=0) for key in self.grad_mvar_base.keys()}
        #
        # self.grad_mavg.append(copy.deepcopy(self.grad_mavg_base))
        # self.grad_mavgsq.append(copy.deepcopy(self.grad_mavgsq_base))
        # self.grad_mvar.append(copy.deepcopy(self.grad_mvar_base))
        # self.grad_criterion.append(copy.deepcopy(self.grad_criterion_base))
        
        
        # ### update similarity map ###
        # len_key = len(self.corr_map.keys())
        # if len_key > 1:
        #     total_corr = 0.0
        #     total_corr_count = 0
        #     for i in range(len_key):
        #         for j in range(i+1, len_key):
        #             total_corr += self.corr_map[i][j]
        #             total_corr_count += 1
        #     self.initial_corr = total_corr / total_corr_count
        # else:
        #     self.initial_corr = None
        #
        # for i in range(len_key):
        #     # 모든 class의 avg_corr로 initialize
        #     self.corr_map[i][len_key] = self.initial_corr
        #
        # # 자기 자신은 1로 initialize
        # self.corr_map[len_key] = {}
        # self.corr_map[len_key][len_key] = None
        #print("self.corr_map")
        #print(self.corr_map)

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            # print("y")
            # print(y)
            #self.before_model_update()
            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x, y)

            if self.train_count > 2:
                self.get_freeze_idx(logit.detach(), y)
                if np.random.rand() > self.unfreeze_rate:
                    self.freeze_layers()

            _, preds = logit.topk(self.topk, 1, True, True)


            if self.use_amp:
                self.scaler.scale(loss).backward()
                autograd_hacks.compute_grad1(self.model)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                autograd_hacks.compute_grad1(self.model)
                self.optimizer.step()

            # loss.backward()
            # autograd_hacks.compute_grad1(self.model)
            #
            # self.optimizer.step()
            #self.update_gradstat(self.sample_num, y)
            
            if self.sample_num >= 2:
                self.update_correlation(y)
            print(self.sim_matrix)

            if not self.frozen:
                self.calculate_fisher()

            autograd_hacks.clear_backprops(self.model)
            
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            
            if len(self.freeze_idx) == 0:    
                # forward와 backward가 full로 일어날 때
                self.total_flops += (len(y) * (self.forward_flops + self.backward_flops))
            else:
                self.total_flops += (len(y) * (self.forward_flops + self.get_backward_flops()))
                
            # print("total_flops", self.total_flops)
            # self.writer.add_scalar(f"train/total_flops", self.total_flops, self.sample_num)

            self.unfreeze_layers()
            self.freeze_idx = []
            self.after_model_update()

        # print("self.corr_map")
        # print(self.corr_map)

        return total_loss / iterations, correct / num_data

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        # self.corr_map_list.append(copy.deepcopy(self.corr_map))
        # self.sample_count_list.append(copy.deepcopy(self.memory.usage_count))
        # self.labels_list.append(copy.deepcopy(self.memory.labels))
        #
        # # store한 애들 저장
        # corr_map_name = "corr_map_list_T_" + str(self.T) + "_decay_" + str(self.k_coeff) + ".pickle"
        # sample_count_name = "sample_count_list_T_" + str(self.T) + "_decay_" + str(self.k_coeff) + ".pickle"
        # labels_list_name = "labels_list_T_" + str(self.T) + "_decay_" + str(self.k_coeff) + ".pickle"
        #
        # # print("corr_map_name", corr_map_name)
        # # print("sample_count_name", sample_count_name)
        # # print("labels_list_name", labels_list_name)
        #
        # with open(corr_map_name, 'wb') as f:
        #     pickle.dump(self.corr_map_list, f, pickle.HIGHEST_PROTOCOL)
        #
        # with open(sample_count_name, 'wb') as f:
        #     pickle.dump(self.sample_count_list, f, pickle.HIGHEST_PROTOCOL)
        #
        # with open(labels_list_name, 'wb') as f:
        #     pickle.dump(self.labels_list, f, pickle.HIGHEST_PROTOCOL)
        
        return super().online_evaluate(test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time)


    def after_model_update(self):
        self.train_count += 1

    def get_backward_flops(self):
        backward_flops = self.backward_flops
        if self.frozen:
            for i in self.freeze_idx:
                backward_flops -= self.comp_backward_flops[i]
        return backward_flops

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b) # 4
                self.total_flops += (len(logit) * 4) / 10e9
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.total_flops += (len(logit) * 2) / 10e9
        return logit, loss

    def update_memory(self, sample, sample_num):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace]
            idx_to_replace = random.choice(cand_idx)
            self.memory.replace_sample(sample, sample_num, idx_to_replace)
        else:
            self.memory.replace_sample(sample, sample_num)

    @torch.no_grad()
    def update_correlation(self, labels):
        current_corr_map = copy.deepcopy(self.corr_map)
        curr_corr_key_list = list(current_corr_map.keys())

        # n_classes = self.sim_matrix.size(0)
        # batch_size = len(labels)
        # update_matrix = torch.zeros_like(self.sim_matrix).flatten()
        # labelcount_matrix = torch.zeros_like(self.sim_matrix).flatten()
        selected_grads = []
        for n, p in self.model.named_parameters():
            if p.requires_grad is True and p.grad is not None and n in self.selected_mask.keys():
                selected_grads.append(p.grad1.detach().clone().clamp(-1000, 1000).flatten(start_dim=1)[:, self.selected_mask[n]])
        stacked_grads = torch.cat(selected_grads, dim=1)
        similarity_matrix = F.cosine_similarity(stacked_grads.unsqueeze(1), stacked_grads.unsqueeze(0), dim=2)
        self.total_flops += stacked_grads.shape[0]*stacked_grads.shape[0]*2*stacked_grads.shape[1]/10e9

        unique_labels, idxs = torch.unique(labels, sorted=True, return_inverse=True)
        same_labels = labels.unsqueeze(1) == labels.unsqueeze(0)
        diff_labels = ~same_labels
        same_labels.fill_diagonal_(False)
        if torch.any(same_labels):
            self.self_cls_sim += self.grad_ema_ratio*(torch.mean(similarity_matrix[same_labels]) - self.self_cls_sim)
        if torch.any(diff_labels):
            self.other_cls_sim += self.grad_ema_ratio*(torch.mean(similarity_matrix[diff_labels]) - self.other_cls_sim)
        self.sim_matrix = torch.ones([len(self.memory.cls_list), len(self.memory.cls_list)]).to(self.device)*self.other_cls_sim
        self.sim_matrix.fill_diagonal_(self.self_cls_sim)
        
        # for i, label1 in enumerate(unique_labels):
        #     for j, label2 in enumerate(unique_labels[i:]):
        #         if label1 == label2:
        #             num_elements = (idxs == i).sum()
        #             if num_elements > 1:
        #                 label_matrix = similarity_matrix[idxs == i][:, idxs == i+j]
        #                 non_overlap_idx = torch.triu_indices(num_elements, num_elements, 1)
        #                 self.sim_matrix[label1][label2] += self.grad_ema_ratio * (
        #                             (label_matrix[non_overlap_idx[0], non_overlap_idx[1]]).mean() - self.sim_matrix[label1][label2])
        #         else:
        #             self.sim_matrix[label1][label2] += self.grad_ema_ratio *(
        #                     (similarity_matrix[idxs == i][:, idxs == i+j]).mean() - self.sim_matrix[label1][label2])

        n = self.sim_matrix.size(0)
        i1, j1 = torch.triu_indices(n, n, 1)
        self.sim_matrix[j1, i1] = self.sim_matrix[i1, j1]

        # print(self.sim_matrix)
        # count_matrix = torch.ones_like(similarity_matrix).fill_diagonal_(0.0).flatten()
        # index_matrix = (labels.unsqueeze(1)*n_classes + labels.unsqueeze(0)).flatten()
        # similarity_matrix = similarity_matrix.fill_diagonal_(0.0).flatten()
        #
        # update_matrix = update_matrix.scatter_add(0, index_matrix, similarity_matrix)
        # labelcount_matrix = labelcount_matrix.scatter_add(0, index_matrix, count_matrix)
        # update_matrix = update_matrix.view(self.sim_matrix.shape)
        # labelcount_matrix = labelcount_matrix.view(self.sim_matrix.shape)
        # update_mask = labelcount_matrix > 0
        # update_matrix[update_mask] /= labelcount_matrix[update_mask]
        # self.sim_matrix[update_mask] += self.grad_ema_ratio * (update_matrix[update_mask] - self.sim_matrix[update_mask])
        # # print(self.sim_matrix)

        # for i in range(len(labels)):
        #     for j in range(i+1, len(labels)):
        #         if labels[i] < labels[j]:
        #             y1, y2 = labels[i].item(), labels[j].item()
        #         else:
        #             y1, y2 = labels[j].item(), labels[i].item()
        #         current_corr_map[y1][y2].append(similarity_matrix[i][j])
        #
        # for key_i in curr_corr_key_list:
        #     for key_j in curr_corr_key_list:
        #         if not math.isnan(np.mean(current_corr_map[key_i][key_j])):
        #             if self.corr_map[key_i][key_j] is None:
        #                 self.corr_map[key_i][key_j] = np.mean(current_corr_map[key_i][key_j])
        #             else:
        #                 self.corr_map[key_i][key_j] += self.grad_ema_ratio * (
        #                             np.mean(current_corr_map[key_i][key_j]) - self.corr_map[key_i][key_j])

        # for layer in list(self.selected_mask.keys()):
        #     cor_dic = {}
        #     for n, p in self.model.named_parameters():
        #         if p.requires_grad is True and p.grad is not None and n==layer:
        #             if not p.grad.isnan().any():
        #                 for i, y in enumerate(labels):
        #                     sub_sampled = p.grad1[i].clone().detach().clamp(-1000, 1000).flatten()[self.selected_mask[n]]
        #                     if y.item() not in cor_dic.keys():
        #                         cor_dic[y.item()] = [sub_sampled]
        #                     else:
        #                         cor_dic[y.item()].append(sub_sampled)
        #     centered_list = []
        #     key_list = list(cor_dic.keys())
        #
        #     for key in key_list:
        #         stacked_tensor = torch.stack(cor_dic[key])
        #         norm_tensor = torch.norm(stacked_tensor, p=2, dim=1) # make unit vector
        #
        #         for i in range(len(norm_tensor)):
        #             stacked_tensor[i] /= norm_tensor[i]
        #
        #         centered_list.append(stacked_tensor)
        #
        #     for i, key_i in enumerate(key_list):
        #         for j, key_j in enumerate(key_list):
        #             if key_i > key_j:
        #                 continue
        #             matmul_result = torch.matmul(centered_list[i], centered_list[j].T)
        #             if key_i==key_j:
        #                 #matmul_result.fill_diagonal_(0)
        #                 matmul_result = torch.stack([torch.cat([t[0:max(i,0)],t[min(i+1,len(matmul_result)):]]) for i, t in enumerate(matmul_result)])
        #
        #             cor_i_j = torch.mean(matmul_result).item()
        #             if not math.isnan(cor_i_j):
        #                 current_corr_map[key_i][key_j].append(cor_i_j)
        #
        # for i, key_i in enumerate(curr_corr_key_list):
        #     for j, key_j in enumerate(curr_corr_key_list):
        #         if key_i > key_j:
        #             continue
        #         if self.corr_map[key_i][key_j] is None:
        #             if len(current_corr_map[key_i][key_j]) != 0:
        #                 if not math.isnan(current_corr_map[key_i][key_j][0]):
        #                     self.corr_map[key_i][key_j] = np.mean(current_corr_map[key_i][key_j])
        #         else:
        #             if len(current_corr_map[key_i][key_j]) != 0:
        #                 if not math.isnan(current_corr_map[key_i][key_j][0]):
        #                     self.corr_map[key_i][key_j] += self.grad_ema_ratio * (np.mean(current_corr_map[key_i][key_j]) - self.corr_map[key_i][key_j])
    '''
    def update_correlation(self, labels):
        cor_dic = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad is True and p.grad is not None and n in self.target_layers[-1]:
                if not p.grad.isnan().any():
                    for i, y in enumerate(labels):
                        sub_sampled = p.grad1[i].clone().detach().clamp(-1000, 1000).flatten()[self.selected_mask[n]]
                        if y.item() not in cor_dic.keys():
                            cor_dic[y.item()] = [sub_sampled]
                        else:
                            cor_dic[y.item()].append(sub_sampled)

        centered_list = []
        key_list = list(cor_dic.keys())

        for key in key_list:
            #print("key", key, "len", len(cor_dic[key]))
            stacked_tensor = torch.stack(cor_dic[key])
            #print("stacked_tensor", stacked_tensor.shape)
            #stacked_tensor -= torch.mean(stacked_tensor, dim=0) # make zero mean
            norm_tensor = torch.norm(stacked_tensor, p=2, dim=1) # make unit vector
            
            for i in range(len(norm_tensor)):
                stacked_tensor[i] /= norm_tensor[i]
               
            centered_list.append(stacked_tensor)
        for i, key_i in enumerate(key_list):
            for j, key_j in enumerate(key_list):
                if key_i > key_j:
                    continue           
                cor_i_j = torch.mean(torch.matmul(centered_list[i], centered_list[j].T)).item()
                if self.corr_map[key_i][key_j] is None:
                    if not math.isnan(cor_i_j):
                        self.corr_map[key_i][key_j] = cor_i_j
                else:
                    self.corr_map[key_i][key_j] += self.grad_ema_ratio * (cor_i_j - self.corr_map[key_i][key_j])
    '''

    def get_layer_number(self, n):
        name = n.split('.')
        if name[0] == 'initial':
            return 0
        elif 'group' in name[0]:
            group_num = int(name[0][-1])
            block_num = int(name[2][-1])
            return group_num * 2 + block_num - 1

    @torch.no_grad()
    def calculate_fisher(self):
        group_fisher = [0.0 for _ in range(self.num_blocks)]
        for n, p in list(self.model.named_parameters())[:-2]:
            layer_num = self.get_layer_number(n)
            if p.requires_grad is True and p.grad is not None:
                if not p.grad.isnan().any():
                    block_name = '.'.join(n.split('.')[:-3])
                    get_attr = attrgetter(block_name)
                    group_fisher[layer_num] += (p.grad.clone().detach().clamp(-1000, 1000) ** 2).sum().item()/get_attr(self.model).input_scale
                    if self.unfreeze_rate < 1:
                        self.total_flops += (len(p.grad.clone().detach().flatten())*2+get_attr(self.model).input_size*2) / 10e9

        for i in range(self.num_blocks):
            if i not in self.freeze_idx or not self.frozen:
                self.fisher[i] += self.fisher_ema_ratio * (group_fisher[i] - self.fisher[i])
        self.total_fisher = sum(self.fisher)
        self.cumulative_fisher = [sum(self.fisher[0:i+1]) for i in range(9)]

    def get_flops_parameter(self):
        super().get_flops_parameter()
        self.cumulative_backward_flops = [sum(self.comp_backward_flops[0:i+1]) for i in range(9)]
        self.total_model_flops = self.forward_flops + self.backward_flops

    @torch.no_grad()
    def get_freeze_idx(self, logit, label):
        grad = self.get_grad(logit, label, self.model.fc.weight)
        last_grad = (grad ** 2 ).sum().item()
        if self.unfreeze_rate < 1:
            self.total_flops += len(grad.clone().detach().flatten())/10e9
        batch_freeze_score = last_grad/(self.last_grad_mean+1e-10)
        self.last_grad_mean += self.fisher_ema_ratio * (last_grad - self.last_grad_mean)
        freeze_score = []
        freeze_score.append(1)
        for i in range(9):
            freeze_score.append(self.total_model_flops / (self.total_model_flops - self.cumulative_backward_flops[i]) * (
                        self.total_fisher - self.cumulative_fisher[i]) / (self.total_fisher + 1e-10))
        max_score = max(freeze_score)
        modified_score = []
        modified_score.append(batch_freeze_score)
        for i in range(9):
            modified_score.append(batch_freeze_score*(self.total_fisher - self.cumulative_fisher[i])/(self.total_fisher + 1e-10) + self.cumulative_backward_flops[i]/self.total_model_flops * max_score)
        optimal_freeze = np.argmax(modified_score)
        # print(modified_score, optimal_freeze)
        self.freeze_idx = list(range(9))[0:optimal_freeze]

    @torch.no_grad()
    def get_grad(self, logit, label, weight):
        prob = F.softmax(logit)
        oh_label = F.one_hot(label.long(), self.num_learned_class)

        front = (prob - oh_label).shape
        back = weight.shape
        if self.unfreeze_rate < 1:
            self.total_flops += ((front[0] * back[1] * (2 * front[1] - 1)) / 10e9)

        return torch.matmul((prob - oh_label), weight)


class OurMemory(MemoryBase):
    def __init__(self, memory_size, T, count_decay_ratio, k_coeff, device='cpu'):
        super().__init__(memory_size)
        self.T = T
        self.k_coeff = k_coeff
        self.entered_time = []
        self.count_decay_ratio = count_decay_ratio
        self.device = device
        self.usage_count = torch.Tensor([]).to(self.device)
        self.class_usage_count = torch.Tensor([]).to(self.device)

    def replace_sample(self, sample, sample_num, idx=None):
        super().replace_sample(sample, idx)
        self.usage_count = torch.cat([self.usage_count, torch.zeros(1).to(self.device)])
        self.entered_time.append(sample_num)

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.cls_list)
        self.cls_list.append(class_name)
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.class_usage_count = torch.cat([self.class_usage_count, torch.zeros(1).to(self.device)])

    # balanced probability retrieval
    def balanced_retrieval(self, size):
        sample_size = min(size, len(self.images))
        memory_batch = []
        cls_idx = np.random.choice(len(self.cls_list), sample_size)
        for cls in cls_idx:
            i = np.random.choice(self.cls_idx[cls], 1)[0]
            memory_batch.append(self.images[i])
            self.usage_count[i] += 1
            self.class_usage_count[self.labels[i]] += 1
        return memory_batch

    
    def retrieval(self, size, similarity_matrix=None):
        # for use count decaying
        if len(self.images) > size:
            self.count_decay_ratio = size / (len(self.images)*self.k_coeff)  #(self.k_coeff / (len(self.images)*self.count_decay_ratio))
            # print("count_decay_ratio", self.count_decay_ratio)
            self.usage_count *= (1-self.count_decay_ratio)
            self.class_usage_count *= (1-self.count_decay_ratio)
        
        if similarity_matrix is None:
            return self.balanced_retrieval(size)
        else:
            sample_size = min(size, len(self.images))
            weight = self.get_similarity_weight(similarity_matrix)
            sample_idx = np.random.choice(len(self.images), sample_size, p=weight, replace=False)
            memory_batch = list(np.array(self.images)[sample_idx])
            for i in sample_idx:
                self.usage_count[i] += 1
                self.class_usage_count[self.labels[i]] += 1
            return memory_batch
        
    def get_similarity_weight(self, sim_matrix):
        n_cls = len(self.cls_list)
        # cls_cnt_sum = torch.zeros(n_cls)
        # for i, cnt in enumerate(self.usage_count):
        #     cls_cnt_sum[self.labels[i]] += cnt
        # print('\n\n')
        # print("cls_cnt_sum:", cls_cnt_sum.numpy().round(4))
        # print("cls_cnt_avg:", (cls_cnt_sum/torch.Tensor([len(self.cls_idx[i]) for i in range(n_cls)])).numpy().round(4))
        # sim_matrix = torch.zeros((n_cls, n_cls))
        self_score = torch.ones(n_cls).to(self.device)
        self_score -= sim_matrix.diag()
        #     for j in range(i, n_cls):
        #         sim_matrix[i][j] = sim_dict[i][j]
        #         sim_matrix[j][i] = sim_dict[i][j]
        #         if i == j:
        #             self_score -= sim_dict[i][i]
        # print("sim_matrix:\n", sim_matrix.numpy().round(4))

        cls_score_sum = (sim_matrix * self.class_usage_count).sum(dim=1)
        # print("cls_score_sum:", cls_score_sum.numpy().round(4))

        sample_score = cls_score_sum[torch.LongTensor(self.labels).to(self.device)] + self.usage_count.to(self.device)*self_score[torch.LongTensor(self.labels).to(self.device)]

        sample_score /= len(self.images)
        # print("sample_score mean, std:", sample_score.mean().item(), sample_score.std().item())
        # cls_score_sum = torch.zeros(n_cls)
        # for i in range(len(self.images)):
        #     cls_score_sum[self.labels[i]] += sample_score[i]
        # print("cls_score sum, mean:", cls_score_sum.numpy().round(4), (cls_score_sum/torch.Tensor(self.cls_count)).numpy().round(4))

        prob = F.softmax(-sample_score/self.T, dim=0)

        # cls_prob_sum = torch.zeros(n_cls)
        # for i in range(len(self.images)):
        #     cls_prob_sum[self.labels[i]] += prob[i]
        # print("prob sum, mean:", cls_prob_sum.numpy().round(4), (cls_prob_sum / torch.Tensor(self.cls_count)).numpy().round(4))
        # print('\n\n')

        return prob.cpu().numpy()
    
    
