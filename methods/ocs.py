import torch
import torch.nn as nn
import copy
import os
import time
import PIL

from methods.cl_manager import MemoryBase
from methods.er_new import ER

from utils.data_loader import MultiProcessLoader
from utils.augment import Preprocess

class OCS(ER):
    def __init__(self, train_datalist, test_datalist, device, **kwargs):
        self.n_tasks = kwargs["n_tasks"]
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.total_samples = len(train_datalist)
        self.count_get_call = 0
        self.delayed_base = 0
        self.base = 0
        
    def initialize_future(self):
        self.samples_per_task = len(self.train_datalist) // self.n_tasks
        self.data_stream = iter(self.train_datalist)
        self.memory_batch_size //= 2
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory_size += (self.temp_batch_size) // 2
        self.memory = OCSMemory(self.memory_size, self.device, self.sigma)
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
        
        self.task_num = 0
        self.candidates = []

        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        
        for i in range(self.future_steps):
            self.load_batch()

    def online_before_task(self, task):
        # task 변경 적용
        self.task_num = task
        
        # coreset을 기반으로 memory update
        if task > 0:
            self.memory.update_coreset(self.model, self.candidates, self.train_datalist, data_dir=self.data_dir)

        self.memory.task_change(self.task_num)
        self.candidates = []
        

    def online_train(self, iterations=1):
                                                                                                                                                                                                                          
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        
        for i in range(iterations):
            
            self.model.train()
            data = self.get_batch()
            
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            x_stream = x[:self.temp_batch_size]
            y_stream = y[:self.temp_batch_size]
            x_memory = x[self.temp_batch_size:]
            y_memory = y[self.temp_batch_size:]
            
            self.before_model_update()
        
            # Picking some samples from stream
            pick = []
            if (self.base % (len(self.train_datalist) // (self.n_tasks * 2))) < 100 * self.temp_batch_size: # 2 == # of sub_step
                pick = torch.randperm(len(x_stream))[:self.temp_batch_size // 2]
                # debug
                # self.candidates += list(pick + self.base)
                # self.memory.update_coreset(self.model, self.candidates, self.train_datalist, data_dir=self.data_dir)
                # self.candidates = []
            else:
                _eg = self.compute_and_flatten_example_grads(self.model, self.optimizer, x_stream, y_stream, self.task_num)
                _g = torch.mean(_eg, dim=0)
                ref_grads = None
                
                if len(x_memory) > 0:
                    ref_pred = self.model(x_memory)
                    ref_loss = self.criterion(ref_pred, y_memory)
                    ref_loss.backward()
                    ref_grads = copy.deepcopy(self.flatten_grads(self.model))
                    
                pick = self.sample_selection(_g, _eg, ref_grads)[:self.temp_batch_size // 2]
                    
                # Select Candidates only when sample coreset
                if (self.samples_per_task // 2 <= self.base % self.samples_per_task < self.samples_per_task) \
                    and ((self.count_get_call - self.base * self.online_iter) == (self.temp_batch_size * self.online_iter - 1)):
                    self.candidates += list(pick + self.base)
                    # print(self.candidates)
                    
            # default sampling for empty memory
            while not self.memory.is_full():
                default_pick = pick + self.base
                for p in default_pick:
                    if self.memory.is_full(): break
                    self.memory.default_sampling(p)
                
            x_stream = x_stream[pick]
            y_stream = y_stream[pick]
            
            self.optimizer.zero_grad()
            
            # important stream loss
            logit, loss = self.model_forward(x_stream, y_stream)
            
            # coreset memory loss
            if len(x_memory) > 0:
                print("Memory is capable")
                logit_memory, loss_memory = self.model_forward(x_memory, y_memory)
                loss += (0.5 * loss_memory) / len(x_memory) # 0.5 is hyperparameter
                
            _, preds = logit.topk(self.topk, 1, True, True)
            
            # update
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.after_model_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y_stream.unsqueeze(1)).item()
            num_data += y_stream.size(0)

        return total_loss / iterations, correct / num_data
    
    def get_batch(self):
        self.count_get_call += 1
        self.base = int((self.count_get_call // (self.online_iter * self.temp_batch_size)) * self.temp_batch_size)
        batch = self.dataloader.get_batch()
        self.load_batch()
        return batch
    
    # From OCS paper, github.com/rahafaljundi/OCS
    def sample_selection(self, g, eg, ref_grads=None, attn=None):
        # grap gpu memory
        g, eg = g.to(self.device), eg.to(self.device)
        if ref_grads is not None: ref_grads = ref_grads.to(self.device)
        
        ng = torch.norm(g)
        neg = torch.norm(eg, dim=1)
        mean_sim = torch.matmul(g,eg.t()) / torch.maximum(ng*neg, torch.ones_like(neg)*1e-6)
        negd = torch.unsqueeze(neg, 1)

        cross_div = torch.matmul(eg,eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), torch.ones_like(negd)*1e-6)
        mean_div = torch.mean(cross_div, 0)

        coreset_aff = torch.tensor(0.).to(self.device)
        if ref_grads is not None:
            ref_ng = torch.norm(ref_grads)
            
            coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng*neg, torch.ones_like(neg)*1e-6)

        measure = mean_sim - mean_div + 1000 * coreset_aff # tau = 1000
        _, u_idx = torch.sort(measure, descending=True)
        
        # release all gpu memory
        del g, eg, neg, mean_sim, negd, cross_div, mean_div, coreset_aff, measure
        return u_idx.cpu().numpy()
    
    def compute_and_flatten_example_grads(self, m, criterion, data, target, task_id):
        _eg = []
        criterion2 = nn.CrossEntropyLoss(reduction='none').to(self.device)
        m.eval()
        m.zero_grad()
        pred = m(data)
        loss = criterion2(pred, target)
        for idx in range(len(data)):
            loss[idx].backward(retain_graph=True)
            _g = self.flatten_grads(m, numpy_output=True)
            _eg.append(torch.Tensor(_g))
            m.zero_grad()
        return torch.stack(_eg)
    
    # Extract gradients from the model
    def flatten_grads(self, m, numpy_output=False, bias=True, only_linear=False):
        total_grads = []
        for name, param in m.named_parameters():
            if only_linear:
                if (bias or not 'bias' in name) and 'linear' in name:
                    total_grads.append(param.grad.detach().view(-1))
            else:
                if (bias or not 'bias' in name) and not 'bn' in name and not 'IC' in name:
                    try:
                        total_grads.append(param.grad.detach().view(-1))
                    except AttributeError:
                        pass
                        #print('no_grad', name)
        total_grads = torch.cat(total_grads)
        if numpy_output:
            return total_grads.cpu().detach().numpy()
        return total_grads
    
    def update_memory(self, sample):
        pass

class OCSMemory(MemoryBase):
    
    def __init__(self, memory_size, device, sigma):
        super().__init__(memory_size)
        self.task = 0
        self.task_idx = []
        self.task_count = []
        self.sample_per_class = memory_size
        self.sample_per_task = memory_size
        self.device = device
        self.sigma = sigma
    
    def is_full(self):
        return len(self.images) >= self.memory_size
    
    def task_change(self, task):
        self.task_idx.append([])
        self.task = task
        
    # fill the wasted memory with random samples
    def default_sampling(self, sample):
        self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
        self.task_idx[self.task].append(len(self.images))
        self.images.append(sample)
        self.labels.append(self.cls_dict[sample['klass']])
        print(self.task_idx)
        print(self.cls_idx)
        
    def add_new_class(self, class_name):
        super().add_new_class(class_name)
        self.sample_per_class = self.memory_size // len(self.cls_list)
    
    def update_coreset(self, model, candidates, train_datalist, candidate_size=500, data_dir = None):
        
        
        self.sample_per_task = self.memory_size // (self.task + 1)
        self.sample_per_class = self.memory_size // (len(self.cls_list))
        self.class_per_task = len(self.cls_list) // (self.task + 1)
        
        ###########################
        # UPDATE EXISTING CORESET #
        ###########################
        for t in range(self.task - 1):
            if (len(self.task_idx[t]) != 0):
                preprocess = Preprocess()
                images = []
                labels = []
                for idx in self.task_idx[t]:
                    img_name = self.images[idx]["file_name"]
                    img_path = os.path.join(data_dir, img_name)
                    image = PIL.Image.open(img_path).convert("RGB")
                    images.append(preprocess(image).to(self.device))
                    labels.append(self.labels[idx])
                
                images = torch.stack(images)
                labels = torch.Tensor(labels).long().to(self.device)
                pred = model(images)
                criterion = nn.CrossEntropyLoss().to(self.device)
                loss = criterion(pred, labels)
                loss.backward()
                
                _tid_eg = self.compute_and_flatten_example_grads(model, nn.CrossEntropyLoss(reduction='none'), images, labels, t)
                _tid_g = torch.mean(_tid_eg, 0)
                sorted = self.sample_selection(_tid_g, _tid_eg)
                # 중요도 순으로 정렬
                self.task_idx[t] = [self.task_idx[t][i] for i in sorted]
                
                # CLASS 중요도 순으로 정렬 (DISJOINT)
                if self.sigma == 0:
                    for c_t in range(t * self.class_per_task, (t + 1) * self.class_per_task):
                        new_cls_idx_segment = []
                        for idx in self.task_idx[t]:
                            for c_idx in self.cls_idx[c_t]:
                                if c_idx == idx:
                                    new_cls_idx_segment.append(idx)
                        self.cls_idx[c_t] = new_cls_idx_segment
                            
        ###########
        # ADD NEW #
        ###########
        candidates = candidates[:candidate_size]
        pick = torch.randperm(len(candidates))
        candidates = [candidates[i] for i in pick]
        
        # TASK 별로 공평하게 분배 (BLURRY)
        if self.sigma > 0:    
            candidates = candidates[:self.sample_per_task]  
            for c_idx in candidates:
                if self.memory_size > len(self.images):
                    self.task_idx[self.task].append(len(self.images))
                    self.images.append(train_datalist[c_idx])
                    self.labels.append(self.cls_dict[train_datalist[c_idx]['klass']])
                else:
                    switch_idx = max(self.task_idx, key=len).pop()
                    self.task_idx[self.task].append(switch_idx)
                    self.images[switch_idx] = train_datalist[c_idx]
                    self.labels[switch_idx] = self.cls_dict[train_datalist[c_idx]['klass']]
            
            print("BLURRY REPORT")
            print('task_idx', self.task_idx)
            
        # CLASS 별로 공평하게 분배 (DISJOINT)
        else:
            for c_idx in candidates:
                candidate = train_datalist[c_idx]
                cls_of_candidate = self.cls_dict[candidate['klass']]
                if len(self.cls_idx[cls_of_candidate]) < self.sample_per_class:
                    if self.memory_size > len(self.images):
                        self.task_idx[self.task].append(len(self.images))
                        self.cls_idx[cls_of_candidate].append(len(self.images))
                        self.images.append(candidate)
                        self.labels.append(self.cls_dict[candidate['klass']])
                    else:
                        print(max(self.cls_idx, key=len))
                        switch_idx = max(self.cls_idx, key=len).pop()
                        switch_task = cls_of_candidate // self.class_per_task
                        self.task_idx[switch_task].pop(self.task_idx[switch_task].index(switch_idx))
                        self.task_idx[self.task].append(switch_idx)
                        self.cls_idx[cls_of_candidate].append(switch_idx)
                        self.images[switch_idx] = candidate
                        self.labels[switch_idx] = self.cls_dict[candidate['klass']]
        
            print("DISJOINT REPORT")
            print('task_idx', self.task_idx)
            print('cls_idx', self.cls_idx)
            
            
        print("FIANL REPORT")
        print([self.cls_dict[image['klass']] for image in self.images], len(self.images))
            
            
    def compute_and_flatten_example_grads(self, m, criterion, data, target, task_id):
        _eg = []
        criterion2 = nn.CrossEntropyLoss(reduction='none').to(self.device)
        m.eval()
        m.zero_grad()
        pred = m(data)
        loss = criterion2(pred, target)
        for idx in range(len(data)):
            loss[idx].backward(retain_graph=True)
            _g = self.flatten_grads(m, numpy_output=True)
            _eg.append(torch.Tensor(_g))
            m.zero_grad()
        return torch.stack(_eg)
    
    # Extract gradients from the model
    def flatten_grads(self, m, numpy_output=False, bias=True, only_linear=False):
        total_grads = []
        for name, param in m.named_parameters():
            if only_linear:
                if (bias or not 'bias' in name) and 'linear' in name:
                    total_grads.append(param.grad.detach().view(-1))
            else:
                if (bias or not 'bias' in name) and not 'bn' in name and not 'IC' in name:
                    try:
                        total_grads.append(param.grad.detach().view(-1))
                    except AttributeError:
                        pass
                        #print('no_grad', name)
        total_grads = torch.cat(total_grads)
        if numpy_output:
            return total_grads.cpu().detach().numpy()
        return total_grads
    
    def sample_selection(self, g, eg, ref_grads=None, attn=None):
        ng = torch.norm(g)
        neg = torch.norm(eg, dim=1)
        mean_sim = torch.matmul(g,eg.t()) / torch.maximum(ng*neg, torch.ones_like(neg)*1e-6)
        negd = torch.unsqueeze(neg, 1)

        cross_div = torch.matmul(eg,eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), torch.ones_like(negd)*1e-6)
        mean_div = torch.mean(cross_div, 0)

        coreset_aff = 0.
        if ref_grads is not None:
            ref_ng = torch.norm(ref_grads)
            coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng*neg, torch.ones_like(neg)*1e-6)

        measure = mean_sim - mean_div + 1000 * coreset_aff # tau = 1000
        _, u_idx = torch.sort(measure, descending=True)
        return u_idx.cpu().numpy()