# When we make a new one, we should inherit the Finetune class.
import logging
import numpy as np
from methods.cl_manager import CLManagerBase
from utils.train_utils import DR_loss, Accuracy
import torch
import torch.nn as nn
import copy
from utils.data_loader import ImageDataset, cutmix_data, MultiProcessLoader, get_statistics
import math
import utils.train_utils 
from utils.train_utils import select_optimizer, select_model, select_scheduler
logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")

class ETF(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        self.in_channels = self.model.fc.in_features
        
        # 아래 것들 config 파일에 추가!
        # num class = 100, eval_class = 60 
        self.num_classes = kwargs["num_class"]
        print("num_classes!!", self.num_classes)
        self.eval_classes = 1 #kwargs["num_eval_class"]
        
        self.criterion = DR_loss().to(self.device)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.model = select_model(self.model_name, self.dataset, 1).to(self.device)
        print("model")
        print(self.model)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)
        self.etf_initialize()
        # ETF 꼴로 fc를 load!
        '''
        copy_state_dict = copy.deepcopy(self.model.state_dict())
        for key in self.model.state_dict.keys():
            if key == 'fc.weight': # weight는 etf structure로
                copy_state_dict[key] = self.etf_initialize()
            elif key == 'fc.bias': # bias는 0으로 
                copy_state_dict[key] = torch.zeros_like(self.model.state_dict[key])
        self.model.load_state_dict(copy_state_dict)
        '''
        #self.model.fc = None

    def model_forward(self, x, y):
        #do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        do_cutmix = False

        """Forward training data."""
        target = self.etf_vec[:, y].t()
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit, feature = self.model(x, get_feature=True)
                #loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                loss = lam * self.criterion(feature, labels_a) + (1 - lam) * self.criterion(feature, labels_b)
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit, feature = self.model(x, get_feature=True)
                feature = self.pre_logits(feature)
                loss = self.criterion(feature, target)

        #self.total_flops += (len(y) * self.forward_flops)

        # accuracy calculation
        with torch.no_grad():
            cls_score = feature @ self.etf_vec
            acc, _ = self.compute_accuracy(cls_score[:, :self.eval_classes], y)
            acc = acc.item()
            #print(acc)

        return logit, loss, feature, acc

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x


    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)

            self.before_model_update()

            self.optimizer.zero_grad()

            # logit can not be used anymore
            _, loss, feature, acc = self.model_forward(x,y)

            #_, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            #self.total_flops += (len(y) * self.backward_flops)

            self.after_model_update()

            total_loss += loss.item()
            #correct += torch.sum(preds == y.unsqueeze(1)).item()
            #self.model = select_model(self.model_name, self.dataset, 1).to(self.device)
        #print("model")
        #print(self.model)
        #self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        #self.scheduler = select_scheduler(self.sched_name, self.optimizer)       
        num_data += y.size(0)

        return total_loss / iterations, acc #, correct / num_data
        
    def etf_initialize(self):
        logger.info("ETF head : evaluating {} out of {} classes.".format(self.eval_classes, self.num_classes))

        orth_vec = self.generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        self.etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1))).to(self.device)
        print("etf shape")
        print(self.etf_vec.shape)
        print("etf angle")
        print(self.get_angle(self.etf_vec[:,0],self.etf_vec[:,1]), self.get_angle(self.etf_vec[:,0],self.etf_vec[:,-1]))
        #return self.etf_vec

    def get_angle(self, a, b):
        inner_product = (a * b).sum(dim=0)
        a_norm = a.pow(2).sum(dim=0).pow(0.5)
        b_norm = b.pow(2).sum(dim=0).pow(0.5)
        cos = inner_product / (2 * a_norm * b_norm)
        angle = torch.acos(cos)
        return angle

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        rand_mat = np.random.random(size=(feat_in, num_classes))
        orth_vec, _ = np.linalg.qr(rand_mat) # qr 분해를 통해서 orthogonal한 basis를 get
        orth_vec = torch.tensor(orth_vec).float()
        assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
            "The max irregular value is : {}".format(
                torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
        return orth_vec

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
        self.update_memory(sample)
        self.temp_future_batch.append(sample)
        self.future_num_updates += self.online_iter

        if self.future_num_updates >= 1:
            self.temp_future_batch = []
            self.generate_waiting_batch(int(self.future_num_updates))
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        self.eval_classes += 1


    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
            self.writer.add_scalar(f"train/add_new_class", 1, sample_num)
        else:
            self.writer.add_scalar(f"train/add_new_class", 0, sample_num)
        
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            #train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
            train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

        if self.sample_num % 100 == 0 and self.sample_num !=0:
            #print("self.etf_vec", self.etf_vec.shape)
            #print("self.etf_vec[:, :self.eval_classes]", self.etf_vec[:, :self.eval_classes].shape)
            #self.infer_whole_memory(self.etf_vec[:, :self.eval_classes].T)
            pass
    
    '''
    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        # fc layer update는 안해도 됨
    '''

    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            self.waiting_batch.append(self.memory.retrieval(self.memory_batch_size))

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)


    def sub_simple_test(self, x, softmax=False, post_process=False):
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        cls_score = cls_score[:, :self.eval_classes]
        assert not softmax
        '''
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score
        '''
        return cls_score


    def simple_test(self, img, gt_label, return_feature=True):
        """Test without augmentation."""
        '''
        if return_backbone:
            x = self.extract_feat(img, stage='backbone')
            return x
        x = self.extract_feat(img)
        '''
        _, feature = self.model(img, get_feature=True)
        res = self.sub_simple_test(feature, post_process=False)
        res = res.argmax(dim=-1)
        if return_feature:
            return torch.eq(res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist(), feature
        else:
            return torch.eq(res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist()


    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        total_acc = 0.0
        label = []
        feature_dict = {}
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                res, features = self.simple_test(x, y, return_feature=True)
                features = self.pre_logits(features)

                unique_y = torch.unique(y).tolist()
                for u_y in unique_y:
                    indices = (y == u_y).nonzero(as_tuple=True)[0]
                    if u_y not in feature_dict.keys():
                        feature_dict[u_y] = [torch.index_select(features, 0, indices)]
                    else:
                        feature_dict[u_y].append(torch.index_select(features, 0, indices))

                #print("features", features.shape)
                #print("y", y.shape)
                target = self.etf_vec[:, y].t()
                loss = self.criterion(features, target)

                # accuracy calculation
                with torch.no_grad():
                    cls_score = features @ self.etf_vec
                    _, correct_count = self.compute_accuracy(cls_score[:, :self.eval_classes], y)
                    total_correct += correct_count
                    total_num_data += y.size(0)
                    '''
                    total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                    total_num_data += y.size(0)

                    xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                    correct_l += correct_xlabel_cnt.detach().cpu()
                    num_data_l += xlabel_cnt.detach().cpu()
                    '''
                    total_loss += loss.item()
                
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        #cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc}

        return ret
