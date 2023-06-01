# When we make a new one, we should inherit the Finetune class.
import logging
import numpy as np
from methods.cl_manager import CLManagerBase
from methods.etf_er import ETF_ER
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

class ETF_ER_SELFSUP(ETF_ER):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        super().__init__(train_datalist, test_datalist, device, **kwargs)
        
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

        self.total_flops += (len(y) * self.forward_flops)

        # accuracy calculation
        with torch.no_grad():
            cls_score = feature @ self.etf_vec
            acc, _ = self.compute_accuracy(cls_score[:, :self.eval_classes], y)
            acc = acc.item()
            #print(acc)

        return logit, loss, feature, acc

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

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.total_flops += (len(y) * self.backward_flops)

            self.after_model_update()

            total_loss += loss.item()
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

    def add_new_class(self, class_name):
        self.cls_dict[class_name] = len(self.exposed_classes)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        self.eval_classes += 1


