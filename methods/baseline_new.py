# When we make a new one, we should inherit the Finetune class.
import logging
import numpy as np
from methods.cl_manager import CLManagerBase
import copy
import pickle
logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")

class BASELINE(CLManagerBase):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        super().__init__(train_datalist, test_datalist, device, **kwargs)

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

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
            self.writer.add_scalar(f"train/add_new_class", 1, sample_num)
        else:
            self.writer.add_scalar(f"train/add_new_class", 0, sample_num)
        
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()
        
        '''
        if self.sample_num % 100 == 0 and self.sample_num !=0:
            for name, param in self.model.named_parameters():
                if "fc.weight" in name:
                    fc_weight = copy.deepcopy(param)
                    fc_pickle_name = "baseline_num_" + str(self.sample_num) + "_fc.pickle"
                    print("fc_pickle_name", fc_pickle_name)
                    with open(fc_pickle_name, 'wb') as f:
                        pickle.dump(fc_weight, f, pickle.HIGHEST_PROTOCOL)
                     
            feature_pickle_name = "baseline_num_" + str(self.sample_num) + "_feature.pickle"
            class_pickle_name = "baseline_num_" + str(self.sample_num) + "_class.pickle"
            print("feature_pickle_name", feature_pickle_name)
            print("class_pickle_name", class_pickle_name)
            self.save_features(feature_pickle_name, class_pickle_name)
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






