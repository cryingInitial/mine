import pickle5 as pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
from torch import linalg as LA

fig_name = "baseline_joint_nc1_result.png"

def get_angle(a, b):
    inner_product = (a * b).sum()
    a_norm = a.pow(2).sum().pow(0.5)
    b_norm = b.pow(2).sum().pow(0.5)
    cos = inner_product / (2 * a_norm * b_norm)
    #angle = torch.acos(cos)
    return cos

def get_within_class_covariance(mean_vec_list, feature_dict):
    # feature dimension 512로 fixed
    W = torch.zeros((512, 512))
    total_num = 0

    for klass in list(feature_dict.keys()):
        for i in range(len(feature_dict[klass])):
            W += torch.outer((feature_dict[klass][i] - mean_vec_list[klass]), (feature_dict[klass][i] - mean_vec_list[klass]))
        total_num += len(feature_dict[klass])
    W /= total_num
    return W

def get_between_class_covariance(mean_vec_list, feature_dict):
    B = torch.zeros((512, 512))

    # for global avg calcuation, not just avg mean_vec, feature mean directly (since it is imbalanced dataset)
    total_feature_dict = []
    for key in feature_dict.keys():
        total_feature_dict.extend(feature_dict[key])

    global_mean_vec = torch.mean(torch.stack(total_feature_dict, dim=0), dim=0)

    for klass in list(feature_dict.keys()):
        #B += (mean_vec_list[klass] - global_mean_vec) * (mean_vec_list[klass] - global_mean_vec).T
        B += torch.outer((mean_vec_list[klass] - global_mean_vec), (mean_vec_list[klass] - global_mean_vec))
    B /= len(list(mean_vec_list.keys()))
    return B, global_mean_vec

def get_nc2(mean_vec_list, global_mean_vec):
    M = []
    K = len(list(mean_vec_list.keys()))
    for key in list(mean_vec_list.keys()):
        recentered_mean = mean_vec_list[key] - global_mean_vec
        M.append(recentered_mean / nn.functional.normalize(recentered_mean, p=2.0))

    M = torch.stack(M, dim=0)

    nc2_matrix = (torch.matmul(M, M.T) / LA.matrix_norm(torch.matmul(M, M.T))) - ((K-1)**-0.5) * (torch.eye(K) - (1/K)*torch.ones((K,K)))
    return LA.matrix_norm(nc2_matrix)

nc1_list = []
nc2_list = []
for index in range(50, 7800, 50):
    print("index", index)
    fc_pickle_name = "baseline_joint_num_" + str(index) + "_fc.pickle"
    feature_pickle_name = "baseline_joint_num_" + str(index) + "_feature.pickle"
    class_pickle_name = "baseline_joint_num_" + str(index) + "_class.pickle"

    with open(fc_pickle_name, 'rb') as f:
        fc_dict = pickle.load(f)

    with open(feature_pickle_name, 'rb') as f:
        feature_dict = pickle.load(f)

    with open(class_pickle_name, 'rb') as f:
        class_dict = pickle.load(f)

    mean_vec_list = {}
    # feature normalize
    for cls in list(feature_dict.keys()):
        feature_dict[cls] = torch.cat(feature_dict[cls]).detach().cpu()
        feature_dict[cls] /= torch.norm(feature_dict[cls], p=2, dim=1, keepdim=True)
        mean_vec_list[cls] = torch.mean(feature_dict[cls], dim=0)
    
    #mean_vec_list = [torch.mean(feature_dict[cls], dim=0) for cls in list(feature_dict.keys())]
    
    '''
    ### angle check ###
    for i in range(len(mean_vec_list)):
        for j in range(i+1, len(mean_vec_list)):
            print("i", i, "j", j, "cos", get_angle(mean_vec_list[i], mean_vec_list[j]))
    print()
    '''
    '''
    print("get within covariance")
    print(get_within_class_covariance(mean_vec_list, feature_dict).shape)
    
    print("get between covariance")
    print(get_between_class_covariance(mean_vec_list, feature_dict).shape)
    '''

    ### check nc1 ###
    W = get_within_class_covariance(mean_vec_list, feature_dict)
    B, global_mean_vec = get_between_class_covariance(mean_vec_list, feature_dict)
    nc1_list.append(torch.trace(torch.matmul(W, B)) / len(mean_vec_list))

    ### check nc2 ###
    #nc2_list.append(get_nc2(mean_vec_list, global_mean_vec))

### plot nc1 ###
plt.plot(range(len(nc1_list)), savgol_filter(nc1_list, 21, 3))
plt.title("nc1_result")
plt.savefig(fig_name)
