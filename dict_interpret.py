import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

iteration = 4
std_dict_pickle_name = str(int(iteration)) + '_std_dict.pickle'
dist_dict_pickle_name = str(int(iteration)) + '_dist_dict.pickle'
fig_name = str(int(iteration)) + "_std_dist.png"

with open(std_dict_pickle_name, 'rb') as f:
    std_dict = pickle.load(f)

with open(dist_dict_pickle_name, 'rb') as f:
    dist_dict = pickle.load(f)
    
keys = list(std_dict.keys())
std_max_len = 0
dist_max_len = 0
for key in keys:
    std = std_dict[key]
    dist = dist_dict[key]
    dist_max_len = max(len(dist), dist_max_len)
    std_max_len = max(len(std), std_max_len)

#print("std_max_len",std_max_len)
#print("dist_max_len",dist_max_len)

plt.subplot(1, 2, 1)
plt.title("std_list")
for key in keys:
    std = std_dict[key]
    x_list = list(range(std_max_len))[-len(std):]
    label = "class" +str(key)
    #plt.plot(x_list, std, label=label)
    plt.plot(x_list, savgol_filter(std, 21, 3), label=label)
for i in range(std_max_len):
    if i%100 == 0 and i!=0:
        plt.axvline(x=i, color='r', linewidth=3)
plt.legend()

plt.subplot(1, 2, 2)
plt.title("dist_list")
for key in keys:
    dist = dist_dict[key]
    x_list = list(range(dist_max_len))[-len(dist):]
    label = "class" + str(key)
    plt.plot(x_list, savgol_filter(dist, 21, 3), label=label)
for i in range(dist_max_len):
    if i%100 == 0 and i!=0:
        plt.axvline(x=i, color='r', linewidth=3)
plt.legend()

#plt.show()
plt.savefig(fig_name)
