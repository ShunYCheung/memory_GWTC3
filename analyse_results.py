import glob
import numpy as np
import matplotlib.pyplot as plt
import math


path_list = glob.glob("/home/shunyin.cheung/memory_GWTC3/run2/*_IMRPhenomXPHM.csv")
s_path_list = sorted(path_list)

labels = []
lnbf_list = []

for file_name in s_path_list:
    data = np.genfromtxt(file_name)
    lnbf = np.log(np.nansum(data)/len(data))
    lnbf_list.append(lnbf)
    
    text = file_name.split('weights_')
    text2 = text[1].split('_IMRPhenomXPHM')
    label = text2[0]
    labels.append(label)

#print(labels)
labels = np.array(labels)
lnbf_list = np.array(lnbf_list)
result = np.stack((labels, lnbf_list), axis=1)
#print(result)

Moritz_data_xhm = np.genfromtxt('Moritz_log_bfs_xhm.txt')
Moritz_data_prec = np.genfromtxt('Moritz_log_bfs_prec.txt')
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(labels[0:10], lnbf_list[0:10], marker = 'o', linestyle='None', label='Shun')
ax.plot(labels[0:10], Moritz_data_xhm[0:10], marker = 'o', linestyle='None', label='Moritz xhm')
ax.plot(labels[0:10], Moritz_data_prec[0:10], marker = 'o', linestyle='None', label='Moritz prec')

plt.xticks(rotation=50, ha='right')
plt.ylabel(r'$\ln BF_{mem}$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("compare_result_to_Moritz.png")

print(np.nansum(lnbf_list))
np.savetxt("run_5_lnBF_GWTC3.csv", result,fmt="%s", delimiter=',')

clean_lnbf = [x for x in lnbf_list if (math.isnan(x) == False)]
clean_lnbf = [x for x in clean_lnbf if (x < 100)]
print("number of events: ", len(clean_lnbf))
print("total Bayes factor", np.sum(clean_lnbf))

x = np.arange(1, len(clean_lnbf)+1)
y = np.cumsum(clean_lnbf)

plt.figure()
plt.plot(x, y)
plt.xlabel('number of GW events', fontsize=18)
plt.ylabel('ln BF', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.savefig('cumulative_lnBF_GWTC3.pdf')