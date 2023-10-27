import glob
import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')

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

print(np.nansum(lnbf_list))
np.savetxt("results/run_5_lnBF_GWTC3.csv", result,fmt="%s", delimiter=',')

clean_lnbf = [x for x in lnbf_list if (math.isnan(x) == False)]
clean_lnbf = [x for x in clean_lnbf if (x < 100)]
print("number of events: ", len(clean_lnbf))
print("total Bayes factor", np.sum(clean_lnbf))

x = np.arange(1, len(clean_lnbf)+1)
y = np.cumsum(clean_lnbf)

plt.figure()
plt.plot(x, y)
plt.xlabel('number of events', fontsize=18)
plt.ylabel(r'$ \textrm{cumulative} \ln \textrm{BF}_{\textrm{mem}}$', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.savefig('results/cumulative_lnBF_GWTC3.png')

new_labels = np.delete(labels, [10, 16, 33, 39, 45, 47, 48])
new_lnbf = np.delete(lnbf_list, [10, 16, 33, 39, 45, 47, 48])

new_labels[21], new_labels[22] = new_labels[22], new_labels[21]
new_lnbf[21], new_lnbf[22] = new_lnbf[22], new_lnbf[21]

Moritz_data_xhm = np.genfromtxt('Moritz_log_bfs_xhm.txt')
Moritz_data_prec = np.genfromtxt('Moritz_log_bfs_prec.txt')

Moritz_data_xhm = np.array(Moritz_data_xhm)
Moritz_data_prec = np.array(Moritz_data_prec)

Moritz_data_xhm = np.delete(Moritz_data_xhm, [15])
Moritz_data_prec = np.delete(Moritz_data_prec, [15])

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(new_labels[0:44], new_lnbf[0:44], marker = 'o', linestyle='None', label='IMRPhenomXPHM')
ax.plot(new_labels[0:44], Moritz_data_xhm, marker = '+', linestyle='None', label='IMRPhenomXHM')
ax.plot(new_labels[0:44], Moritz_data_prec, marker = 'x', linestyle='None', label='NRSur7dq4')

plt.xticks(rotation=50, ha='right')
plt.ylabel(r'$\ln BF_{mem}$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("results/compare_result_to_Moritz.png")
plt.savefig("results/compare_result_to_Moritz.pdf")