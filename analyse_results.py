import glob
import numpy as np
import matplotlib.pyplot as plt
import math

path_list = glob.glob("/home/shunyin.cheung/memory_GWTC3/run1/*_weights_IMRPhenomXPHM.csv")
s_path_list = sorted(path_list)

labels = []
lnbf_list = []

for file_name in s_path_list:
    data = np.genfromtxt(file_name)
    lnbf = np.log(np.sum(data)/len(data))
    lnbf_list.append(lnbf)
    
    text = file_name.split('/run1_')
    text2 = text[1].split('_weights')
    label = text2[0]
    labels.append(label)



"""
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(labels, lnbf_list, marker = 'o', linestyle='None')
ax.axhline(0, linestyle='dashed', color='black')
plt.xticks(rotation=50, ha='right')
plt.ylabel(r'$\ln BF_{mem}$')
plt.grid()
plt.tight_layout()
plt.savefig("GWTC3_lnbf_plot_part.png")
"""
print(np.nansum(lnbf_list))
np.savetxt("run_2_lnBF_GWTC3.csv", lnbf_list)
"""
plt.figure()
plt.hist(lnbf_list)
plt.xlabel('$lnBF_{mem}$')
plt.savefig('hist_lnbf_GWTC3.png')
"""

clean_lnbf = [x for x in lnbf_list if (math.isnan(x) == False)]

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