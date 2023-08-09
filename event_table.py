from gwosc import datasets
import glob
import numpy as np
import h5py

def call_event_table():

    event_label = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608', 'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190403', 'GW190408', 'GW190412', 'GW190413_134308', 'GW190413_052954', 'GW190421', 'GW190426_190642', 'GW190503', 'GW190512', 'GW190513', 'GW190514', 'GW190517', 'GW190519', 'GW190521', 'GW190521_074359', 'GW190527', 'GW190602', 'GW190620', 'GW190630', 'GW190701', 'GW190706', 'GW190707', 'GW190708', 'GW190719', 'GW190720', 'GW190725', 'GW190727', 'GW190728', 'GW190731', 'GW190803', 'GW190805', 'GW190814', 'GW190828_065509', 'GW190828_063405', 'GW190910', 'GW190915', 'GW190916', 'GW190917', 'GW190924', 'GW190925', 'GW190926', 'GW190930', 'GW191103', 'GW191105', 'GW191109', 'GW191113', 'GW191126', 'GW191127', 'GW191129', 'GW191204_171526', 'GW191204_110529', 'GW191215', 'GW191216', 'GW191219', 'GW191222', 'GW191230', 'GW200105', 'GW200112', 'GW200115', 'GW200128', 'GW200129', 'GW200202', 'GW200208_222617', 'GW200208_130117', 'GW200209', 'GW200210', 'GW200216', 'GW200219', 'GW200220_124850', 'GW200220_061928', 'GW200224', 'GW200225', 'GW200302', 'GW200306', 'GW200308', 'GW200311_115853', 'GW200316', 'GW200322']

    path_list = glob.glob("/home/shunyin.cheung/GWOSC_posteriors/*.h5")

    s_path_list = sorted(path_list)
    #print(s_path_list)


    detectors = []
    durations = []
    trigger_time = []

    # The following for loop accesses the durations in the public data. 
    # Because the categories are non-standard between differen events, 
    #it uses a lot of if statements to find the right statements.

    for file_name in s_path_list:
        #print(file_name)
        with h5py.File(file_name,'r+') as f1:
            wf_name1 = 'C01:IMRPhenomXPHM'
            wf_name2 = 'C01:IMRPhenomXPHM:HighSpin'
            if wf_name1 in list(f1.keys()):
                if 'config' in list(f1['C01:IMRPhenomXPHM']['config_file'].keys()):
                    duration = float(list(f1['C01:IMRPhenomXPHM']['config_file']['config']['duration'])[0])
                elif 'input' in list(f1['C01:IMRPhenomXPHM']['config_file'].keys()):
                    duration = float(list(f1['C01:IMRPhenomXPHM']['config_file']['input']['padding'])[0])
                else: 
                    print('category does not exist')
            elif wf_name2 in list(f1.keys()):
                duration = float(list(f1['C01:IMRPhenomXPHM:HighSpin']['config_file']['config']['duration'])[0])
            else:
                print('These are not IMRPhenomXPHM posteriors')

            durations.append(duration)
        


    for event in event_label:
        detector = datasets.event_detectors(event)
        detectors.append(list(detector))
        tt = datasets.event_gps(event)
        trigger_time.append(tt)



    event_table = list(zip(event_label, s_path_list, trigger_time, detectors, durations))
    return event_table
#print(call_event_table())