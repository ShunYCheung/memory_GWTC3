from reweight_mem_parallel import reweight_mem_parallel
import json
from create_post_dict import create_post_dict
from event_table import call_event_table
import sys

event_number = int(sys.argv[1])

if __name__ == '__main__':

    
    events = [call_event_table()[event_number]]
    
    
    waveform = "IMRPhenomXPHM"
    
    for count, i in enumerate(events):
        event_name, file_path, trigger_time, detectors, duration = i
        samples_dict, meta_dict, config_dict, priors_dict, psds, calibration = create_post_dict(file_path)
        print("reweighting {}".format(event_name), "{0}/{1} events reweighted".format(count+1, len(events)))
        weights, bf = reweight_mem_parallel(event_name, 
                                            samples_dict, 
                                            meta_dict,
                                            config_dict,
                                            priors_dict,
                                            calibration,
                                            detectors,
                                            "/home/shunyin.cheung/memory_GWTC3/Shun_test_run"
                                            ,"test_likelihood_{0}_weights".format(event_name), 
                                            psds = psds,
                                            n_parallel=4)
        
        
        
        
        
        
        
        