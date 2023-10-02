from reweight_mem_parallel import reweight_mem_parallel
import json
from create_post_dict import create_post_dict, extract_relevant_info, process_bilby_result
from event_table import call_event_table
import sys
import pandas as pd
import numpy as np
import os
import bilby

event_number = int(sys.argv[1])

if __name__ == '__main__':

    events = [call_event_table()[event_number]]
    
    for count, i in enumerate(events):
        event_name, file_path, trigger_time, duration, waveform, data_file = i
        print(f"opening {file_path}")
        
        extension = os.path.splitext(file_path)[1].lstrip('.')
        if 'h5' in extension:
            samples_dict, meta_dict, config_dict, priors_dict, psds, calibration = create_post_dict(file_path, waveform)
            args = extract_relevant_info(meta_dict, config_dict)
        elif 'json' in extension:
            result = bilby.core.result.read_in_result(file_path)
            samples_dict = result.posterior
            args = process_bilby_result(result.meta_data['command_line_args'])
            priors_dict = result.priors
            psds=None
        else:
            print('Cannot recognise file type.')
            
        print("reweighting {}".format(event_name), "{0}/{1} events reweighted".format(count+1, len(events)))
        weights, bf = reweight_mem_parallel(event_name, 
                                            samples_dict, 
                                            args,
                                            priors_dict,
                                            "/home/shunyin.cheung/memory_GWTC3/run2",
                                            "weights_{}".format(event_name), 
                                            data_file=data_file,
                                            psds = psds,
                                            calibration = None,
                                            n_parallel=4)
        
        
        
        
        
        
        
        