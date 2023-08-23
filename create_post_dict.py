import h5py
from bilby.core.utils.io import recursively_load_dict_contents_from_group, decode_from_hdf5, decode_bilby_json
import pandas as pd
from bilby.core.prior import Prior, PriorDict, DeltaFunction, ConditionalDeltaFunction
#from bilby.core.prior.dict import from_dictionary
import json


def create_post_dict(file_name):
    with h5py.File(file_name, "r") as ff:
        data = recursively_load_dict_contents_from_group(ff, '/')

    posterior_samples = pd.DataFrame(data['C01:IMRPhenomXPHM']['posterior_samples'])
    meta = data['C01:IMRPhenomXPHM']['meta_data']['meta_data']
    config = data['C01:IMRPhenomXPHM']['config_file']['config']
    priors = data['C01:IMRPhenomXPHM']['priors']['analytic']
    psds = data['C01:IMRPhenomXPHM']['psds']
    calibration = data['C01:IMRPhenomXPHM']['calibration_envelope']
    #calibration = data['C01:IMRPhenomXPHM']['priors']['calibration']
    
    for key in list(priors.keys()):
        val = priors[key][0]
        priors[key] = val
    #del priors['chirp_mass']
    #del priors['mass_ratio']
    
    val = data['C01:IMRPhenomXPHM']['priors']['analytic']['chirp_mass']
    cl = val.split("(")
    if cl[0] == "UniformInComponentsChirpMass":
        complete_cl = "bilby.gw.prior.UniformInComponentsChirpMass("
        cl[0] = complete_cl
        string = ''.join(cl)
        data['C01:IMRPhenomXPHM']['priors']['analytic']['chirp_mass']=string
        #print(string)
    
    val = data['C01:IMRPhenomXPHM']['priors']['analytic']['mass_ratio']
    cl = val.split("(")
    if cl[0] == "UniformInComponentsMassRatio":
        complete_cl = "bilby.gw.prior.UniformInComponentsMassRatio("
        cl[0] = complete_cl
        string = ''.join(cl)
        data['C01:IMRPhenomXPHM']['priors']['analytic']['mass_ratio']=string
        #print(string)
        
    
    
    priors = PriorDict(data['C01:IMRPhenomXPHM']["priors"]['analytic'])
    print(calibration)
    return posterior_samples, meta, config, priors, psds, calibration