import h5py
from bilby.core.utils.io import recursively_load_dict_contents_from_group, decode_from_hdf5
import pandas as pd

def create_post_dict(file_name):
    with h5py.File(file_name, "r") as ff:
        data = recursively_load_dict_contents_from_group(ff, '/')

    posterior_samples = pd.DataFrame(data['C01:IMRPhenomXPHM']['posterior_samples'])
    meta = data['C01:IMRPhenomXPHM']['meta_data']['meta_data']
    config = data['C01:IMRPhenomXPHM']['config_file']['config']
    priors = data['C01:IMRPhenomXPHM']['priors']['analytic']
    

    return posterior_samples, meta, config, priors