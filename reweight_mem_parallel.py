"""
Code for a function "reweight" to calculate the weights to turn a proposal posterior into a target posterior.
Also calculates the Bayes factor between no memory and memory hypothesis. 

Author: Shun Yin Cheung
"""

import sys

import h5py
import pandas as pd
import numpy as np
import bilby
import gwmemory
import lal
import json
import copy
import pickle
from tqdm import tqdm

import gwpy
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
import matplotlib.pyplot as plt

import utils
from utils import get_alpha
import multiprocessing as mp
import functools
from scipy.signal import get_window
from scipy.signal.windows import tukey
from scipy.special import logsumexp

from waveforms import osc_freq_XPHM, mem_freq_XPHM, osc_freq_XHM, mem_freq_XHM



def reweight_mem_parallel(event_name, samples, meta, config, priors, detectors, out_folder, outfile_name_w, data_file=None, psds = None, calibration=None, n_parallel=2):

    """
    A function that calculates the weights to turn a posterior with a proposal distribution into 
    the posterior of the target distribution. The weights are saved into a .csv file 
    at the end of the code.
    """

    logger = bilby.core.utils.logger

    
    # adds in detectors and the specs for the detectors. 
    if data_file is not None:
        print("opening {}".format(data_file))
        with open(data_file, 'rb') as f:
            data_dump = pickle.load(f)
        ifo_list = data_dump.interferometers
        sampling_frequency = ifo_list.sampling_frequency
        duration = ifo_list.duration
        minimum_frequency = fmin
    else:
        sampling_frequency = meta['sampling_frequency'][0]
        maximum_frequency = 896 #meta['f_final'][0]
        minimum_frequency = meta['f_low'][0]
        reference_frequency = float(config['reference-frequency'][0])
        roll_off = float(config['tukey-roll-off'][0])
        duration = meta['duration'][0]
        post_trigger_duration = float(config['post-trigger-duration'][0])
        trigger_time = float(config['trigger-time'][0])
        end_time = trigger_time + post_trigger_duration
        start_time = end_time - duration
        psd_duration = 32*duration    # deprecated
        psd_start_time = start_time - psd_duration
        psd_end_time = start_time
        
        ifo_list = call_data_GWOSC(logger, config, 
                                   calibration, samples, detectors,
                                   start_time, end_time, 
                                   psd_start_time, psd_end_time, 
                                   duration, sampling_frequency, 
                                   roll_off, minimum_frequency, maximum_frequency,
                                   psds_array=psds)

    
    
    waveform_name = meta['approximant'][0]
    print(ifo_list)
    if waveform_name == "IMRPhenomXPHM":
        osc_model = osc_freq_XPHM
        mem_model = mem_freq_XPHM

    elif waveform_name == "IMRPhenomXHM":
        osc_model = osc_freq_XHM
        mem_model = mem_freq_XHM
    
    
    # test if vanilla = osc
    waveform_generator_vanilla = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model= bilby.gw.source.lal_binary_black_hole,
        parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(duration=duration,
                                roll_off=roll_off,
                                minimum_frequency=minimum_frequency,
                                maximum_frequency=maximum_frequency,
                                sampling_frequency=sampling_frequency,
                                reference_frequency=reference_frequency,
                                waveform_approximant=waveform_name,
                                catch_waveform_errors=True)

    )
    
    waveform_generator_osc = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model= osc_model,
        waveform_arguments=dict(duration=duration,
                                roll_off=roll_off,
                                minimum_frequency=minimum_frequency,
                                sampling_frequency=sampling_frequency)

    )

    waveform_generator_mem = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model= mem_model,
        waveform_arguments=dict(duration=duration,
                                roll_off=roll_off,
                                minimum_frequency=minimum_frequency,
                                sampling_frequency=sampling_frequency)

    )
    
    
    if meta['time_marginalization'][0]=="True":
        print('time marginalisation on')
        time_marginalization = True
        jitter_time = True
    else:
        time_marginalization = False
        jitter_time = False
    
    if meta['distance_marginalization'][0]=="True":
        print('distance marginalisation on')
        distance_marginalization = True
    else:
        distance_marginalization = False
    if calibration is not None:
        print("calibration marginalisation on")
        calibration_marginalization = True
        calibration_lookup_table = calibration
        #calibration_lookup_table = {"H1":"/home/daniel.williams/events/O3/event_repos/GW150914/C01_offline/calibration/H1.dat",
        #                            "L1":"/home/daniel.williams/events/O3/event_repos/GW150914/C01_offline/calibration/L1.dat", }
        #print(np.shape(calibration_lookup_table['H1']))
    else:
        calibration_marginalization = False

    
    priors2 = copy.copy(priors) # for some reason the priors change after putting it into the likelihood. 
    # Hence, defining new ones for the seocnd likelihood.
    
    old_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_osc,
        time_marginalization = time_marginalization,
        distance_marginalization = distance_marginalization,
        distance_marginalization_lookup_table = "'TD.npz'.npz",
        calibration_marginalization = calibration_marginalization,
        jitter_time=jitter_time,
        priors = priors,
        reference_frame = config['reference-frame'][0],
        calibration_lookup_table = calibration_lookup_table,
        #number_of_response_curves = 300,
    )
    
    file_paths={"H1":"/home/daniel.williams/events/O3/event_repos/GW150914/C01_offline/calibration/H1.dat",
                                   "L1":"/home/daniel.williams/events/O3/event_repos/GW150914/C01_offline/calibration/L1.dat", }
    
    # for some reason the calibration model change to Recalibrate after using it in old lieklihodo hence, defining a new one. 
    if calibration is not None:
        for i in range(len(detectors)):
            """
            ifo_list[i].calibration_model = bilby.gw.calibration.Precomputed(
                    f"{ifo_list[i].name}",
                    calibration[f"{ifo_list[i].name}"],
                    frequency_array=ifo_list[i].frequency_array[ifo_list[i].frequency_mask],
                    parameters= samples,
                )
            """
            ifo_list[i].calibration_model = bilby.gw.calibration.Precomputed.from_envelope_file(
                    file_paths[f"{ifo_list[i].name}"],
                    frequency_array=ifo_list[i].frequency_array[ifo_list[i].frequency_mask],
                    n_nodes=int(config['spline-calibration-nodes'][0]),
                    label=ifo_list[i].name,
                    n_curves=1000,
                )
            

    
    new_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_vanilla,
        time_marginalization = time_marginalization,
        distance_marginalization = distance_marginalization,
        distance_marginalization_lookup_table = "'TD.npz'.npz",
        calibration_marginalization = calibration_marginalization,
        jitter_time=jitter_time,
        priors = priors2,
        reference_frame = config['reference-frame'][0],
        calibration_lookup_table = calibration_lookup_table,
        #number_of_response_curves = 300,
    )
    
    # Define the proposal likelihood which is stored in the data file.
    weights_list, weights_sq_list, proposal_likelihood_list, target_likelihood_list, ln_weights_list = reweight_parallel(samples, 
                                                                                                  old_likelihood, 
                                                                                                  new_likelihood,
                                                                                                  n_parallel)

    # Calulate the effective number of samples.
    neff = (np.sum(weights_list))**2 /np.sum(weights_sq_list)
    print("effective no. of samples = {}".format(neff))

    efficiency = neff/len(weights_list)
    print("{} percent efficiency".format(efficiency*100))


    # Calculate the Bayes factor
    bf_v2 = 1/(len(ln_weights_list)) * np.exp(logsumexp(ln_weights_list))
    print("new Bayes factor = {}".format(bf_v2))
    
    lnbf_v2 = logsumexp(ln_weights_list) - np.log(len(ln_weights_list))
    print("new log Bayes factor = {}".format(lnbf_v2))
    
    
    # save into textfile
    np.savetxt(out_folder+"/{0}_{1}.csv".format(outfile_name_w, waveform_name), 
               weights_list, 
               delimiter=",")
    np.savetxt(out_folder+"/{0}_{1}_proposal_likelihood.csv".format(outfile_name_w, waveform_name), 
               proposal_likelihood_list, 
               delimiter=",")
    np.savetxt(out_folder+"/{0}_{1}_target_likelihood.csv".format(outfile_name_w, waveform_name), 
               target_likelihood_list, 
               delimiter=",")

    
    return weights_list, bf_v2
    


def reweighting(data, old_likelihood, new_likelihood):
    ln_weights_list=[]
    weights_list = []
    weights_sq_list = []
    proposal_likelihood_list = []
    target_likelihood_list = []
    length = data.shape[0]
    for i in range(1):
        use_stored_likelihood=True
        
        if i % 1000 == 0:
            print("reweighted {0} samples out of {1}".format(i+1, length))
        
        if use_stored_likelihood:
            old_likelihood_values = data['log_likelihood'].iloc[i]
            #print(data['log_likelihood'].iloc[i])
            #print(data.log_likelihood.iloc[i])
        else:
            old_likelihood.parameters = data.iloc[i].to_dict()
            old_likelihood_values = old_likelihood.log_likelihood_ratio()
    
        new_likelihood.parameters = data.iloc[i].to_dict()
        new_likelihood_values = new_likelihood.log_likelihood_ratio()
        print("ln_noise_evidence",new_likelihood.noise_log_likelihood())
        
        ln_weights = new_likelihood_values-old_likelihood_values
        weights = np.exp(new_likelihood_values-old_likelihood_values)
        weights_sq = np.square(weights)
        weights_list.append(weights)
        weights_sq_list.append(weights_sq)
        proposal_likelihood_list.append(old_likelihood_values)
        target_likelihood_list.append(new_likelihood_values)
        ln_weights_list.append(ln_weights)

    return weights_list, weights_sq_list, proposal_likelihood_list, target_likelihood_list, ln_weights_list



def reweight_parallel(samples, old_likelihood, new_likelihood, n_parallel=2):
    print("activate multiprocessing")
    p = mp.Pool(n_parallel)

    data=samples
    new_data = copy.deepcopy(data)
  
    posteriors = np.array_split(new_data, n_parallel)
  
    new_results = []
    for i in range(n_parallel):
        res = copy.deepcopy(posteriors[i])
        new_results.append(res)
 
    iterable = [(new_result, old_likelihood, new_likelihood) for new_result in new_results]

    res = p.starmap(reweighting, iterable)
 
    p.close()
    weights_list_comb = np.concatenate([r[0] for r in res])
    weights_sq_list_comb = np.concatenate([r[1] for r in res])
    proposal_comb = np.concatenate([r[2] for r in res])
    target_comb = np.concatenate([r[3] for r in res])    
    ln_weights_comb = np.concatenate([r[4] for r in res])      
    return weights_list_comb, weights_sq_list_comb, proposal_comb, target_comb, ln_weights_comb



def call_data_GWOSC(logger, config, calibration, samples, detectors, start_time, end_time, psd_start_time, psd_end_time, duration, sampling_frequency, roll_off, minimum_frequency, maximum_frequency, psds_array=None, plot=False):
    ifo_list = bilby.gw.detector.InterferometerList([])
    
    # define interferometer objects
    for det in detectors:   
        logger.info("Downloading analysis data for ifo {}".format(det))
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        
        channel_type = 'DCS-CALIB_STRAIN_C02'
        
        channel = f"{det}:{channel_type}"
        
        kwargs = dict(
            start=start_time,
            end=end_time,
            verbose=False,
            allow_tape=True,
        )

        type_kwargs = dict(
            dtype="float64",
            subok=True,
            copy=False,
        )
        
        data = gwpy.timeseries.TimeSeries.get(channel, **kwargs).astype(
                **type_kwargs)
    
        # Resampling timeseries to sampling_frequency using lal.
        lal_timeseries = data.to_lal()
        lal.ResampleREAL8TimeSeries(
            lal_timeseries, float(1/sampling_frequency)
        )
        data = TimeSeries(
            lal_timeseries.data.data,
            epoch=lal_timeseries.epoch,
            dt=lal_timeseries.deltaT
        )
    
        #print('2', data)
        # define some attributes in ifo
        ifo.strain_data.roll_off = roll_off
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        
        # set data as the strain data
        ifo.strain_data.set_from_gwpy_timeseries(data)
        
        
        # compute the psd
        if psds_array is not None:
            print("Using pre-computed psd from results file")
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psds_array[det][: ,0], psd_array=psds_array[det][:, 1]
            )
        else:
            logger.info("Downloading psd data for ifo {}".format(det))                  
            psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, sample_rate=16384)

            # again, we resample the psd_data using lal.
            psd_lal_timeseries = psd_data.to_lal()
            lal.ResampleREAL8TimeSeries(
                psd_lal_timeseries, float(1/sampling_frequency)
            )
            psd_data = TimeSeries(
                psd_lal_timeseries.data.data,
                epoch=psd_lal_timeseries.epoch,
                dt=psd_lal_timeseries.deltaT
            )

            psd_alpha = 2 * roll_off / duration  
            
            psd = psd_data.psd(        
                fftlength=duration, overlap=0.5*duration, window=("tukey", psd_alpha), method="median"
            )

            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                frequency_array=psd.frequencies.value, psd_array=psd.value
            )
        
        """
        ifo.calibration_model = bilby.gw.calibration.CubicSpline(prefix=f"recalib_{ifo.name}_",
                minimum_frequency=ifo.minimum_frequency,
                maximum_frequency=ifo.maximum_frequency,
                n_points=int(config['spline-calibration-nodes'][0]))
        """
        if calibration is not None:
            print(f'{det}: Using pre-computed calibration model')
            model = bilby.gw.calibration.Precomputed
            det = ifo.name
            file_paths={"H1":"/home/daniel.williams/events/O3/event_repos/GW150914/C01_offline/calibration/H1.dat",
                        "L1":"/home/daniel.williams/events/O3/event_repos/GW150914/C01_offline/calibration/L1.dat", }
            
            ifo.calibration_model = model.from_envelope_file(
                file_paths[det],
                frequency_array=ifo.frequency_array[ifo.frequency_mask],
                n_nodes=int(config['spline-calibration-nodes'][0]),
                label=det,
                n_curves=1000,
            )
            """
            ifo.calibration_model = model(
                f"{det}",
                calibration[det],
                frequency_array=ifo.frequency_array[ifo.frequency_mask],
                parameters= samples,
            )
            """
        
        ifo_list.append(ifo)
        if plot==True:
            plt.figure()
            plt.loglog(ifo.frequency_array, ifo.power_spectral_density_array)
            plt.show()

    return ifo_list

##############################################

