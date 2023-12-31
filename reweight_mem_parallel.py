
import numpy as np
import bilby
import lal
import copy
import pickle
import gwpy
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.special import logsumexp

from waveforms import mem_freq_XPHM_v2


def reweight_mem_parallel(event_name, samples, args, priors, out_folder, outfile_name_w, data_file=None, psds = None, calibration=None, n_parallel=2):

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
        maximum_frequency = args['maximum_frequency']
        minimum_frequency = args['minimum_frequency']
        reference_frequency = args['reference_frequency']
        roll_off = args['tukey_roll_off']
        duration = ifo_list.duration
    else:
        sampling_frequency = args['sampling_frequency']
        maximum_frequency = args['maximum_frequency']
        minimum_frequency = args['minimum_frequency']
        reference_frequency = args['reference_frequency']
        roll_off = args['tukey_roll_off']
        duration = args['duration']
        post_trigger_duration = args['post_trigger_duration']
        trigger_time = args['trigger_time']
        
        detectors = args['detectors']
        if 'V1' in detectors:
            detectors.remove('V1')
        
        if args['trigger_time'] is not None:
            end_time = trigger_time + post_trigger_duration
            start_time = end_time - duration
        elif args['start_time'] is not None:
            start_time = args['start_time']
            end_time = args['end_time']
        else:
            print("Error: Trigger time or start time not extracted properly.")
            exit()
        
        ifo_list = call_data_GWOSC(logger, args, 
                                   calibration, samples, 
                                   detectors, start_time, 
                                   end_time, sampling_frequency, 
                                   roll_off, minimum_frequency, 
                                   maximum_frequency, psds_array=psds)
    
    waveform_name = args['waveform_approximant']
        
    # test if bilby oscillatory waveform = gwmemory oscillatory waveform.
    waveform_generator_osc = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model= bilby.gw.source.lal_binary_black_hole,
        parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(duration=duration,
                                minimum_frequency=minimum_frequency,
                                maximum_frequency=maximum_frequency,
                                sampling_frequency=sampling_frequency,
                                reference_frequency=reference_frequency,
                                waveform_approximant=waveform_name,
                               )
    )
    
    # define oscillatory + memory model using gwmemory.
    waveform_generator_full = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model= mem_freq_XPHM_v2,
        parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(duration=duration,
                                roll_off=roll_off,
                                minimum_frequency=minimum_frequency,
                                maximum_frequency=maximum_frequency,
                                sampling_frequency=sampling_frequency,
                                reference_frequency=reference_frequency,
                                bilby_generator = waveform_generator_osc)
    )
    
    if args['time_marginalization']=="True":
        print('time marginalisation on')
        time_marginalization = True
        jitter_time = True
    else:
        time_marginalization = False
        jitter_time = False
    
    if args['distance_marginalization']=="True":
        print('distance marginalisation on')
        distance_marginalization = True
    else:
        distance_marginalization = False
    if args['time_marginalization']:
        print('time marginalisation on')
        time_marginalization = True
        jitter_time = True
    else:
        time_marginalization = False
        jitter_time = False
    
    if args['distance_marginalization']:
        print('distance marginalisation on')
        distance_marginalization = True
    else:
        distance_marginalization = False
    
    
    priors2 = copy.copy(priors) # for some reason the priors change after putting it into the likelihood object. 
    # Hence, defining new ones for the second likelihood object.
    
    proposal_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_osc,
        time_marginalization = time_marginalization,
        distance_marginalization = distance_marginalization,
        distance_marginalization_lookup_table = "'TD.npz'.npz",
        jitter_time=jitter_time,
        priors = priors,
        reference_frame = args['reference_frame'],
        time_reference = args['time_reference'],
    )
    
    target_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_full,
        time_marginalization = time_marginalization,
        distance_marginalization = distance_marginalization,
        distance_marginalization_lookup_table = "'TD.npz'.npz",
        jitter_time=jitter_time,
        priors = priors2,
        reference_frame = args['reference_frame'],
        time_reference = args['time_reference'],
    )
    
    weights_list, weights_sq_list, proposal_likelihood_list, target_likelihood_list, ln_weights_list \
        = reweight_parallel(samples, proposal_likelihood, target_likelihood, priors2, n_parallel)
    
    # Calculate efficiency and Bayes factors.
    print('Reweighting results')

    neff = (np.sum(weights_list))**2 /np.sum(weights_sq_list)
    efficiency = neff/len(weights_list)
    bf = 1/(len(ln_weights_list)) * np.exp(logsumexp(ln_weights_list))
    lnbf = logsumexp(ln_weights_list) - np.log(len(ln_weights_list))
    
    print("effective no. of samples = {}".format(neff))
    print("{} percent efficiency".format(efficiency*100))
    print("Bayes factor = {}".format(bf))
    print("Log Bayes factor = {}".format(lnbf))
    
    # save weights, proposal and target likelihoods into a .txt file
    np.savetxt(out_folder+"/{0}_{1}.csv".format(outfile_name_w, waveform_name), 
               weights_list, 
               delimiter=",")
    np.savetxt(out_folder+"/{0}_{1}_proposal_likelihood.csv".format(outfile_name_w, waveform_name), 
               proposal_likelihood_list, 
               delimiter=",")
    np.savetxt(out_folder+"/{0}_{1}_target_likelihood.csv".format(outfile_name_w, waveform_name), 
               target_likelihood_list, 
               delimiter=",")

    return weights_list, bf
    
 
def reweighting(data, proposal_likelihood, target_likelihood, priors):
    logger = bilby.core.utils.logger
    ln_weights_list=[]
    weights_list = []
    weights_sq_list = []
    proposal_likelihood_list = []
    target_likelihood_list = []
    
    reference_dict = {'geocent_time': priors['geocent_time'],
                 'luminosity_distance': priors['luminosity_distance']}
    
    length = data.shape[0]
    
    for i in range(length):
        use_stored_likelihood=False
        
        if i % 1000 == 0:
            print("reweighted {0} samples out of {1}".format(i+1, length))
            logger.info("{:0.2f}".format(i / length * 100) + "%")
        
        if use_stored_likelihood:
            proposal_likelihood_values = data['log_likelihood'].iloc[i]
            log_likelihood_GWOSC = proposal_likelihood_values
            #print("GWOSC values")
            #print("log likelihood ratio from GWOSC = ", proposal_likelihood_values)
            #print("log likelihood from GWOSC = ", log_likelihood_GWOSC)
            #print("log noise evidence from GWOSC = ", ln_noise_evidence)
            
        else:
            proposal_likelihood.parameters.update(data.iloc[i].to_dict())
            proposal_likelihood.parameters.update(reference_dict)
            proposal_likelihood_values = proposal_likelihood.log_likelihood_ratio()
            #print("log likelihood ratio from proposal = ", proposal_likelihood_values)
            #print("log likelihood from proposal = ",  proposal_likelihood.log_likelihood())
            #print("log noise evidence from proposal = ", proposal_likelihood.noise_log_likelihood())
            
        reference_dict = {'geocent_time': priors['geocent_time'],
                 'luminosity_distance': priors['luminosity_distance']}
        target_likelihood.parameters.update(data.iloc[i].to_dict())
        target_likelihood.parameters.update(reference_dict)
        target_likelihood_values = target_likelihood.log_likelihood_ratio()
        
        #print("My values")
        #print("log likelihood ratio = ", target_likelihood_values)
        #print("log likelihood (not ratio) = ", target_likelihood.log_likelihood())
        #print("ln_noise_evidence = ", target_likelihood.noise_log_likelihood())
        
        ln_weights = target_likelihood_values-proposal_likelihood_values

        #print("difference in log likelihood", ln_weights)
        
        weights = np.exp(target_likelihood_values-proposal_likelihood_values)
        weights_sq = np.square(weights)
        weights_list.append(weights)
        weights_sq_list.append(weights_sq)
        proposal_likelihood_list.append(proposal_likelihood_values)
        target_likelihood_list.append(target_likelihood_values)
        ln_weights_list.append(ln_weights)

    return weights_list, weights_sq_list, proposal_likelihood_list, target_likelihood_list, ln_weights_list


def reweight_parallel(samples, proposal_likelihood, target_likelihood, priors, n_parallel=2):
    print("activate multiprocessing")
    p = mp.Pool(n_parallel)

    data=samples
    new_data = copy.deepcopy(data)  
    posteriors = np.array_split(new_data, n_parallel)

    new_results = []
    for i in range(n_parallel):
        res = copy.deepcopy(posteriors[i])
        new_results.append(res)
 
    iterable = [(new_result, proposal_likelihood, target_likelihood, priors) for new_result in new_results]
    res = p.starmap(reweighting, iterable)
    
    p.close()
    weights_list_comb = np.concatenate([r[0] for r in res])
    weights_sq_list_comb = np.concatenate([r[1] for r in res])
    proposal_comb = np.concatenate([r[2] for r in res])
    target_comb = np.concatenate([r[3] for r in res])    
    ln_weights_comb = np.concatenate([r[4] for r in res])      
    return weights_list_comb, weights_sq_list_comb, proposal_comb, target_comb, ln_weights_comb


def call_data_GWOSC(logger, args, calibration, samples, detectors, start_time, end_time, sampling_frequency, roll_off, minimum_frequency, maximum_frequency, psds_array=None, plot=False):
    
    ifo_list = bilby.gw.detector.InterferometerList([])
    
    # define interferometer objects
    for det in detectors:   
        logger.info("Downloading analysis data for ifo {}".format(det))
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        
        channel_type = args['channel_dict'][det]
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
    
        # define some attributes in ifo
        ifo.strain_data.roll_off = roll_off
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        
        # set data as the strain data
        ifo.strain_data.set_from_gwpy_timeseries(data)
        
        # compute the psd
        if det in psds_array.keys():
            print("Using pre-computed psd from results file")
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psds_array[det][: ,0], psd_array=psds_array[det][:, 1]
            )
        else:
            print('Error: PSD is missing!')
            exit()

        ifo_list.append(ifo)

    return ifo_list

