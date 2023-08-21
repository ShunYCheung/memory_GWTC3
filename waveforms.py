import gwmemory
import bilby
import copy

from scipy.signal import get_window
from scipy.signal.windows import tukey
import utils


def osc_time_XPHM(times, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, iota, phase, 
                  **kwargs):
        trigger_time = kwargs.get('trigger_time')
        minimum_frequency = kwargs.get('minimum_frequency')
        sampling_frequency = kwargs.get('sampling_frequency')

        
        xphm = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', 
                                              minimum_frequency=minimum_frequency, 
                                              sampling_frequency=sampling_frequency, 
                                              distance=luminosity_distance, 
                                              q= mass_ratio, 
                                              total_mass=total_mass, 
                                              spin_1=[spin_1x, spin_1y, spin_1z], 
                                              spin_2=[spin_2x, spin_2y, spin_2z], 
                                              times=times)

        oscillatory, times = xphm.time_domain_oscillatory(inc=iota, phase=phase)
        window = tukey(surr_times.size, 0.1)

        plus_new = oscillatory["plus"]
        cross_new = oscillatory['cross']
        plus = plus_new * window
        cross = cross_new * window

        return {"plus": plus, "cross": cross}


def mem_time_XPHM(times, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, iota, phase, 
                  **kwargs):
        trigger_time = kwargs.get('trigger_time')
        minimum_frequency = kwargs.get('minimum_frequency')
        sampling_frequency = kwargs.get('sampling_frequency')

        xphm = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM',
                                               minimum_frequency=minimum_frequency, 
                                               sampling_frequency=sampling_frequency, 
                                               distance=luminosity_distance, 
                                               q= mass_ratio, 
                                               total_mass=total_mass, 
                                               spin_1=[spin_1x, spin_1y, spin_1z], 
                                               spin_2=[spin_2x, spin_2y,spin_2z], 
                                               times=times)
    
        oscillatory, times = xphm.time_domain_oscillatory(inc=iota, phase=phase)
        memory, times = xphm.time_domain_memory(inc=iota, phase=phase)

        window = tukey(surr_times.size, 0.05)

        plus_new2 = oscillatory2["plus"]+memory2["plus"]
        cross_new2 = oscillatory2['cross']+memory2["cross"]
        plus2 = plus_new2 * window
        cross2 = cross_new2 * window
 
        return {"plus": plus2, "cross": cross2}


def mem_freq_XPHM(frequencies, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
                  iota, phase,**kwargs):
        """
        Generates the frequency domain strain of the oscillatory + memory waveform using the approximant IMRPhenomXPHM.
        """
        
        # retrieve the key arguments
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        
        # define the time series based on the frequencies.
        # start_time = post_trigger_duration - duration
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=2-duration)
        series.frequency_array = frequencies
        
        # Create a generator
        xphm = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', 
                                              minimum_frequency=minimum_frequency,
                                              sampling_frequency=sampling_frequency, 
                                              distance=luminosity_distance, 
                                              q= mass_ratio, 
                                              total_mass=total_mass, 
                                              spin_1=[spin_1x, spin_1y, spin_1z], 
                                              spin_2=[spin_2x, spin_2y, spin_2z], 
                                              times=series.time_array)
        
        # call the time domain oscillatory and memory components. 
        osc, xphm_times = xphm.time_domain_oscillatory(inc=iota, phase=phase)
        mem, xphm_times = xphm.time_domain_memory(inc=iota, phase=phase)
        plus = osc['plus'] + mem['plus']
        cross = osc['cross'] + mem['cross']
        
        # apply a tapering before nfft.
        window = tukey(xphm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}
        
        # perform nfft to obtain frequency domain strain
        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)

        return waveform_fd


def osc_freq_XPHM(frequencies, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
                  iota, phase, **kwargs):
        """
        Generate the time domain strain of the oscillatory-only waveform using the approximant IMRPhenomXPHM.
        """
        
        # retrieve the key arguments
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        
        # define the time series based on the frequencies.
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=2-duration)
        series.frequency_array = frequencies
        
        # Create a generator
        waveform = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', 
                                                  minimum_frequency=minimum_frequency,
                                                  sampling_frequency=sampling_frequency, 
                                                  distance=luminosity_distance, 
                                                  q= mass_ratio, 
                                                  total_mass=total_mass, 
                                                  spin_1=[spin_1x, spin_1y, spin_1z], 
                                                  spin_2=[spin_2x, spin_2y, spin_2z], 
                                                  times=series.time_array)
        
        # call the time-domain oscillatory component only.
        osc, surr_times = waveform.time_domain_oscillatory(inc=iota, phase=phase)
        plus = osc['plus']
        cross = osc['cross']

        # apply tapering befor nfft.
        window = tukey(surr_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}
        
        # perform nfft to convert time-domain to frequency-domain strain.
        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)

        return waveform_fd

    
    
def mem_freq_XPHM_only(frequencies, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
                  iota, phase,**kwargs):
        
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        
        
                               
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=2-duration)
        series.frequency_array = frequencies
        
        xphm = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', 
                                              minimum_frequency=minimum_frequency, 
                                              sampling_frequency=sampling_frequency, 
                                              distance=luminosity_distance, 
                                              q= mass_ratio, 
                                              total_mass=total_mass, 
                                              spin_1=[spin_1x, spin_1y, spin_1z], 
                                              spin_2=[spin_2x, spin_2y, spin_2z], 
                                              times=series.time_array)
        
        osc, xphm_times = xphm.time_domain_oscillatory(inc=iota, phase=phase)
        mem, xphm_times = xphm.time_domain_memory(inc=iota, phase=phase)
        plus = mem['plus']
        cross = mem['cross']

        window = tukey(xphm_times.size, alpha=0.2)

        new_plus = plus * window
        new_cross = cross * window
        
        
        waveform = {'plus': new_plus, 'cross': new_cross}
        

        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)
        
        new_plus2 = osc['plus'] * window
        new_cross2 = osc['cross'] * window
        
        
        waveform2 = {'plus': new_plus2, 'cross': new_cross2}

        waveform_fd2 = utils.nfft(waveform2, sampling_frequency=sampling_frequency)
        
        return waveform_fd    


def mem_freq_XHM(frequencies, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
                  iota, phase,**kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=2-duration)
        series.frequency_array = frequencies
        
        xhm = gwmemory.waveforms.Approximant(name='IMRPhenomXHM', minimum_frequency=minimum_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[spin_1x, spin_1y, spin_1z], spin_2=[spin_2x, spin_2y, spin_2z], times=series.time_array)
        

        osc, xhm_times = xhm.time_domain_oscillatory(modes = modes, inc=iota, phase=phase)
        mem, xhm_times = xhm.time_domain_memory(modes = modes, inc=iota, phase=phase)          # no gamma_lmlm argument is needed, as it is deprecated.
        plus = osc['plus'] + mem['plus']
        cross = osc['cross'] + mem['cross']

        window = tukey(xhm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}

        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)

        return waveform_fd

    
def osc_freq_XHM(frequencies, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
                  iota, phase, **kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')

        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=2-duration)
        series.frequency_array = frequencies
        
        xhm = gwmemory.waveforms.Approximant(name='IMRPhenomXHM', minimum_frequency=minimum_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[spin_1x, spin_1y, spin_1z], spin_2=[spin_2x, spin_2y, spin_2z], 
                                                      times=series.time_array)


        osc, xhm_times = xhm.time_domain_oscillatory(modes = modes, inc=iota, phase=phase)
        plus = osc['plus']
        cross = osc['cross']

        window = tukey(xhm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}

        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)

        return waveform_fd

    