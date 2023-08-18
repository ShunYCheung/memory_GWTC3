import h5py

def create_post_dict(file_name):
    with h5py.File(file_name,'r+') as f1:
        wf_name1 = 'C01:IMRPhenomXPHM'
        wf_name2 = 'C01:IMRPhenomXPHM:HighSpin'
        if wf_name1 in list(f1.keys()):
            posterior_samples = f1['C01:IMRPhenomXPHM']['posterior_samples']
            samples = dict(
                # parameters I need
                mass_ratio = posterior_samples['mass_ratio'],
                total_mass = posterior_samples['total_mass'],
                luminosity_distance = posterior_samples['luminosity_distance'],
                spin_1x = posterior_samples['spin_1x'],
                spin_1y = posterior_samples['spin_1y'],
                spin_1z = posterior_samples['spin_1z'],
                spin_2x = posterior_samples['spin_2x'],
                spin_2y = posterior_samples['spin_2y'],
                spin_2z = posterior_samples['spin_2z'],
                iota = posterior_samples['iota'],
                phase = posterior_samples['phase'],

                # parameters I don't need but Bilby insists I include
                ra = posterior_samples['ra'],
                dec = posterior_samples['dec'],
                mass_1 = posterior_samples['mass_1'],
                mass_2 = posterior_samples['mass_2'],
                geocent_time = posterior_samples['geocent_time'],
                psi = posterior_samples['psi'],
                tilt_1 = posterior_samples['tilt_1'],
                tilt_2 = posterior_samples['tilt_2'],
                phi_12 = posterior_samples['phi_12'],
                phi_jl = posterior_samples['phi_jl'],
                a_1 = posterior_samples['a_1'],
                a_2 = posterior_samples['a_2'],


            )

        elif wf_name2 in list(f1.keys()):
            posterior_samples = f1['C01:IMRPhenomXPHM:HighSpin']['posterior_samples']
            samples = dict(
                
                mass_ratio = posterior_samples['mass_ratio'],
                total_mass = posterior_samples['total_mass'],
                luminosity_distance = posterior_samples['luminosity_distance'],
                spin_1x = posterior_samples['spin_1x'],
                spin_1y = posterior_samples['spin_1y'],
                spin_1z = posterior_samples['spin_1z'],
                spin_2x = posterior_samples['spin_2x'],
                spin_2y = posterior_samples['spin_2y'],
                spin_2z = posterior_samples['spin_2z'],
                iota = posterior_samples['iota'],
                phase = posterior_samples['phase'],

                ra = posterior_samples['ra'],
                dec = posterior_samples['dec'],
                mass_1 = posterior_samples['mass_1'],
                mass_2 = posterior_samples['mass_2'],
                geocent_time = posterior_samples['geocent_time'],
                psi = posterior_samples['psi'],
                tilt_1 = posterior_samples['tilt_1'],
                tilt_2 = posterior_samples['tilt_2'],
                phi_12 = posterior_samples['phi_12'],
                phi_jl = posterior_samples['phi_jl'],
                a_1 = posterior_samples['a_1'],
                a_2 = posterior_samples['a_2'],
            )
    

    return samples