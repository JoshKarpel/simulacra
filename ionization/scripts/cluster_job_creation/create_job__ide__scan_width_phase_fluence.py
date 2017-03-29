import os
import shutil
import argparse

import numpy as np

import compy as cp
import ionization as ion
import ionization.cluster as clu
import ionization.integrodiff as ide
from compy.units import *

if __name__ == '__main__':
    # get command line arguments
    parser = argparse.ArgumentParser(description = 'Create an Ionization vs Pulse Width, Phase, and Fluence job.')
    parser.add_argument('job_name',
                        type = str,
                        help = 'the name of the job')
    parser.add_argument('--dir', '-d',
                        action = 'store', default = os.getcwd(),
                        help = 'directory to put the job directory in. Defaults to cwd')
    parser.add_argument('--overwrite', '-o',
                        action = 'store_true',
                        help = 'force overwrite existing job directory if there is a name collision')
    parser.add_argument('--verbosity', '-v',
                        action = 'count', default = 0,
                        help = 'set verbosity level')
    parser.add_argument('--dry',
                        action = 'store_true',
                        help = 'do not attempt to actually submit the job')

    args = parser.parse_args()

    with cp.utils.Logger('compy', 'ionization', stdout_level = 31 - ((args.verbosity + 1) * 10)) as logger:
        # job type options
        job_processor = ion.cluster.IDEJobProcessor

        job_dir = os.path.join(args.dir, args.job_name)

        if os.path.exists(job_dir):
            if not args.overwrite and not clu.ask_for_bool('A job with that name already exists. Overwrite?', default = 'No'):
                clu.abort_job_creation()
            else:
                shutil.rmtree(job_dir)

        parameters = []

        test_charge = electron_charge * clu.ask_for_input('Test Particle Electric Charge (in electron charges)?', default = 1, cast_to = float)
        test_mass = electron_mass * clu.ask_for_input('Test Particle Mass (in electron masses)?', default = 1, cast_to = float)
        test_width = bohr_radius * clu.ask_for_input('Gaussian Test Wavefunction Width (in Bohr radii)?', default = 1, cast_to = float)

        prefactor = np.sqrt(pi) * (test_width ** 2) * ((test_charge / hbar) ** 2)
        tau_alpha = 4 * test_mass * (test_width ** 2) / hbar

        parameters.append(clu.Parameter(name = 'test_charge',
                                        value = test_charge))
        parameters.append(clu.Parameter(name = 'test_mass',
                                        value = test_mass))
        parameters.append(clu.Parameter(name = 'test_width',
                                        value = test_width))

        parameters.append(clu.Parameter(name = 'prefactor',
                                        value = prefactor))

        parameters.append(clu.Parameter(name = 'kernel',
                                        value = ide.gaussian_kernel))

        parameters.append(clu.Parameter(name = 'kernel_kwargs',
                                        value = dict(tau_alpha = tau_alpha)))

        parameters.append(clu.Parameter(name = 'minimum_time_step',
                                        value = asec * clu.ask_for_input('Minimum Time Step (in as)?', default = .01, cast_to = float)))

        parameters.append(clu.Parameter(name = 'maximum_time_step',
                                        value = asec * clu.ask_for_input('Maximum Time Step (in as)?', default = 10, cast_to = float)))

        parameters.append(clu.Parameter(name = 'time_step',
                                        value = asec * clu.ask_for_input('Initial Time Step (in as)?', default = .1, cast_to = float)))

        parameters.append(clu.Parameter(name = 'eps_on',
                                        value = clu.ask_for_input('Fractional Truncation Error Control on y or dydt?', default = 'dydt', cast_to = str)))

        parameters.append(clu.Parameter(name = 'eps',
                                        value = clu.ask_for_input('Fractional Truncation Error Limit?', default = 1e-6, cast_to = float)))

        time_bound_in_pw = clu.Parameter(name = 'time_bound_in_pw',
                                         value = clu.ask_for_input('Time Bound (in pulse widths)?', default = 30, cast_to = float))
        parameters.append(time_bound_in_pw)

        checkpoints = clu.ask_for_bool('Checkpoints?', default = True)
        parameters.append(clu.Parameter(name = 'checkpoints',
                                        value = checkpoints))
        if checkpoints:
            parameters.append(clu.Parameter(name = 'checkpoint_every',
                                            value = clu.ask_for_input('How many time steps per checkpoint?', default = 50, cast_to = int)))

        # pulse parameters
        pulse_parameters = []

        pulse_type_q = clu.ask_for_input('Pulse Type? [sinc/gaussian/sech]', default = 'sinc')
        if pulse_type_q == 'sinc':
            pulse_type = ion.SincPulse
        elif pulse_type_q == 'gaussian':
            pulse_type = ion.GaussianPulse
        elif pulse_type_q == 'sech':
            pulse_type = ion.SechPulse
        else:
            raise ValueError("Pulse type ({}) was not one of 'sinc', 'gaussian', or 'sech'".format(pulse_type_q))

        pulse_width = clu.Parameter(name = 'pulse_width',
                                    value = asec * np.array(clu.ask_for_eval('Pulse Widths (in as)?', default = 'np.array([50, 100, 200, 300, 400, 500])')),
                                    expandable = True)
        pulse_parameters.append(pulse_width)

        fluence = clu.Parameter(name = 'fluence',
                                value = (J / (cm ** 2)) * np.array(clu.ask_for_eval('Pulse Fluence (in J/cm^2)?', default = 'np.array([.1, 1, 5, 10, 20])')),
                                expandable = True)
        pulse_parameters.append(fluence)

        phases = clu.Parameter(name = 'phase',
                               value = np.linspace(0, twopi, clu.ask_for_input('Number of Phases?', default = 100, cast_to = int)),
                               expandable = True)
        pulse_parameters.append(phases)

        window_time_in_pw = clu.Parameter(name = 'window_time_in_pw',
                                          value = clu.ask_for_input('Window Time (in pulse widths)?', default = time_bound_in_pw.value - 1, cast_to = float))
        window_width_in_pw = clu.Parameter(name = 'window_width_in_pw',
                                           value = clu.ask_for_input('Window Width (in pulse widths)?', default = 0.5, cast_to = float))
        parameters.append(window_time_in_pw)
        parameters.append(window_width_in_pw)

        pulses = tuple(ion.SincPulse(**d,
                                     window = ion.SymmetricExponentialTimeWindow(window_time = d['pulse_width'] * window_time_in_pw.value,
                                                                                 window_width = d['pulse_width'] * window_width_in_pw.value))
                       for d in clu.expand_parameters_to_dicts(pulse_parameters))

        if pulse_type != ion.SincPulse:
            pulses = tuple(pulse_type(pulse_width = p.pulse_width, fluence = p.fluence, phase = p.phase,
                                      omega_carrier = p.omega_carrier,
                                      pulse_center = p.pulse_center,
                                      window = p.window)
                           for p in pulses)

        parameters.append(clu.Parameter(name = 'electric_potential_dc_correction',
                                        value = clu.ask_for_bool('Perform Electric Field DC Correction?', default = True)))

        parameters.append(clu.Parameter(name = 'electric_potential',
                                        value = pulses,
                                        expandable = True))

        print('Generating parameters...')

        spec_kwargs_list = clu.expand_parameters_to_dicts(parameters)
        specs = []

        print('Generating specifications...')

        for ii, spec_kwargs in enumerate(spec_kwargs_list):
            name = '{}_pw={}asec_flu={}Jcm2_phase={}pi'.format(
                pulse_type_q,
                uround(spec_kwargs['electric_potential'].pulse_width, asec),
                uround(spec_kwargs['electric_potential'].fluence, Jcm2),
                uround(spec_kwargs['electric_potential'].phase, pi)
            )

            time_bound = spec_kwargs['time_bound_in_pw'] * spec_kwargs['electric_potential'].pulse_width
            spec = ide.AdaptiveIntegroDifferentialEquationSpecification(name,
                                                                        file_name = str(ii),
                                                                        time_initial = -time_bound, time_final = time_bound,
                                                                        **spec_kwargs)

            spec.pulse_type = pulse_type
            spec.pulse_width = spec_kwargs['electric_potential'].pulse_width
            spec.fluence = spec_kwargs['electric_potential'].fluence
            spec.phase = spec_kwargs['electric_potential'].phase

            specs.append(spec)

        clu.specification_check(specs)

        submit_string = clu.format_chtc_submit_string(args.job_name, len(specs), checkpoints = checkpoints)
        clu.submit_check(submit_string)

        # point of no return
        shutil.rmtree(job_dir, ignore_errors = True)

        clu.create_job_subdirs(job_dir)
        clu.save_specifications(specs, job_dir)
        clu.write_specifications_info_to_file(specs, job_dir)
        clu.write_parameters_info_to_file(parameters + pulse_parameters, job_dir)

        job_info = {'name': args.job_name,
                    'job_processor_type': job_processor,  # set at top of if-name-main
                    'number_of_sims': len(specs),
                    'specification_type': specs[0].__class__,
                    'external_potential_type': specs[0].electric_potential.__class__,
                    }
        clu.write_job_info_to_file(job_info, job_dir)

        clu.write_submit_file(submit_string, job_dir)

        if not args.dry:
            clu.submit_job(job_dir)
