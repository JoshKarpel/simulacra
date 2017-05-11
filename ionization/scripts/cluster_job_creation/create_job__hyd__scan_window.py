import os
import shutil
import argparse

import numpy as np

import compy as cp
import compy.cluster as clu
import ionization as ion
import ionization.cluster as iclu
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

    with cp.utils.LogManager('compy', 'ionization', stdout_level = 31 - ((args.verbosity + 1) * 10)) as logger:
        # job type options
        job_processor = iclu.PulseJobProcessor

        job_dir = os.path.join(args.dir, args.job_name)

        if os.path.exists(job_dir):
            if not args.overwrite and not clu.ask_for_bool('A job with that name already exists. Overwrite?', default = 'No'):
                clu.abort_job_creation()
            else:
                shutil.rmtree(job_dir)

        parameters = []

        # get input from the user to define the job
        spec_type, mesh_kwargs = iclu.ask_mesh_type()

        initial_state = clu.Parameter(name = 'initial_state',
                                      value = ion.HydrogenBoundState(clu.ask_for_input('Initial State n?', default = 1, cast_to = int),
                                                                     clu.ask_for_input('Initial State l?', default = 0, cast_to = int)))
        parameters.append(initial_state)

        numeric_basis_q = False
        if spec_type == ion.SphericalHarmonicSpecification:
            numeric_basis_q = clu.ask_for_bool('Use numeric eigenstate basis?', default = True)
            if numeric_basis_q:
                parameters.append(clu.Parameter(name = 'use_numeric_eigenstates_as_basis',
                                                value = True))
                parameters.append(clu.Parameter(name = 'numeric_eigenstate_l_max',
                                                value = clu.ask_for_input('Numeric Eigenstate Maximum l?', default = 10, cast_to = int)))
                parameters.append(clu.Parameter(name = 'numeric_eigenstate_energy_max',
                                                value = eV * clu.ask_for_input('Numeric Eigenstate Max Energy (in eV)?', default = 50, cast_to = float)))

        if not numeric_basis_q:
            if clu.ask_for_bool('Overlap only with initial state?', default = 'yes'):
                parameters.append(clu.Parameter(name = 'test_states',
                                                value = [initial_state.value]))
            else:
                largest_n = clu.ask_for_input('Largest Bound State n to Overlap With?', default = 5, cast_to = int)
                parameters.append(clu.Parameter(name = 'test_states',
                                                value = tuple(ion.HydrogenBoundState(n, l) for n in range(largest_n + 1) for l in range(n))))

        parameters.append(clu.Parameter(name = 'time_step',
                                        value = asec * clu.ask_for_input('Time Step (in as)?', default = 1, cast_to = float)))

        time_initial_in_pw = clu.Parameter(name = 'initial_time_in_pw',
                                           value = clu.ask_for_input('Initial Time (in pulse widths)?', default = -35, cast_to = float))
        parameters.append(time_initial_in_pw)

        parameters.append(clu.Parameter(name = 'final_time_in_pw',
                                        value = clu.ask_for_input('Final Time (in pulse widths)?', default = 40, cast_to = float)))

        extra_time = clu.Parameter(name = 'extra_time',
                                   value = asec * clu.ask_for_input('Extra Time (in as)?', default = 0, cast_to = float))
        parameters.append(extra_time)

        checkpoints = clu.ask_for_bool('Checkpoints?', default = True)
        parameters.append(clu.Parameter(name = 'checkpoints',
                                        value = checkpoints))
        if checkpoints:
            parameters.append(clu.Parameter(name = 'checkpoint_every',
                                            value = clu.ask_for_input('How many time steps per checkpoint?', default = 50, cast_to = int)))

        outer_radius_default = mesh_kwargs['outer_radius'] / bohr_radius
        parameters.append(clu.Parameter(name = 'mask',
                                        value = ion.RadialCosineMask(inner_radius = bohr_radius * clu.ask_for_input('Mask Inner Radius (in Bohr radii)?', default = outer_radius_default * .8, cast_to = float),
                                                                     outer_radius = bohr_radius * clu.ask_for_input('Mask Outer Radius (in Bohr radii)?', default = outer_radius_default, cast_to = float),
                                                                     smoothness = clu.ask_for_input('Mask Smoothness?', default = 8, cast_to = int))))

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
                                    value = asec * np.array(clu.ask_for_eval('Pulse Widths (in as)?', default = '[50, 100, 200, 300, 400, 500]')),
                                    expandable = True)
        pulse_parameters.append(pulse_width)

        fluence = clu.Parameter(name = 'fluence',
                                value = (J / (cm ** 2)) * np.array(clu.ask_for_eval('Pulse Fluence (in J/cm^2)?', default = '[.1, 1, 5, 10, 20]')),
                                expandable = True)
        pulse_parameters.append(fluence)

        phases = clu.Parameter(name = 'phase',
                               value = np.array(clu.ask_for_eval('Pulse CEP (in rad)?', default = 'np.linspace(0, pi, 50)')),
                               expandable = True)
        pulse_parameters.append(phases)

        window_time_in_pw = clu.Parameter(name = 'window_time_in_pw',
                                          value = clu.ask_for_eval('Window Time (in pulse widths)?', default = 'np.linspace(3, 50, 50)'))
        window_width_in_pw = clu.Parameter(name = 'window_width_in_pw',
                                           value = clu.ask_for_input('Window Width (in pulse widths)?', default = 0.2, cast_to = float))
        parameters.append(window_time_in_pw)
        parameters.append(window_width_in_pw)

        pulses = tuple(ion.SincPulse(**d,
                                     window = ion.SymmetricExponentialTimeWindow(window_time = d['pulse_width'] * window,
                                                                                 window_width = d['pulse_width'] * window_width_in_pw.value))
                       for d in clu.expand_parameters_to_dicts(pulse_parameters) for window in window_time_in_pw.value)

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

        parameters.append(clu.Parameter(name = 'store_norm_by_l',
                                        value = clu.ask_for_bool('Store Norm-by-L?', default = False)))

        parameters.append(clu.Parameter(name = 'store_data_every',
                                        value = clu.ask_for_input('Store Data Every?', default = 1, cast_to = int)))

        parameters.append(clu.Parameter(name = 'snapshot_indices',
                                        value = clu.ask_for_eval('Snapshot Indices?', default = '[]')))

        snapshot_times = asec * np.array(clu.ask_for_eval('Snapshot Times (in asec)?', default = '[]'))
        snapshot_times_in_pw = np.array(clu.ask_for_eval('Snapshot Times (in pulse widths)?', default = '[]'))

        print('Generating parameters...')

        spec_kwargs_list = clu.expand_parameters_to_dicts(parameters)
        specs = []

        print('Generating specifications...')

        for ii, spec_kwargs in enumerate(spec_kwargs_list):
            electric_potential = spec_kwargs['electric_potential']
            name = '{}_pw={}asec_flu={}Jcm2_phase={}pi_window={}asec'.format(
                pulse_type_q,
                uround(electric_potential.pulse_width, asec),
                uround(electric_potential.fluence, Jcm2),
                uround(electric_potential.phase, pi),
                uround(electric_potential.window.window_time, asec),
            )

            time_initial = spec_kwargs['initial_time_in_pw'] * electric_potential.pulse_width
            time_final = spec_kwargs['final_time_in_pw'] * electric_potential.pulse_width + extra_time
            snapshot_times = np.concatenate((snapshot_times, electric_potential.pulse_width * snapshot_times_in_pw))

            spec = spec_type(name,
                             file_name = str(ii),
                             time_initial = time_initial, time_final = time_final,
                             **mesh_kwargs, **spec_kwargs)

            spec.pulse_type = pulse_type
            spec.pulse_width = electric_potential.pulse_width
            spec.fluence = electric_potential.fluence
            spec.phase = electric_potential.phase

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
