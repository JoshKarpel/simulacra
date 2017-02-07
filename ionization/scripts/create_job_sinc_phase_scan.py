# create an Ionization vs Pulse Width job

import sys
import os
import shutil
import logging
import argparse
import json
import pickle
import functools as ft

import numpy as np

import compy as cp
import ionization as ion
import ionization.cluster as clu
from compy.units import *

if __name__ == '__main__':
    # get command line arguments
    parser = argparse.ArgumentParser(description = 'Create an Ionization vs Pulse Width job.')
    parser.add_argument('job_name',
                        type = str,
                        help = 'the name of the job')
    parser.add_argument('--dir', '-d',
                        action = 'store', default = os.getcwd(),
                        help = 'directory to put the job directory in')
    parser.add_argument('--overwrite', '-o',
                        action = 'store_true',
                        help = 'force overwrite existing job directory if there is a name collision')
    parser.add_argument('--verbosity', '-v',
                        action = 'count', default = 0)

    args = parser.parse_args()

    with cp.utils.Logger('compy', 'ionization', stdout_level = 31 - ((args.verbosity + 1) * 10)) as logger:
        # job type options
        job_processor = ion.cluster.SincPulseJobProcessor

        job_dir = os.path.join(args.dir, args.job_name)

        if os.path.exists(job_dir):
            if not args.overwrite and not clu.ask_for_bool('A job with that name already exists. Overwrite?', default = 'No'):
                clu.abort_job_creation()

        parameters = []

        # get input from the user to define the job
        spec_type, mesh_kwargs = clu.ask_mesh_type()

        initial_state = clu.Parameter(name = 'initial_state',
                                      value = ion.HydrogenBoundState(clu.ask_for_input('Initial State n?', default = 1, cast_to = int),
                                                                     clu.ask_for_input('Initial State l?', default = 0, cast_to = int)))
        parameters.append(initial_state)

        if clu.ask_for_bool('Overlap only with initial state?', default = 'yes'):
            parameters.append(clu.Parameter(name = 'test_states',
                                            value = [initial_state.value]))
        else:
            largest_n = clu.ask_for_input('Largest Bound State n to Overlap With?', default = 5, cast_to = int)
            parameters.append(clu.Parameter(name = 'test_states',
                                            value = tuple(ion.HydrogenBoundState(n, l) for n in range(largest_n + 1) for l in range(n))))

        parameters.append(clu.Parameter(name = 'time_step',
                                        value = asec * clu.ask_for_input('Time Step (in as)?', default = 1, cast_to = float)))

        time_bound_in_pw = clu.Parameter(name = 'time_bound_in_pw',
                                         value = clu.ask_for_input('Time Bound (in pulse widths)?', default = 30, cast_to = float))
        parameters.append(time_bound_in_pw)

        minimum_time_final = clu.Parameter(name = 'minimum_time_final',
                                           value = asec * clu.ask_for_input('Minimum Final Time (in as)?', default = 0, cast_to = float))
        parameters.append(minimum_time_final)
        if minimum_time_final.value > 0:
            parameters.append(clu.Parameter(name = 'extra_time_step',
                                            value = asec * clu.ask_for_input('Extra Time Step (in as)?', default = 1, cast_to = float)))

        checkpoints = clu.ask_for_bool('Checkpoints?', default = True)
        parameters.append(clu.Parameter(name = 'checkpoints',
                                        value = checkpoints))
        if checkpoints:
            parameters.append(clu.Parameter(name = 'checkpoint_every',
                                            value = clu.ask_for_input('How many time steps per checkpoint?', default = 50, cast_to = int)))

        outer_radius_default = mesh_kwargs['outer_radius'] / bohr_radius
        parameters.append(clu.Parameter(name = 'mask',
                                        value = ion.RadialCosineMask(inner_radius = bohr_radius * clu.ask_for_input('Mask Inner Radius (in Bohr radii)?', default = outer_radius_default - 50, cast_to = float),
                                                                     outer_radius = bohr_radius * clu.ask_for_input('Mask Outer Radius (in Bohr radii)?', default = outer_radius_default, cast_to = float),
                                                                     smoothness = clu.ask_for_input('Mask Smoothness?', default = 8, cast_to = int))))

        # pulse parameters
        pulse_parameters = []

        pulse_width = clu.Parameter(name = 'pulse_width',
                                    value = asec * np.array(clu.ask_for_eval('Pulse Widths (in as)?', default = 'np.array([50, 100, 200, 300, 400, 500])')),
                                    expandable = True)
        pulse_parameters.append(pulse_width)

        fluence = clu.Parameter(name = 'fluence',
                                value = (J / (cm ** 2)) * np.array(clu.ask_for_eval('Pulse Fluence (in J/cm^2)?', default = 'np.array([.1, 1, 5, 10, 20])')),
                                expandable = True)
        pulse_parameters.append(fluence)

        window_time_in_pw = clu.Parameter(name = 'window_time_in_pw',
                                          value = clu.ask_for_input('Window Time (in pulse widths)?', default = time_bound_in_pw.value - 1, cast_to = float))
        window_width_in_pw = clu.Parameter(name = 'window_width_in_pw',
                                           value = clu.ask_for_input('Window Width (in pulse widths)?', default = 0.5, cast_to = float))
        parameters.append(window_time_in_pw)
        parameters.append(window_width_in_pw)

        sinc_pulses = tuple(ion.SincPulse(**d,
                                          window = ion.SymmetricExponentialTimeWindow(window_time = d['pulse_width'] * window_time_in_pw.value,
                                                                                      window_width = d['pulse_width'] * window_width_in_pw.value))
                            for d in clu.expand_parameters_to_dicts(pulse_parameters))

        number_of_phases = clu.ask_for_input('Number of phases?', default = 32, cast_to = int)


        def amplitude_function(cutoff, amplitude, frequency):
            return np.where(np.abs(frequency) < cutoff, amplitude, 0)


        def phase_function(phase, frequency):
            return np.where(frequency >= 0, phase, -phase)

        parameters.append(clu.Parameter(name = 'electric_potential',
                                        value = tuple(ion.GenericElectricField(ft.partial(amplitude_function, sinc.frequency_cutoff, sinc.amplitude_per_frequency),
                                                                               ft.partial(phase_function, phase),
                                                                               frequency_upper_limit = sinc.frequency_cutoff * 20,
                                                                               frequency_points = 2 ** 15,
                                                                               name = 'pw={}asec_flu={}Jcm2_phase={}pi'.format(uround(sinc.pulse_width, asec),
                                                                                                                               uround(sinc.fluence, Jcm2),
                                                                                                                               uround(phase, pi)),
                                                                               extra_information = {'phase': phase, 'pulse_width': sinc.pulse_width})
                                                      for sinc in sinc_pulses
                                                      for phase in np.linspace(0, twopi, number_of_phases)),
                                        expandable = True))

        parameters.append(clu.Parameter(name = 'store_norm_by_l',
                                        value = clu.ask_for_bool('Store Norm-by-L?')))

        print('Generating parameters...')

        spec_kwargs_list = clu.expand_parameters_to_dicts(parameters)
        specs = []

        for ii, spec_kwargs in enumerate(spec_kwargs_list):
            time_bound = spec_kwargs['time_bound_in_pw'] * spec_kwargs['electric_potential'].pulse_width
            spec = spec_type(spec_kwargs['electric_potential'].name,
                             file_name = str(ii),
                             time_initial = -time_bound, time_final = time_bound,
                             **mesh_kwargs, **spec_kwargs)

            if spec.electric_potential.phase == 'cos':
                spec.electric_potential.dc_correction_time = spec.time_final

            specs.append(spec)

        clu.specification_check(specs)

        submit_string = clu.format_chtc_submit_string(args.job_name, len(specs), checkpoints = checkpoints)
        clu.submit_check(submit_string)

        # point of no return
        shutil.rmtree(job_dir, ignore_errors = True)

        clu.create_job_dirs(job_dir)
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

        clu.submit_job(job_dir)
