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

    with cp.utils.Logger('compy', 'ionization', stdout_level = 31 - ((args.verbosity + 1) * 10)) as logger:
        # job type options
        job_processor = iclu.ConvergenceJobProcessor

        job_dir = os.path.join(args.dir, args.job_name)

        if os.path.exists(job_dir):
            if not args.overwrite and not clu.ask_for_bool('A job with that name already exists. Overwrite?', default = 'No'):
                clu.abort_job_creation()
            else:
                shutil.rmtree(job_dir)

        parameters = []

        # get input from the user to define the job
        spec_type = ion.SphericalHarmonicSpecification

        t_bound = 250 * asec

        r_bound = clu.Parameter('r_bound',
                                value = bohr_radius * cp.cluster.ask_for_input('R Bound (Bohr radii)?', default = 50, cast_to = float))
        parameters.append(r_bound)

        parameters.append(clu.Parameter('delta_r',
                                        value = bohr_radius * clu.ask_for_eval('Radial Mesh Spacings (in Bohr radii)?', default = 'np.logspace(0, -2, 50)'),
                                        expandable = True))

        parameters.append(clu.Parameter('time_step',
                                        value = asec * clu.ask_for_eval('Time Steps (in asec)?', default = 'np.logspace(0, -2, 50)'),
                                        expandable = True))

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

        outer_radius_default = uround(r_bound.value, bohr_radius, 2)
        parameters.append(clu.Parameter(name = 'mask',
                                        value = ion.RadialCosineMask(inner_radius = bohr_radius * clu.ask_for_input('Mask Inner Radius (in Bohr radii)?', default = outer_radius_default * .8, cast_to = float),
                                                                     outer_radius = bohr_radius * clu.ask_for_input('Mask Outer Radius (in Bohr radii)?', default = outer_radius_default, cast_to = float),
                                                                     smoothness = clu.ask_for_input('Mask Smoothness?', default = 8, cast_to = int))))

        parameters.append(clu.Parameter(name = 'electric_potential',
                                        value = ion.Rectangle(start_time = -t_bound * .9,
                                                              end_time = t_bound * .9,
                                                              amplitude = atomic_electric_field * clu.ask_for_input('Electric Field Amplitude (in AEF)?', default = .1, cast_to = float),
                                                              window = ion.SymmetricExponentialTimeWindow(window_time = 100 * asec, window_width = 5 * asec)),
                                        expandable = True))

        checkpoints = clu.ask_for_bool('Checkpoints?', default = True)
        parameters.append(clu.Parameter(name = 'checkpoints',
                                        value = checkpoints))
        if checkpoints:
            parameters.append(clu.Parameter(name = 'checkpoint_every',
                                            value = clu.ask_for_input('How many time steps per checkpoint?', default = 50, cast_to = int)))

        parameters.append(clu.Parameter(name = 'store_data_every',
                                        value = clu.ask_for_input('Store Data Every?', default = 1, cast_to = int)))

        print('Generating parameters...')

        spec_kwargs_list = clu.expand_parameters_to_dicts(parameters)
        specs = []

        print('Generating specifications...')

        for ii, spec_kwargs in enumerate(spec_kwargs_list):
            delta_r = spec_kwargs['delta_r']
            r_points = int(r_bound.value / delta_r)

            time_step = spec_kwargs['time_step']
            electric_potential_amp = spec_kwargs['electric_potential'].amplitude

            name = f'amp={uround(electric_potential_amp, atomic_electric_field, 3)}_R={uround(r_bound.value, bohr_radius, 3)}br_RP={r_points}_dt={uround(time_step, asec, 5)}asec'

            spec = spec_type(name,
                             file_name = str(ii),
                             l_bound = 50,
                             time_initial = -t_bound, time_final = t_bound,
                             electric_field_dc_correction = False,
                             r_points = r_points,
                             **spec_kwargs)

            spec.pulse_amplitude = electric_potential_amp

            specs.append(spec)

        clu.specification_check(specs)

        submit_string = clu.format_chtc_submit_string(args.job_name, len(specs), checkpoints = checkpoints)
        clu.submit_check(submit_string)

        # point of no return
        shutil.rmtree(job_dir, ignore_errors = True)

        clu.create_job_subdirs(job_dir)
        clu.save_specifications(specs, job_dir)
        clu.write_specifications_info_to_file(specs, job_dir)
        clu.write_parameters_info_to_file(parameters, job_dir)

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
