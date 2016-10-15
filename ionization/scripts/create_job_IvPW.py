# create an Ionization vs Pulse Width job

import argparse
import json

import compy.quantum.hydrogenic as hyd
import compy.quantum.hydrogenic.cluster as hyd_cluster

import compy as cp
from compy import cluster
from compy.units import *

if __name__ == '__main__':
    # job type options
    job_processor = hyd_cluster.IonizationJobProcessor

    # get command line arguments
    parser = argparse.ArgumentParser(description = 'Create an Ionization vs Pulse Width job.')
    parser.add_argument('job_name',
                        type = str,
                        help = 'the name of the job')
    parser.add_argument('--overwrite', '-o',
                        action = 'store_true',
                        help = 'force overwrite existing job directory if there is a name collision')

    args = parser.parse_args()

    # get input from the user to define the job
    initial_state_n = cp.utils.ask_for_input('Initial State n', default = 1, cast_to = int)
    initial_state_l = cp.utils.ask_for_input('Initial State l', default = 0, cast_to = int)
    initial_state = hyd.BoundState(initial_state_n, initial_state_l)

    spec_type, mesh_specifier = hyd_cluster.ask_mesh_type()

    pulse_width_first = cp.utils.ask_for_input('First Pulse Width (in as)', default = 1, cast_to = float)
    pulse_width_last = cp.utils.ask_for_input('Last Pulse Width (in as)', default = 200, cast_to = float)
    pulse_width_step = cp.utils.ask_for_input('Pulse Width Step Size (in as)', default = 1, cast_to = float)
    pulse_widths = asec * np.arange(pulse_width_first, pulse_width_last + (0.5 * pulse_width_step), pulse_width_step)

    time_step = asec * cp.utils.ask_for_input('Time Step (in as)', default = .1, cast_to = float)

    minimum_final_time = asec * cp.utils.ask_for_input('Minimum Final Time (in as)', default = 5000, cast_to = float)

    extra_time_step = asec * cp.utils.ask_for_input('Extra Time Step (in as)', default = 1, cast_to = float)

    # pulse_fluence = (m_to_cm ** 2) * ask_for_input('Pulse Fluence (in J/cm^2)', default = 7.5, cast_to = float)

    pulse_frequency_ratio = cp.utils.ask_for_input('Pulse Frequency Ratio', default = 5, cast_to = float)

    number_of_modes = cp.utils.ask_for_input('Pulse Number of Modes', default = 71, cast_to = int)

    largest_check_n = cp.utils.ask_for_input('Largest Bound State n to Overlap With', default = 5, cast_to = int)
    states = hyd.BoundState(largest_check_n).states_below()

    checkpoints = cp.utils.ask_for_input('Checkpoints? ([y]/n)', default = 'y', cast_to = str)
    if checkpoints == 'y':
        checkpoints = True
    if checkpoints == 'n':
        checkpoints = False

    print('Generating parameters...')

    specs = []

    for i, pulse_width in enumerate(pulse_widths):
        # external_potential = EnergyNormalizedWindowedCosinePulse(pulse_width, pulse_fluence, first_frequency_ratio = pulse_frequency_ratio, number_of_modes = number_of_modes)
        external_potential = hyd.UniformLinearlyPolarizedElectricField()

        time_bound = 1.1 * external_potential.window_time
        extra_time = minimum_final_time - time_bound

        spec = spec_type('PW_{}as'.format(np.around(pulse_width / asec, 3)), file_name = str(i),
                         **mesh_specifier,
                         external_potential = external_potential,
                         initial_state = initial_state, states = states,
                         time_initial = -time_bound, time_final = time_bound, time_step = time_step,
                         extra_time = extra_time, extra_time_step = extra_time_step,
                         atomic_number = 1, imaginary_potential_amplitude = 1 * atomic_electric_potential, imaginary_potential_range = 2 * bohr_radius,
                         checkpoints = checkpoints, checkpoint_at = 20)

        specs.append(spec)

        cluster.parameter_check(specs)

    submit_string = cluster.format_chtc_submit_string(args.job_name, len(specs), checkpoints = checkpoints)
    cluster.submit_check(submit_string)

    job_info = {'name': args.job_name,
                'number_of_sims': len(specs),
                'specification': spec.__class__.__name__,
                'external_potential': external_potential.__class__.__name__,
                'job_processor': job_processor}  # set at top of if-name-main
    with open('job.info', mode = 'w') as f:
        json.dump(job_info, f)

    cluster.create_job_dirs(args.job_name)
    cluster.save_parameters(specs)
    cluster.write_parameter_info_to_file(specs)
    cluster.write_submit_file(submit_string)

    cluster.submit_job()
