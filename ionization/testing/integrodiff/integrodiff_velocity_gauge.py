import logging
import os

from tqdm import tqdm

import numpy as np
import scipy.integrate as integrate

import compy as cp
import ionization as ion
from ionization import integrodiff as ide
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        pw = 100 * asec
        t_bound = 10

        L = 1 * bohr_radius
        m = electron_mass
        q = electron_charge

        times = np.linspace(-pw * t_bound, pw * t_bound, 1e3)
        dt = np.abs(times[1] - times[0])
        t_total = times[-1] - times[0]

        times_to_indices = {time: ii for ii, time in enumerate(times)}

        sinc = ion.SincPulse(pulse_width = pw)
        efield = ion.GaussianPulse(pulse_width = pw, fluence = 1 * Jcm2, omega_carrier = sinc.omega_carrier)
        # efield = ion.SincPulse(pulse_width = pw, fluence = 1 * Jcm2)
        electric_field_vs_time = efield.get_electric_field_amplitude(times)
        avg_field = integrate.simps(electric_field_vs_time, x = times) / t_total

        efield = efield + ion.Rectangle(start_time = times[0], end_time = times[-1], amplitude = -avg_field)
        electric_field_vs_time = efield.get_electric_field_amplitude(times)

        vector_potential_vs_time = dt * np.cumsum(electric_field_vs_time)

        print(electric_field_vs_time)
        print(vector_potential_vs_time)

        alpha_from_start = dt * np.cumsum(vector_potential_vs_time) * q / m

        print(alpha_from_start)

        plt_kwargs = dict(
            target_dir = OUT_DIR,
        )

        cp.utils.xy_plot('electric_field_vs_time',
                         times, electric_field_vs_time,
                         x_label = r'Time $t$', x_scale = 'asec',
                         y_label = r'$\mathcal{E}(t)$', y_scale = 'AEF',
                         **plt_kwargs)

        cp.utils.xy_plot('vector_potential_vs_time',
                         times, vector_potential_vs_time * q,
                         x_label = r'Time $t$', x_scale = 'asec',
                         y_label = r'$ e \, \mathcal{A}(t)$', y_scale = 'atomic_momentum',
                         **plt_kwargs)

        cp.utils.xy_plot('alpha_vs_time',
                         times, alpha_from_start,
                         x_label = r'Time $t$', x_scale = 'asec',
                         y_label = r'$ \alpha(t, t_0) $', y_scale = 'bohr_radius',
                         **plt_kwargs)


        @cp.utils.memoize
        def a_kern(time_diff):
            return (2 * (L ** 2)) + 1j * hbar * time_diff / (2 * m)


        @cp.utils.memoize
        def b_kern(time_end, time_start):
            return alpha_from_start[times_to_indices[time_end]] - alpha_from_start[times_to_indices[time_start]]


        def kernel(t_end):
            def k(t):
                pre = np.sqrt(pi) / 4
                exp = np.exp(-(b_kern(t_end, t) ** 2) / (4 * a_kern(t_end - t)))
                diff = 2 * a_kern(t_end - t) - (b_kern(t_end, t) ** 2)
                div = a_kern(t_end - t) ** (5 / 2)

                return pre * exp * diff / div

            return k


        # for ref_time in times[::int(len(times) / 20)]:
        #     kernel_from_ref_time = kernel(ref_time)
        #     eval_kernel_from_ref_time = [kernel_from_ref_time(t) for t in times]
        #
        #     cp.utils.xy_plot(f'kernel_from_t={uround(ref_time, asec, 0)}asec_vs_time',
        #                      times, np.real(eval_kernel_from_ref_time), np.imag(eval_kernel_from_ref_time), np.abs(eval_kernel_from_ref_time),
        #                      line_labels = ('Real', 'Imag', 'Abs'),
        #                      x_label = r'Time $t$', x_scale = 'asec',
        #                      y_label = r"$K(t, t')$",
        #                      **plt_kwargs)

        a = np.zeros(len(times), dtype = np.complex128)
        a[0] = 1

        # FORWARD EULER

        prefactor = -np.sqrt(2 / pi) * L * (electron_charge / electron_mass) ** 2

        for t in tqdm(times[1:]):
            current_index = times_to_indices[t]
            a_dot = prefactor * vector_potential_vs_time[current_index - 1]
            # print(a_dot)

            # print()
            # print('t curr', current_index, t)
            # print('vp curr', vector_potential_vs_time[current_index - 1])
            # print('vp slice', vector_potential_vs_time[:current_index])
            # print('a slice', a[:current_index])
            # print('t slice', times[:current_index])
            kern = kernel(t)
            # print('k slice', np.array([kern(tt) for tt in times[:current_index]]))

            # print('prod', vector_potential_vs_time[:current_index] * a[:current_index] * np.array([kern(tt) for tt in times[:current_index]]))

            a_dot *= integrate.simps(y = vector_potential_vs_time[:current_index] * a[:current_index] * np.array([kern(tt) for tt in times[:current_index]]),
                                     x = times[:current_index])

            # print('adot', a_dot)
            # print(np.abs(a[current_index - 1]) ** 2, a[current_index - 1], dt * a_dot)

            a[current_index] = a[current_index - 1] + (dt * a_dot)

        ## COMPARE TO LENGTH GAUGE
        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)
        sim = ide.AdaptiveIntegroDifferentialEquationSpecification('compare',
                                                                   electric_potential = efield,
                                                                   time_initial = times[0], time_final = times[-1],
                                                                   maximum_time_step = 1 * asec,
                                                                   prefactor = prefactor,
                                                                   kernel = ide.gaussian_kernel, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                   electric_potential_dc_correction = False, ).to_simulation()
        sim.run_simulation()
        sim.plot_a_vs_time(**plt_kwargs)

        print(sim.info())

        cp.utils.xy_plot('a_vs_time',
                         times, np.abs(a) ** 2,
                         x_label = r'Time $t$', x_scale = 'asec',
                         y_label = r'$\left| a \right|^2$', y_lower_limit = 0, y_upper_limit = 1,
                         **plt_kwargs)
