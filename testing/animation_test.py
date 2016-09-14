import os

import compy as cp
import compy.quantum.hydrogenic as hyd
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger() as logger:
        initial_state = hyd.BoundState(1, 0, 0)

        # e_field = hyd.Rectangle(start_time = 20 * asec, end_time = 180 * asec, amplitude = 1 * atomic_electric_field)
        # e_field = hyd.SineWave(omega = twopi / (50 * asec), amplitude = 1 * atomic_electric_field, window_time = 150 * asec, window_width = 20 * asec)
        e_field = None

        t_init = -200 * asec
        t_final = -t_init

        spec = hyd.CylindricalSliceSpecification('cyl_slice', time_initial = t_init, time_final = t_final,
                                                 z_points = 2 ** 9, rho_points = 2 ** 8,
                                                 z_bound = 30 * bohr_radius, rho_bound = 30 * bohr_radius,
                                                 initial_state = initial_state,
                                                 electric_potential = e_field,
                                                 animated = True, animation_dir = OUT_DIR)
        sim = hyd.ElectricFieldSimulation(spec)

        sim.run_simulation()
