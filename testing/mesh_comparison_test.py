import os

import compy as cp
import compy.quantum.hydrogenic as hyd
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger() as logger:
        spec = hyd.CylindricalSliceSpecification('cyl_slice',
                                                 # rho_points = 4, z_points = 4,
                                                 # rho_bound = 1 * bohr_radius, z_bound = 1 * bohr_radius,
                                                 # time_final = 5 * asec,
                                                 # internal_potential = hyd.NuclearPotential(charge = proton_charge),
                                                 # electric_potential = hyd.Rectangle(start_time = 20 * asec, end_time = 180 * asec, amplitude = 1 * atomic_electric_field)
                                                 )
        sim = hyd.ElectricFieldSimulation(spec)

        logger.info(spec)
        logger.info(sim)

        print(sim.times / asec)

        sim.run_simulation()

        print(sim.mesh.delta_z / bohr_radius)
        print(sim.mesh.delta_rho / bohr_radius)

        print(sim.norm_vs_time)
        print(sim.state_overlaps_vs_time[sim.spec.initial_state])
        print(sim.electric_field_amplitude_vs_time / atomic_electric_field)

        print(electron_mass_reduced, spec.test_mass, proton_mass / spec.test_mass)
