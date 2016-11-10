import os

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization') as logger:
        # spec = ion.SphericalHarmonicSpecification('spherical', mesh_type = ion.SphericalHarmonicMesh, dipole_gauges = [])
        # sim = ion.ElectricFieldSimulation(spec)
        #
        # print(sim.info())
        #
        # print(sim.mesh.get_kinetic_energy_matrix_operators())
        #
        # sim.run_simulation()
        #
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        spec = ion.SphericalHarmonicSpecification('lagrangian', mesh_type = ion.LagrangianSphericalHarmonicMesh, dipole_gauges = [],
                                                  # r_points = 3, spherical_harmonics_max_l = 2, test_states = [ion.BoundState(1)]
                                                  )
        sim = ion.ElectricFieldSimulation(spec)

        print(sim.info())

        sim.run_simulation()

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        print(sim.mesh.get_kinetic_energy_matrix_operators()[0].toarray())
