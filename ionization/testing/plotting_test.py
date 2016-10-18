import os

import compy as cp
import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger() as logger:
        states = (ion.BoundState(n, l) for n in range(5) for l in range(n))

        for initial_state in states:
            spec = ion.CylindricalSliceSpecification('cyl_slice__{}_{}'.format(initial_state.n, initial_state.l),
                                                     initial_state = initial_state)
            sim = ion.ElectricFieldSimulation(spec)

            sim.mesh.plot_g(save = True, target_dir = OUT_DIR)
            # sim.mesh.plot_g(save = True, target_dir = OUT_DIR, grayscale = True)
