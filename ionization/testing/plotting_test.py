import os

import compy.quantum.hydrogenic as hyd

import compy as cp

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger() as logger:
        states = (hyd.BoundState(n, l) for n in range(5) for l in range(n))

        for initial_state in states:
            spec = hyd.CylindricalSliceSpecification('cyl_slice__{}_{}'.format(initial_state.n, initial_state.l),
                                                     initial_state = initial_state)
            sim = hyd.ElectricFieldSimulation(spec)

            sim.mesh.plot_g(save = True, target_dir = OUT_DIR)
            # sim.mesh.plot_g(save = True, target_dir = OUT_DIR, grayscale = True)
