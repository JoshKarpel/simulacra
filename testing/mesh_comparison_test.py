import os

import compy as cp
import compy.quantum.hydrogenic as hyd

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger() as logger:
        spec = hyd.CylindricalSliceSpecification('cyl_slice')
        sim = hyd.ElectricFieldSimulation(spec)

        logger.info(spec)
        logger.info(sim)

        sim.run_simulation()

        print(sim.norm_vs_time)
