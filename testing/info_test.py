import os

import compy as cp
import compy.quantum.hydrogenic as hyd
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger() as logger:
        spec = hyd.CylindricalSliceSpecification('test',
                                                 animated = True,
                                                 checkpoints = True,
                                                 extra_time = 1 * asec,
                                                 electric_potential = hyd.Rectangle())
        sim = hyd.ElectricFieldSimulation(spec)

        print(spec.info())

        print()

        print(sim.info())

        logger.info('\n' + sim.info())
