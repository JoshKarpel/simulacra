import os

import compy as cp
import ionization as ion
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger() as logger:
        spec = ion.CylindricalSliceSpecification('test',
                                                 animated = True,
                                                 checkpoints = True,
                                                 extra_time = 1 * asec,
                                                 electric_potential = ion.potentials.Rectangle())
        sim = ion.ElectricFieldSimulation(spec)

        print(spec.info())

        print()

        print(sim.info())

        logger.info('\n' + sim.info())
