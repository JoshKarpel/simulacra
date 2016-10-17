import os

import compy.quantum.hydrogenic as hyd

import compy as cp
from compy.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger() as logger:
        t = np.array([0, .25, .5, .75, 1])
        # t = .25
        r = 1 * bohr_radius
        distance_along_polarization = 1 * bohr_radius

        coords = {'t': t, 'r': r, 'distance_along_polarization': distance_along_polarization, 'test_charge': electron_charge}

        coulomb = hyd.potentials.NuclearPotential()
        logger.info(coulomb)
        logger.info(repr(coulomb))
        logger.info(coulomb(**coords) / eV)

        sine = hyd.potentials.SineWave.from_frequency(frequency = 1 * Hz, amplitude = 1 * atomic_electric_field)
        logger.info(sine)
        logger.info(repr(sine))
        logger.info(sine(**coords) / eV)

        combined = hyd.potentials.PotentialSum(coulomb, sine)
        logger.info(combined)
        logger.info(repr(combined))
        logger.info(combined(**coords) / eV)

        coords = {'t': t, 'r': r, 'distance_along_polarization': distance_along_polarization, 'test_charge': electron_charge}
        logger.info(combined(**coords) / eV)

        comb_by_add = coulomb + sine
        print(comb_by_add)
        print(repr(comb_by_add))

        comb_again = comb_by_add + sine
        print(comb_again)
        print(repr(comb_again))
