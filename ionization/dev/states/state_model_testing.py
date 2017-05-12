import os

import compy as cp
import numpy as np

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.LogManager('compy', 'ionization') as logger:
        for n in range(1, 4):
            for l in range(n):
                for m in range(-l, l + 1):
                    print(n, l, m)

                    s = ion.HydrogenBoundState(n = n, l = l, m = m)
                    print(str(s))
                    print(repr(s))
                    print()

        s = ion.HydrogenBoundState(1, 0, 0) + ion.HydrogenBoundState(2, 0, 0)
        print(s)
        print()

        print(0.5 * ion.HydrogenBoundState())
        x = ion.HydrogenBoundState() * 0.5
        print(x, type(x))
        print()

        s = 1j * np.sqrt(1 / 3) * ion.HydrogenBoundState(1) + np.sqrt(2 / 3) * ion.HydrogenBoundState(2)
        print(s, type(s))
        s += ion.HydrogenBoundState(3)
        print(s, type(s))

        # s += ion.PotentialEnergy()
        # print(s, type(s))

        # TODO: turn this into unit testing!
