import os

import compy as cp
import compy.cluster as clu
from compy.units import *
import ionization as ion
import ionization.cluster as iclu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with cp.utils.Logger('compy', 'ionization'):
        jp = clu.JobProcessor.load('test_ide.job')

        print(jp)

        jp.plots_dir = OUT_DIR

        jp.make_pulse_parameter_scans_2d()
