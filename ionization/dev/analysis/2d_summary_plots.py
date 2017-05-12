import os

import simulacra as si
import simulacra.cluster as clu


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('compy', 'ionization'):
        jp = clu.JobProcessor.load('hyd__sinc__25pw_20ph_20flu__v2.job')

        print(jp)

        jp.plots_dir = OUT_DIR
        jp.summary_dir = OUT_DIR

        with si.utils.BlockTimer() as t:
            jp.make_pulse_parameter_scans_2d()

        print(t)
