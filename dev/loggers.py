import logging
import os
import sys

import simulacra as si


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

si.utils.ensure_parents_exist(OUT_DIR)

logger = logging.getLogger("simulacra")
logger.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(si.utils.LOG_FORMATTER)
logger.addHandler(stdout_handler)

file_handler = logging.FileHandler(
    os.path.join(OUT_DIR, "{}.log".format(FILE_NAME)), mode="w"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(si.utils.LOG_FORMATTER)
logger.addHandler(file_handler)

if __name__ == "__main__":
    logger.info("hi from script")

    par = si.sims.Specification("test", file_name="test_file_name")
    logger.info(par)
    logger.info(repr(par))

    par.save(target_dir=OUT_DIR)

    par2 = si.sims.Specification.load(os.path.join(OUT_DIR, "test_file_name.par"))

    # si.utils.ask_for_input('who are you?', cast_to = str)

    sim = si.sims.Simulation(par)
    logger.critical(sim)
    logger.critical(repr(sim))

    try:
        raise ValueError
    except ValueError as err:
        logger.exception("ouch")

    print(logger.handlers)

    with si.utils.LogManager(file_logs=True, file_dir=OUT_DIR) as withlogger:
        print(withlogger.handlers)
        print(withlogger.handlers)
        withlogger.info("im a fancy logger. should print only once")

    print(logger.handlers)

    logger.debug("only once as well")
