import os
import logging
from typing import Optional, Callable, Iterable

from .. import sims

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def find_or_init_sim_from_spec(spec, search_dir: Optional[str] = None, file_extension = '.sim'):
    """
    Try to load a :class:`simulacra.Simulation` by looking for a pickled :class:`simulacra.core.Simulation` named ``{search_dir}/{spec.file_name}.{file_extension}``.
    If that fails, create a new Simulation from `spec`.

    Parameters
    ----------
    spec
        The filename of this specification is what will be searched for.
    search_dir
        The directory to look for the simulation in.
    file_extension
        The simulation file extension.

    Returns
    -------
    sim
        The simulation, either loaded or initialized.
    """
    search_dir = search_dir or os.getcwd()
    try:
        path = os.path.join(search_dir, spec.file_name + file_extension)
        sim = sims.Simulation.load(file_path = path)
    except FileNotFoundError:
        sim = spec.to_sim()

    return sim


def run_from_simlib(spec, simlib = None, **kwargs):
    sim = find_or_init_sim_from_spec(spec, search_dir = simlib)

    if sim.status != sims.Status.FINISHED:
        sim.run(**kwargs)
        sim.save(target_dir = simlib)

    return sim
