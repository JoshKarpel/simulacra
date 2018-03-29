import collections
import datetime
import itertools
import logging
import os
from copy import copy
from typing import Any, Iterable, Optional, Callable, Type, Tuple, Union, List

from tqdm import tqdm

from .. import sims, vis, utils, exceptions
from .. import units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SimulationResult:
    """A class that represents the results of Simulation run on a cluster."""

    def __init__(self, sim, job_processor):
        """
        Initialize a SimulationResult from a Simulation and JobProcessor, picking up information from both.

        Do not store direct references to the Simulation to ensure that the garbage collector can clean it up.

        :param sim: a Simulation
        :param job_processor: a JobProcessor
        """
        self.name = copy(sim.name)
        self.file_name = copy(int(sim.file_name))
        self.plots_dir = job_processor.plots_dir

        self.init_time = copy(sim.init_time)
        self.start_time = copy(sim.start_time)
        self.end_time = copy(sim.end_time)
        self.elapsed_time = copy(sim.elapsed_time.total_seconds())
        self.running_time = copy(sim.running_time.total_seconds())


class JobProcessor(sims.Beet):
    """
    A class that processes a collection of pickled Simulations. Should be subclassed for specialization.

    Attributes
    ----------
    running_time
        The total running time of all the simulations in the job.
    elapsed_time
        The elapsed time of the job (first simulation started to last simulation ended).
    """

    simulation_type = sims.Simulation
    simulation_result_type = SimulationResult

    def __init__(self, job_name: str, job_dir_path: str):
        """
        Parameters
        ----------
        job_name : :class:`str`
            The name of the job.
        job_dir_path : :class:`str`
            The path to the job directory.
        """
        super().__init__(job_name)

        self.job_dir_path = job_dir_path

        for dir in (self.inputs_dir, self.outputs_dir, self.plots_dir, self.movies_dir, self.summaries_dir):
            utils.ensure_dir_exists(dir)

        self.sim_names = self.get_sim_names_from_specs()
        self.sim_count = len(self.sim_names)
        self.unprocessed_sim_names = set(self.sim_names)

        self.data = collections.OrderedDict((sim_name, None) for sim_name in self.sim_names)

    def __str__(self):
        return '{} for job {}, processed {}/{} Simulations'.format(self.__class__.__name__, self.name, self.sim_count - len(self.unprocessed_sim_names), self.sim_count)

    @property
    def inputs_dir(self):
        return os.path.join(self.job_dir_path, 'inputs')

    @property
    def outputs_dir(self):
        return os.path.join(self.job_dir_path, 'outputs')

    @property
    def plots_dir(self):
        return os.path.join(self.job_dir_path, 'plots')

    @property
    def movies_dir(self):
        return os.path.join(self.job_dir_path, 'movies')

    @property
    def summaries_dir(self):
        return os.path.join(self.job_dir_path, 'summaries')

    @property
    def running_time(self):
        return datetime.timedelta(
            seconds = sum(r.running_time for r in self.data.values() if r is not None)
        )

    @property
    def elapsed_time(self):
        earliest = min(r.init_time for r in self.data.values() if r is not None)
        latest = max(r.end_time for r in self.data.values() if r is not None)

        return latest - earliest

    def get_sim_names_from_specs(self):
        """Get a list of Simulation file names based on their Specifications."""
        return sorted([f.strip('.spec') for f in os.listdir(self.inputs_dir)], key = int)

    def get_sim_names_from_sims(self):
        """Get a list of Simulation file names actually found in the output directory."""
        return sorted([f.strip('.sim') for f in os.listdir(self.outputs_dir)], key = int)

    def save(
        self,
        target_dir: Optional[str] = None,
        file_extension: str = '.job',
        **kwargs,
    ):
        return super().save(target_dir = target_dir, file_extension = file_extension, **kwargs)

    def _load_sim(self, sim_file_name: str, **load_kwargs) -> sims.Simulation:
        """
        Load a :class:`Simulation` by its ``file_name``.

        Parameters
        ----------
        sim_file_name : :class:`str`
            The ``file_name`` of the :class:`Simulation` to load.
        load_kwargs
            Keyword arguments are passed to the ``load`` method of the :class:`Simulation``.

        Returns
        -------
        :class:`Simulation`
            The loaded :class:`Simulation`.
        """
        sim_path = os.path.join(self.outputs_dir, f'{sim_file_name}.sim')

        try:
            sim = self.simulation_type.load(os.path.join(sim_path), **load_kwargs)
            logger.debug(f'Loaded {sim_file_name}.sim from job {self.name}')
            if sim.status != sims.Status.FINISHED:
                raise exceptions.UnfinishedSimulation(f'{sim_file_name}.sim from job {self.name} exists but is not finished')
            return sim
        except FileNotFoundError as e:
            raise exceptions.MissingSimulation(f'Failed to find completed {sim_file_name}.sim from job {self.name}')

    def load_sims(self, force_reprocess: bool = False):
        """
        Process the job by loading newly-downloaded Simulations and generating SimulationResults from them.

        Parameters
        ----------
        force_reprocess : :class:`bool`
            If ``True``, process all Simulations in the output directory regardless of prior processing status.
        """
        with utils.BlockTimer() as t:
            logger.info('Loading simulations from job {}'.format(self.name))

            if force_reprocess:
                sim_names = tqdm(copy(self.sim_names), ncols = 80)
            else:
                new_sims = self.unprocessed_sim_names.intersection(self.get_sim_names_from_sims())  # only process newly-downloaded Simulations
                sim_names = tqdm(new_sims)

            for sim_name in sim_names:
                try:
                    sim = self._load_sim(sim_name)
                    self.data[sim_name] = self.simulation_result_type(sim, job_processor = self)
                    self.unprocessed_sim_names.discard(sim_name)

                    self.save(target_dir = self.job_dir_path, ensure_dir_exists = False)
                except exceptions.UnfinishedSimulation as e:
                    logger.debug(e)
                except Exception as e:
                    logger.exception(f'Exception encountered while processing simulation {sim_name}')

        logger.info(f'Finished loading simulations from job {self.name}. Failed to find {len(self.unprocessed_sim_names)} / {self.sim_count} simulations. Elapsed time: {t.wall_time_elapsed}')

    def summarize(self):
        with utils.BlockTimer() as t:
            self.make_time_diagnostics_plot()
            self.write_time_diagnostics_to_file()

            # self.write_to_txt()
            # self.write_to_csv()

            self.make_summary_plots()

        logger.info(f'Finished summaries for job {self.name}. Elapsed time: {t.wall_time_elapsed}')

    def write_to_csv(self):
        raise NotImplementedError

    def write_to_txt(self):
        raise NotImplementedError

    def select_by_kwargs(self, **kwargs) -> Iterable[SimulationResult]:
        """
        Return all of the :class:`SimulationResult` that match the key-value pairs passed as keyword arguments.

        Parameters
        ----------
        kwargs
            Key-value pairs to match against.

        Returns
        -------

        """
        out = []

        for sim_result in (r for r in self.data.values() if r is not None):
            if all(getattr(sim_result, key) == val for key, val in kwargs.items()):
                out.append(sim_result)

        return out

    def select_by_lambda(self, test_function: Callable) -> Iterable[SimulationResult]:
        """
        Return all of the :class:`SimulationResult` for which ``test_function(sim_result)`` evaluates to ``True``.

        Parameters
        ----------
        test_function : callable
            A test function that will be called on simulation results to determine whether they should be in the result set.

        Returns
        -------

        """
        return [
            sim_result
            for sim_result in self.data.values()
            if test_function(sim_result) and sim_result is not None
        ]

    @utils.memoize
    def parameter_set(self, parameter: 'Parameter'):
        """Get the set of values of a parameter from the collected data."""
        return set(getattr(result, parameter) for result in self.data.values())

    def make_summary_plots(self):
        """Hook method for making automatic summary plots from collected data."""
        pass

    def write_time_diagnostics_to_file(self):
        """Write time diagnostic information for the job to a text file in the job directory."""
        path = os.path.join(self.job_dir_path, f'{self.name}_diagnostics.txt')
        with open(path, mode = 'w') as f:
            f.write('\n'.join((
                f'Diagnostic Data for {self.name}:',
                '',
                f'{self.sim_count - len(self.unprocessed_sim_names)} {self.simulation_type.__name__}s',
                f'Simulation Result Type: {self.simulation_result_type.__name__}',
                '',
                f'Elapsed Time: {self.elapsed_time}',
                f'Combined Runtime: {self.running_time}',
                f'Speedup Factor: {u.uround(self.running_time / self.elapsed_time)}',
                '',
                f'Earliest Sim Init: {min(r.init_time for r in self.data.values() if r is not None)}',
                f'Latest Sim Init: {max(r.init_time for r in self.data.values() if r is not None)}',
                f'Earliest Sim Start: {min(r.start_time for r in self.data.values() if r is not None)}',
                f'Latest Sim Start: {max(r.start_time for r in self.data.values() if r is not None)}',
                f'Earliest Sim Finish: {min(r.end_time for r in self.data.values() if r is not None)}',
                f'Latest Sim Finish: {max(r.end_time for r in self.data.values() if r is not None)}',
            )))

        logger.debug(f'Wrote diagnostic information for job {self.name} to {path}')

    def make_time_diagnostics_plot(self):
        """Save a diagnostics plot to the job directory.."""

        sim_numbers = [result.file_name for result in self.data.values() if result is not None]
        running_time = [result.running_time for result in self.data.values() if result is not None]

        vis.xy_plot(
            f'{self.name}__diagnostics',
            sim_numbers,
            running_time,
            line_kwargs = [dict(linestyle = '', marker = '.')],
            y_unit = 'hours',
            x_label = 'Simulation Number', y_label = 'Time',
            title = f'{self.name} Diagnostics',
            target_dir = self.summaries_dir
        )

        logger.debug(f'Generated diagnostics plot for job {self.name}')


def combine_job_processors(*job_processors, job_dir_path = None):
    sim_type = job_processors[0].simulation_type
    jp_type = job_processors[0].__class__
    combined_jp = jp_type(
        name = '-'.join(jp.name for jp in job_processors),
        job_dir_path = job_dir_path,
        simulation_type = sim_type,
    )

    combined_jp.data = collections.OrderedDict((ii, copy(sim_result)) for ii, (sim_name, sim_result) in enumerate(itertools.chain(jp.data for jp in job_processors)))

    return combined_jp
