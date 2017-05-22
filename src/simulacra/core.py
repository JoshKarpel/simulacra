import datetime
import gzip
import itertools
import pickle
import uuid
from copy import deepcopy

import logging
import os
import subprocess

import matplotlib.pyplot as _plt

from . import utils


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SimulacraException(Exception):
    """Base exception for all Simulacra exceptions."""
    pass


class Beet:
    """
    A class that provides an easy interface for pickling and unpickling instances.

    Two Beets compare and hash equal if they have the same uid attribute, a uuid4 generated during initialization.
    """

    def __init__(self, name, file_name = None):
        """
        Construct a Beet with the given name and file_name.

        The file_name is automatically derived from the name if None is given.
        
        Parameters
        ----------
        name : :class:`str`
            The internal name of the Beet.
        file_name : :class:`str`
            The desired external name of the Beet. Illegal characters are stripped before use.
        """
        self.name = str(name)
        if file_name is None:
            file_name = self.name

        file_name_stripped = utils.strip_illegal_characters(str(file_name))
        if file_name_stripped != file_name:
            logger.warning('Using file name {} instead of {} for {}'.format(file_name_stripped, file_name, self.name))
        self.file_name = file_name_stripped

        self.initialized_at = datetime.datetime.utcnow()
        self.uid = uuid.uuid4()

        logger.info('Initialized {}'.format(repr(self)))

    def __str__(self):
        return '{}: {} ({}) [{}]'.format(self.__class__.__name__, self.name, self.file_name, self.uid)

    def __repr__(self):
        return utils.field_str(self, 'name', 'file_name', 'uid')

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def copy(self):
        """Return a deepcopy of the Beet."""
        return deepcopy(self)

    def save(self, target_dir = None, file_extension = '.beet', compressed = True):
        """
        Atomically pickle the Beet to a file.
        
        Parameters
        ----------
        target_dir : :class:`str`
            The directory to save the Beet to.
        file_extension : :class:`str`
            The file extension to name the Beet with (for keeping track of things, no actual effect).
        compressed : :class:`bool`
            Whether to compress the Beet using gzip.

        Returns
        -------
        :class:`str`
            The path to the saved Beet.
        """
        if target_dir is None:
            target_dir = os.getcwd()

        file_path = os.path.join(target_dir, self.file_name + file_extension)
        file_path_working = file_path + '.working'

        utils.ensure_dir_exists(file_path_working)

        if compressed:
            op = gzip.open
        else:
            op = open

        with op(file_path_working, mode = 'wb') as file:
            pickle.dump(self, file, protocol = -1)

        os.replace(file_path_working, file_path)

        logger.debug('Saved {} {} to {}'.format(self.__class__.__name__, self.name, file_path))

        return file_path

    @classmethod
    def load(cls, file_path):
        """
        Load a Beet from `file_path`.
        
        Parameters
        ----------
        file_path
            The path to load a Beet from.

        Returns
        -------
        :class:`Beet`
            The loaded Beet.
        """
        try:
            with gzip.open(file_path, mode = 'rb') as file:
                beet = pickle.load(file)
        except OSError:
            with open(file_path, mode = 'rb') as file:
                beet = pickle.load(file)

        logger.debug('Loaded {} {} from {}'.format(beet.__class__.__name__, beet.name, file_path))

        return beet

    def info(self):
        return str(self)


class Specification(Beet):
    """
    A class that contains the information necessary to run a simulation.

    It should be subclassed for each type of simulation and all additional information necessary to run that kind of simulation should be added via keyword arguments.
    """

    simulation_type = None

    def __init__(self, name, file_name = None, **kwargs):
        """
        Construct a Specification.
        
        Any number of additional keyword arguments can be passed. They will be stored as attributes if they don't conflict with any attributes already set.
        
        Parameters
        ----------
        name : :class:`str`
            The internal name of the Specification.
        file_name : :class:`str`
        kwargs
            Any number of keyword arguments, which will be stored as attributes.
        """
        super().__init__(name, file_name = file_name)

        for k, v in ((k, v) for k, v in kwargs.items() if k not in self.__dict__):
            setattr(self, k, v)
            logger.debug('{} stored additional attribute {} = {}'.format(self.name, k, v))

    def save(self, target_dir = None, file_extension = '.spec', compressed = True):
        """
        Atomically pickle the Specification to a file.
        
        Parameters
        ----------
        target_dir : :class:`str`
            The directory to save the Specification to.
        file_extension : :class:`str`
            The file extension to name the Specification with (for keeping track of things, no actual effect).
        compressed : :class:`bool`
            Whether to compress the Beet using gzip.

        Returns
        -------
        :class:`str`
            The path to the saved Specification.
        """
        return super().save(target_dir = target_dir, file_extension = file_extension, compressed = compressed)

    def to_simulation(self):
        """Return a Simulation of the type associated with the Specification, generated from this instance."""
        try:
            return self.simulation_type(self)
        except TypeError:
            return Simulation(self)

    def info(self):
        """Return a string describing the parameters of the Specification."""
        return ''

    def clone(self, **kwargs):
        """Return a clone of the Specification, with modifications defined by the kwargs."""
        new_spec = self.copy()

        for k, v in kwargs.items():
            setattr(new_spec, k, v)

        return new_spec


# Simulation status names
STATUS_INI = 'initialized'
STATUS_RUN = 'running'
STATUS_FIN = 'finished'
STATUS_PAU = 'paused'
STATUS_ERR = 'error'


class Simulation(Beet):
    """
    A class that represents a simulation.

    It should be subclassed and customized for each variety of simulation.
    
    Attributes
    ----------
    status : :class:`str`
        The status of the Simulation. One of ``'initialized'``, ``'running'``, ``'finished'``, ``'paused'``, or ``'error'``.
    """

    _status = utils.RestrictedValues('status', {'', STATUS_INI, STATUS_RUN, STATUS_FIN, STATUS_PAU, STATUS_ERR})

    def __init__(self, spec):
        """
        Construct a Simulation from a Specification.
        
        Simulations should generally be instantiated using Specification.to_simulation() to avoid possible mismatches.
        
        Parameters
        ----------
        spec : :class:`Specification`
            The :class:`Specification` for the Simulation.
        """
        self.spec = spec

        super().__init__(spec.name, file_name = spec.file_name)  # inherit name and file_name from spec

        # diagnostic data
        self.runs = 0
        self.init_time = None
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.latest_run_time = None
        self.running_time = datetime.timedelta()

        self._status = ''
        self.status = STATUS_INI

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, s):
        if s == self.status:
            raise ValueError('Tried to set status of {} to its current status'.format(self.name))

        now = datetime.datetime.utcnow()

        if s == STATUS_INI:
            self.init_time = now
        elif s == STATUS_RUN:
            if self.latest_run_time is None:
                self.start_time = now
            self.latest_run_time = now
            self.runs += 1
        elif s == STATUS_PAU:
            if self.latest_run_time is not None:
                self.running_time += now - self.latest_run_time
        elif s == STATUS_FIN:
            if self.latest_run_time is not None:
                self.running_time += now - self.latest_run_time
            self.end_time = now
            self.elapsed_time = self.end_time - self.init_time

        self._status = s

        logger.debug("{} {} ({}) status set to {}".format(self.__class__.__name__, self.name, self.file_name, s))

    def save(self, target_dir = None, file_extension = '.sim', compressed = True):
        """
        Atomically pickle the Simulation to a file.
        
        Parameters
        ----------
        target_dir : :class:`str`
            The directory to save the Simulation to.
        file_extension : :class:`str`
            The file extension to name the Simulation with (for keeping track of things, no actual effect).
        compressed : :class:`bool`
            Whether to compress the Beet using gzip.

        Returns
        -------
        :class:`str`
            The path to the saved Simulation.
        """
        if self.status != STATUS_FIN:
            self.status = STATUS_PAU

        return super().save(target_dir = target_dir, file_extension = file_extension, compressed = compressed)

    def __str__(self):
        return '{}: {} ({}) [{}]  |  {}'.format(self.__class__.__name__, self.name, self.file_name, self.uid, self.spec)

    def __repr__(self):
        return '{}(spec = {}, uid = {})'.format(self.__class__.__name__, repr(self.spec), self.uid)

    def run_simulation(self):
        """Hook method for running the Simulation, whatever that may entail."""
        raise NotImplementedError

    def info(self):
        """Return a string describing the parameters of the Simulation and its associated Specification."""
        diag = ['Status: {}'.format(self.status),
                '   Start Time: {}'.format(self.init_time),
                '   Latest Load Time: {}'.format(self.latest_run_time),
                '   End Time: {}'.format(self.end_time),
                '   Elapsed Time: {}'.format(self.elapsed_time),
                '   Run Time: {}'.format(self.running_time)]

        return '\n'.join((str(self), *diag, self.spec.info()))


class AxisManager:
    """
    A superclass that manages a matplotlib axis for an Animator.
    """

    def __init__(self):
        self.redraw = []

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def initialize(self, simulation):
        """Hook method for initializing the AxisManager."""
        self.sim = simulation
        self.spec = simulation.spec

        self.initialize_axis()

        logger.debug(f'Initialized {self}')

    def assign_axis(self, axis):
        self.axis = axis

        logger.debug(f'Assigned {self} to {axis}')

    def initialize_axis(self):
        logger.debug(f'Initialized axis for {self}')

    def update_axis(self):
        """Hook method for updating the AxisManager's internal state."""
        logger.debug(f'Updated axis for {self}')


class Animator:
    """
    A superclass that handles sending frames to ffmpeg to create animations.

    To actually make an animation there are three hook methods that need to be overwritten: _initialize_figure, _update_data, and _redraw_frame.
    
    An Animator will generally contain a single matplotlib figure with some animation code of its own in addition to a list of :class:`AxisManagers ~<AxisManager>` that handle axes on the figure.

    For this class to function correctly :code:`ffmpeg` must be visible on the system path.
    """

    def __init__(self, postfix = '', target_dir = None,
                 length = 60, fps = 30,
                 colormap = _plt.cm.get_cmap('inferno')):
        """
        
        Parameters
        ----------
        postfix : :class:`str`
            Postfix for the file name of the resulting animation.
        target_dir : :class:`str`
            Directory to place the animation (and work in).
        length : :class:`float`
            The length of the animation.
        fps : :class:`float`
            The FPS of the animation.
        colormap
            The colormap to use for the animation.
        """
        if target_dir is None:
            target_dir = os.getcwd()
        self.target_dir = target_dir

        postfix = utils.strip_illegal_characters(postfix)
        if postfix != '' and not postfix.startswith('_'):
            postfix = '_' + postfix
        self.postfix = postfix

        self.length = int(length)
        self.fps = fps
        self.colormap = colormap

        self.axis_managers = []
        self.redraw = []

        self.sim = None
        self.spec = None
        self.fig = None

    def __str__(self):
        return '{}(postfix = "{}")'.format(self.__class__.__name__, self.postfix)

    def __repr__(self):
        return '{}(postfix = {})'.format(self.__class__.__name__, self.postfix)

    def initialize(self, simulation):
        """
        Initialize the Animation by setting the Simulation and Specification, determining the target path for output, determining fps and decimation, and setting up the ffmpeg subprocess.

        _initialize_figure() is called during the execution of this method. It should assign a matplotlib figure object to self.fig.

        The simulation should have an attribute available_animation_frames that returns an int describing how many raw frames might be available for use by the animation.
        
        :param simulation: a Simulation for the AxisManager to collect data from
        """
        self.sim = simulation
        self.spec = simulation.spec

        self.file_name = '{}{}.mp4'.format(self.sim.file_name, self.postfix)
        self.file_path = os.path.join(self.target_dir, self.file_name)
        utils.ensure_dir_exists(self.file_path)
        try:
            os.remove(self.file_path)  # ffmpeg complains if you try to overwrite an existing file, so remove it first
        except FileNotFoundError:
            pass

        ideal_frame_count = self.length * self.fps
        self.decimation = int(self.sim.available_animation_frames / ideal_frame_count)  # determine ideal decimation from number of available frames in the simulation
        if self.decimation < 1:
            self.decimation = 1  # if there aren't enough frames available
        self.fps = (self.sim.available_animation_frames / self.decimation) / self.length

        self._initialize_figure()  # call figure initialization hook

        # AXES MUST BE ASSIGNED DURING FIGURE INITIALIZATION

        for ax in self.axis_managers:
            ax.initialize(simulation)

        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        canvas_width, canvas_height = self.fig.canvas.get_width_height()
        self.cmdstring = ("ffmpeg",
                          '-y',
                          '-r', '{}'.format(self.fps),  # choose fps
                          '-s', '%dx%d' % (canvas_width, canvas_height),  # size of image string
                          '-pix_fmt', 'argb',  # pixel format
                          '-f', 'rawvideo', '-i', '-',  # tell ffmpeg to expect raw video from the pipe
                          '-vcodec', 'mpeg4',  # output encoding
                          '-q:v', '1',  # maximum quality
                          self.file_path)

        self.ffmpeg = subprocess.Popen(self.cmdstring,
                                       stdin = subprocess.PIPE, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL,
                                       bufsize = -1)

        logger.info('Initialized {}'.format(self))

    def cleanup(self):
        """
        Cleanup method for the Animator's ffmpeg subprocess.

        Should always be called via a try...finally clause (namely, in the finally) in Simulation.run_simulation.
        """
        self.ffmpeg.kill()
        self.ffmpeg.wait()
        logger.info('Cleaned up {}'.format(self))

    def _initialize_figure(self):
        """
        Hook for a method to initialize the Animator's figure.

        Make sure that any plot element that will be mutated during the animation is created using the animation = True keyword argument and has a reference in self.redraw.
        """
        logger.debug('Initialized figure for {}'.format(self))

    def _update_data(self):
        """Hook for a method to update the data for each animated figure element."""
        for ax in self.axis_managers:
            ax.update_axis()

        logger.debug('{} updated data from {} {}'.format(self, self.sim.__class__.__name__, self.sim.name))

    def _redraw_frame(self):
        """Redraw the figure frame."""
        _plt.set_cmap(self.colormap)  # make sure the colormap is correct, in case other figures have been created somewhere

        self.fig.canvas.restore_region(self.background)  # copy the static background back onto the figure

        self._update_data()  # get data from the Simulation and update any plot elements that need to be redrawn

        # draw everything that needs to be redrawn (any plot elements that will be mutated during the animation should be added to self.redraw)
        for rd in itertools.chain(self.redraw, *(ax.redraw for ax in self.axis_managers)):
            self.fig.draw_artist(rd)

        self.fig.canvas.blit(self.fig.bbox)  # blit the canvas, finalizing all of the draw_artists

        logger.debug('Redrew frame for {}'.format(self))

    def send_frame_to_ffmpeg(self):
        """Redraw anything that needs to be redrawn, then write the figure to an RGB string and send it to ffmpeg."""
        self._redraw_frame()

        self.ffmpeg.stdin.write(self.fig.canvas.tostring_argb())

        logger.debug('{} sent frame to ffpmeg from {} {}'.format(self, self.sim.__class__.__name__, self.sim.name))


class Summand:
    """
    An object that can be added to other objects that it shares a superclass with.
    """

    def __init__(self, *args, **kwargs):
        self.summation_class = Sum

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def __iter__(self):
        """When unpacked, yield self, to ensure compatability with Sum's __add__ method."""
        yield self

    def __add__(self, other):
        return self.summation_class(*self, *other)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Sum(Summand):
    """
    A class that represents a sum of Summands.

    Calls to __call__ are passed to the contained Summands and then added together and returned.
    """

    container_name = 'summands'

    def __init__(self, *summands, **kwargs):
        setattr(self, self.container_name, summands)
        super().__init__(**kwargs)

    @property
    def _container(self):
        return getattr(self, self.container_name)

    def __str__(self):
        return '({})'.format(' + '.join([str(s) for s in self._container]))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join([repr(p) for p in self._container]))

    def __iter__(self):
        yield from self._container

    def __add__(self, other):
        """Return a new Sum, constructed from all of the contents of self and other."""
        return self.__class__(*self, *other)  # TODO: no protection against adding together non-similar types

    def __call__(self, *args, **kwargs):
        return sum(x(*args, **kwargs) for x in self._container)
