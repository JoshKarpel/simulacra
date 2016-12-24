import datetime as dt
import logging
import os
import subprocess

import matplotlib.pyplot as _plt

from . import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CompyException(Exception):  # base exception for all compy-specific exceptions
    pass


class Specification(utils.Beet):
    """
    A class that contains the information necessary to run a simulation.

    It should be subclassed for each type of simulation and all additional information necessary to run that kind of simulation should be added via keyword arguments.
    """

    def __init__(self, name, file_name = None, simulation_type = None, **kwargs):
        """
        Construct a Specification.

        Any number of additional keyword arguments can be passed. They will be stored in an attribute called extra_args.

        :param name: the internal name of the Specification
        :param file_name: the desired external name, used for pickling. Illegal characters are stripped before use.
        :param kwargs: extra arguments, stored as attributes
        """
        super(Specification, self).__init__(name, file_name = file_name)

        self.simulation_type = simulation_type

        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.debug('{} stored additional attribute {} = {}'.format(self.name, k, v))

    def save(self, target_dir = None, file_extension = '.spec'):
        """
        Atomically pickle the Specification to {target_dir}/{self.file_name}.{file_extension}, and gzip it for reduced disk usage.

        :param target_dir: directory to save the Specification to
        :param file_extension: file extension to name the Specification with
        :return: the path to the saved Simulation
        """
        return super(Specification, self).save(target_dir, file_extension)

    def to_simulation(self):
        """Return a Simulation of the type associated with the Specification."""
        return self.simulation_type(self)

    def info(self):
        return ''


class Simulation(utils.Beet):
    """
    A class that represents a simulation.

    It should be subclassed and customized for each variety of simulation.
    Ideally, actual computation should be handed off to another object, while the Simulation itself stores the data produced by that object.
    """

    status = utils.RestrictedValues('status', {'initialized', 'running', 'finished', 'paused'})

    def __init__(self, spec, initial_status = 'initialized'):
        """
        Construct a Simulation from a Specification.

        :param spec: the Specification for the Simulation
        :param initial_status: an initial status for the simulation, defaults to 'initialized'
        """
        self.spec = spec
        self.status = initial_status

        super(Simulation, self).__init__(spec.name, file_name = spec.file_name)  # inherit name and file_name from spec
        self.spec.simulation_type = self.__class__

        # diagnostic data
        self.restarts = 0
        self.start_time = dt.datetime.now()
        self.end_time = None
        self.elapsed_time = None
        self.latest_load_time = dt.datetime.now()
        self.run_time = dt.timedelta()

    def save(self, target_dir = None, file_extension = '.sim'):
        """
        Atomically pickle the Simulation to {target_dir}/{self.file_name}.{file_extension}, and gzip it.

        :param target_dir: directory to save the Simulation to
        :param file_extension: file extension to name the Simulation with
        :return: None
        """
        if self.status != 'finished':
            self.run_time += dt.datetime.now() - self.latest_load_time
            self.latest_load_time = dt.datetime.now()

        return super(Simulation, self).save(target_dir, file_extension)

    @classmethod
    def load(cls, file_path, **kwargs):
        """
        Load a Simulation from file_path.

        :param file_path: the path to try to load a Simulation from
        :return: the loaded Simulation
        """
        sim = super(Simulation, cls).load(file_path)

        sim.latest_load_time = dt.datetime.now()
        if sim.status != 'finished':
            sim.restarts += 1

        return sim

    def __str__(self):
        return '{}: {} ({}) [{}]  |  {}'.format(self.__class__.__name__, self.name, self.file_name, self.uid, self.spec)

    def __repr__(self):
        return '{}(spec = {}, uid = {})'.format(self.__class__.__name__, repr(self.spec), self.uid)

    def run_simulation(self):
        """Hook method for running the simulation."""
        raise NotImplementedError

    def info(self):
        """Return a nicely-formatted string containing information about the Simulation."""
        diag = ['Status: {}'.format(self.status),
                '   Start Time: {}'.format(self.start_time),
                '   Latest Load Time: {}'.format(self.latest_load_time),
                '   End Time: {}'.format(self.end_time),
                '   Elapsed Time: {}'.format(self.elapsed_time),
                '   Run Time: {}'.format(self.run_time)]

        return '\n'.join((str(self), *diag, self.spec.info()))


class Animator:
    """
    A superclass that handles sending frames to ffmpeg to create animations.

    To actually make an animation there are three hook methods that need to be overwritten: _initialize_figure, _update_data, and _redraw_frame.

    ffmpeg must be visible on the system path.
    """

    def __init__(self, postfix = '', target_dir = None,
                 length = 30, fps = 30,
                 colormap = _plt.cm.inferno):
        """
        Construct an Animator instance.

        :param postfix: postfix for the file name of the resulting animation
        :param target_dir: directory to place the output (and work in)
        :param length: the length of the animation
        :param fps: the desired frames-per-seconds for the animation (may not be actual fps if not enough/too many available frames)
        :param colormap: the colormap to use in the animation
        """
        if target_dir is None:
            target_dir = os.getcwd()
        self.target_dir = target_dir

        postfix = utils.strip_illegal_characters(postfix)
        if postfix != '' and not postfix.startswith('_'):
            postfix = '_' + postfix
        self.postfix = postfix

        self.length = length
        self.fps = fps
        self.colormap = colormap

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

        self.ffmpeg = subprocess.Popen(self.cmdstring, stdin = subprocess.PIPE, bufsize = -1)

        logger.info('Initialized {}'.format(self))

    def cleanup(self):
        """
        Cleanup method for the Animator's ffmpeg subprocess.

        Should always be called via a try...finally clause (namely, in the finally) in Simulation.run_simulation.
        """
        self.ffmpeg.communicate()
        logger.info('Cleaned up {}'.format(self))

    def _initialize_figure(self):
        """
        Hook for a method to initialize the Animator's figure.

        Make sure that any plot element that will be mutated during the animation is created using the animation = True keyword argument and has a reference in self.redraw.
        """
        logger.debug('Initialized figure for {}'.format(self))

    def _update_data(self):
        """Hook for a method to update the data for each animated figure element."""
        logger.debug('{} updated data from {} {}'.format(self, self.sim.__class__.__name__, self.sim.name))

    def _redraw_frame(self):
        """Redraw the figure frame."""
        _plt.set_cmap(self.colormap)  # make sure the colormap is correct, in case other figures have been created somewhere

        self.fig.canvas.restore_region(self.background)  # copy the static background back onto the figure

        self._update_data()  # get data from the Simulation and update any plot elements that need to be redrawn

        # draw everything that needs to be redrawn (any plot elements that will be mutated during the animation should be added to self.redraw)
        for rd in self.redraw:
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
        super(Sum, self).__init__(**kwargs)

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
