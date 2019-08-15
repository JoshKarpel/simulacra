from typing import Union

from .info import Info


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

    def __add__(self, other: Union["Summand", "Sum"]):
        return self.summation_class(*self, *other)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def info(self) -> Info:
        return Info(header=self.__class__.__name__)


class Sum(Summand):
    """
    A class that represents a sum of Summands.

    Calls to __call__ are passed to the contained Summands and then added together and returned.
    """

    container_name = "summands"

    def __init__(self, *summands, **kwargs):
        setattr(self, self.container_name, summands)
        super().__init__(**kwargs)

    @property
    def _container(self):
        return getattr(self, self.container_name)

    def __str__(self):
        return "({})".format(" + ".join([str(s) for s in self._container]))

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, ", ".join([repr(p) for p in self._container])
        )

    def __iter__(self):
        yield from self._container

    def __getitem__(self, item):
        return self._container[item]

    def __add__(self, other: Union[Summand, "Sum"]):
        """Return a new Sum, constructed from all of the contents of self and other."""
        return self.__class__(*self, *other)

    def __call__(self, *args, **kwargs):
        return sum(x(*args, **kwargs) for x in self._container)

    def info(self) -> Info:
        info = super().info()

        for x in self._container:
            info.add_info(x.info())

        return info
