class SimulacraException(Exception):
    """Base :class:`Exception` for all Simulacra exceptions."""

    pass


class IllegalSphericalHarmonic(SimulacraException):
    """The angular momentum numbers that were asked for are not valid."""

    pass
