class SimulacraException(Exception):
    """Base :class:`Exception` for all Simulacra exceptions."""

    pass


class UnfinishedSimulation(SimulacraException):
    pass


class MissingSimulation(SimulacraException):
    pass


class IllegalSphericalHarmonic(SimulacraException):
    pass
