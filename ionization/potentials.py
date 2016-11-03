import logging

import numpy as np

from compy.units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Potential:
    """
    A class that represents a potential, defined over some kwargs.

    The result of a call to a Potential should be the potential energy / charge (V = J/C for electric interactions, V = J/kg for gravitational interactions, etc.) at the coordinates given by kwargs.

    Caution: use numpy meshgrids to vectorize over multiple kwargs.
    """

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.__class__.__name__

    def __iter__(self):
        yield self

    def __add__(self, other):
        return PotentialSum(self, other)


class PotentialSum:
    """
    A class that handles a group of potentials that should be evaluated together to produce a total potential.

    Caution: the potentials are summed together with no check as to the structure of the sum. Use numpy meshgrids to vectorize over multiple kwargs.
    """

    def __init__(self, *potentials):
        self.potentials = potentials

    def __str__(self):
        return 'Potentials: {}'.format(', '.join([str(p) for p in self.potentials]))

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join([repr(p) for p in self.potentials]))

    def __iter__(self):
        yield from self.potentials

    def __call__(self, **kwargs):
        return sum(p(**kwargs) for p in self.potentials)

    def __add__(self, other):
        try:
            result = PotentialSum(*self.potentials, other.potentials)
        except AttributeError:
            result = PotentialSum(*self.potentials, other)
        return result


class NuclearPotential(Potential):
    """A Potential representing the electric potential of the nucleus of a hydrogenic atom."""

    def __init__(self, charge = 1 * proton_charge):
        super(NuclearPotential, self).__init__()

        self.charge = charge

    def __repr__(self):
        return '{}(charge = {})'.format(self.__class__.__name__, self.charge)

    def __str__(self):
        return '{}(charge = {} e)'.format(self.__class__.__name__, uround(self.charge, proton_charge, 3))

    def __call__(self, *, r, test_charge, **kwargs):
        return coulomb_force_constant * self.charge * test_charge / r


class RadialImaginaryPotential(Potential):
    def __init__(self, center = 20 * bohr_radius, width = 2 * bohr_radius, amplitude = 1 * atomic_electric_potential):
        """
        Construct a RadialImaginaryPotential. The potential is shaped like a Gaussian and has an imaginary amplitude.

        A positive/negative amplitude yields an imaginary potential that causes decay/amplification.

        :param center: the radial coordinate to center the potential on
        :param width: the width (FWHM) of the Gaussian
        :param amplitude: the peak amplitude of the Gaussian
        """
        self.center = center
        self.width = width
        self.amplitude = amplitude

        self.prefactor = -1j * self.amplitude * (proton_charge ** 2)

    def __repr__(self):
        return '{}(center = {}, width = {}, amplitude = {})'.format(self.__class__.__name__, self.center, self.width, self.amplitude)

    def __str__(self):
        return '{}(center = {} Bohr radii, width = {} Bohr radii, amplitude = {} AEP)'.format(self.__class__.__name__,
                                                                                              uround(self.center, bohr_radius, 3),
                                                                                              uround(self.width, bohr_radius, 3),
                                                                                              uround(self.amplitude, atomic_electric_potential, 3))

    def __call__(self, *, r, **kwargs):
        return self.prefactor * np.exp(-(((r - self.center) / self.width) ** 2))


class ElectricFieldWindow:
    def __init__(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, t):
        raise NotImplementedError


class NoWindow(ElectricFieldWindow):
    def __call__(self, t):
        return 1


class LinearRampWindow(ElectricFieldWindow):
    def __init__(self, ramp_on_time = 0 * asec, ramp_time = 50 * asec):
        self.ramp_on_time = ramp_on_time
        self.ramp_time = ramp_time

        super(LinearRampWindow, self).__init__()

    def __str__(self):
        return '{}(ramp on at = {} as, ramp time = {} as)'.format(self.__class__.__name__,
                                                                  uround(self.ramp_on_time, asec, 3),
                                                                  uround(self.ramp_time, asec, 3))

    def __call__(self, t):
        cond = np.greater_equal(t, self.ramp_on_time)
        on = 1
        off = 0

        out_1 = np.where(cond, on, off)

        cond = np.less_equal(t, self.ramp_on_time + self.ramp_time)
        on = np.ones(np.shape(t)) * (t - self.ramp_on_time) / self.ramp_time
        off = 1

        out_2 = np.where(cond, on, off)

        return out_1 * out_2


class ExponentialWindow(ElectricFieldWindow):
    def __init__(self, window_time = 10 * asec, window_width = 10 * asec):
        self.window_time = window_time
        self.window_width = window_width

        super(ExponentialWindow, self).__init__()

    def __call__(self, t):
        return 1 / (1 + np.exp(-(t + self.window_time) / self.window_width)) - 1 / (1 + np.exp(-(t - self.window_time) / self.window_width))


class UniformLinearlyPolarizedElectricField(Potential):
    def __init__(self, window = NoWindow()):
        super(UniformLinearlyPolarizedElectricField, self).__init__()

        self.window = window

    def get_amplitude(self, t):
        return self.window(t)

    def __call__(self, *, t, distance_along_polarization, test_charge, **kwargs):
        return distance_along_polarization * test_charge * self.get_amplitude(t)


class Rectangle(UniformLinearlyPolarizedElectricField):
    def __init__(self, start_time = 0 * asec, end_time = 50 * asec, amplitude = 1 * atomic_electric_field, **kwargs):
        super(Rectangle, self).__init__(**kwargs)

        self.start_time = start_time
        self.end_time = end_time
        self.amplitude = amplitude

    def __str__(self):
        out = '{}(start time = {} as, end time = {} as, amplitude = {} AEF)'.format(self.__class__.__name__,
                                                                                    uround(self.start_time, asec, 3),
                                                                                    uround(self.end_time, asec, 3),
                                                                                    uround(self.amplitude, atomic_electric_field, 3))

        out += ' with {}'.format(self.window)

        return out

    def __repr__(self):
        out = '{}(start_time = {}, end_time = {}, amplitude = {}, window_function = {})'.format(self.__class__.__name__,
                                                                                                self.start_time,
                                                                                                self.end_time,
                                                                                                self.amplitude,
                                                                                                repr(self.window))

        return out

    def get_amplitude(self, t):
        cond = np.greater_equal(t, self.start_time) * np.less_equal(t, self.end_time)
        on = np.ones(np.shape(t))
        off = np.zeros(np.shape(t))

        out = np.where(cond, on, off) * self.amplitude * super(Rectangle, self).get_amplitude(t)

        return out * super(Rectangle, self).get_amplitude(t)


class SineWave(UniformLinearlyPolarizedElectricField):
    def __init__(self, omega, amplitude = 1 * atomic_electric_field, phase = 0, **kwargs):
        super(SineWave, self).__init__(**kwargs)

        self.omega = omega
        self.phase = phase % twopi
        self.amplitude = amplitude

    def __str__(self):
        out = '{}(omega = 2pi * {} THz, wavelength = {} nm, photon energy = {} eV, amplitude = {} AEF, phase = 2pi * {})'.format(self.__class__.__name__,
                                                                                                                                 uround(self.frequency, THz, 3),
                                                                                                                                 uround(self.wavelength, nm, 3),
                                                                                                                                 uround(self.photon_energy, eV, 3),
                                                                                                                                 uround(self.amplitude, atomic_electric_field, 3),
                                                                                                                                 uround(self.phase, twopi, 3))

        out += ' with {}'.format(self.window)

        return out

    def __repr__(self):
        out = '{}(omega = {}, amplitude = {}, phase = {}, window_function = {})'.format(self.__class__.__name__,
                                                                                        self.omega,
                                                                                        self.amplitude,
                                                                                        self.phase,
                                                                                        repr(self.window))

        return out

    @classmethod
    def from_frequency(cls, frequency, amplitude, phase = 0, window_time = None, window_width = None):
        return cls(frequency * twopi, amplitude, phase = phase, window_time = window_time, window_width = window_width)

    @property
    def frequency(self):
        return self.omega / twopi

    @frequency.setter
    def frequency(self, frequency):
        self.omega = frequency * twopi

    @property
    def period(self):
        return 1 / self.frequency

    @period.setter
    def period(self, period):
        self.frequency = 1 / period

    @property
    def wavelength(self):
        return c / self.frequency

    @wavelength.setter
    def wavelength(self, wavelength):
        self.frequency = c / wavelength

    @property
    def photon_energy(self):
        return hbar * self.omega

    @photon_energy.setter
    def photon_energy(self, photon_energy):
        self.omega = photon_energy / hbar

    def get_amplitude(self, t):
        return self.amplitude * np.sin((self.omega * t) + self.phase) * super(SineWave, self).get_amplitude(t)

    def get_peak_amplitude(self):
        return self.amplitude

    def get_peak_power_density(self):
        return 0.5 * c * epsilon_0 * (np.abs(self.amplitude) ** 2)  # TODO: check factor of 1/2 here
