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

    def __call__(self, *args, **kwargs):
        return 0


class NoPotential(Potential):
    pass


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


class HarmonicOscillatorPotential(Potential):
    def __init__(self, k = 8.41042 * N / m, center = 0 * nm):
        self.k = k
        self.center = center

    @classmethod
    def from_frequency_and_mass(cls, omega = 3.038535e15 * Hz, mass = electron_mass):
        return cls(k = mass * (omega ** 2))

    @classmethod
    def from_energy_and_mass(cls, ground_state_energy = 1 * eV, mass = electron_mass):
        return cls.from_frequency_and_mass(omega = 2 * ground_state_energy / hbar, mass = mass)

    def __call__(self, *, distance, **kwargs):
        return 0.5 * self.k * ((distance - self.center) ** 2)

    def omega(self, mass):
        return np.sqrt(self.k / mass)


class RadialImaginaryPotential(Potential):
    def __init__(self, center = 20 * bohr_radius, width = 2 * bohr_radius, decay_time = 100 * asec):
        """
        Construct a RadialImaginaryPotential. The potential is shaped like a Gaussian and has an imaginary amplitude.

        A positive/negative amplitude yields an imaginary potential that causes decay/amplification.

        :param center: the radial coordinate to center the potential on
        :param width: the width (FWHM) of the Gaussian
        :param amplitude: the peak amplitude of the Gaussian
        """
        self.center = center
        self.width = width
        self.decay_time = decay_time
        self.decay_rate = 1 / decay_time

        self.prefactor = -1j * self.decay_rate * hbar

    def __repr__(self):
        return '{}(center = {}, width = {}, decay_time = {})'.format(self.__class__.__name__, self.center, self.width, self.decay_time)

    def __str__(self):
        return '{}(center = {} Bohr radii, width = {} Bohr radii, decay time = {} as)'.format(self.__class__.__name__,
                                                                                              uround(self.center, bohr_radius, 3),
                                                                                              uround(self.width, bohr_radius, 3),
                                                                                              uround(self.decay_time, asec, 3))

    def __call__(self, *, r, **kwargs):
        return self.prefactor * np.exp(-(((r - self.center) / self.width) ** 2))


class ElectricFieldWindow(Potential):
    def __init__(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, t):
        raise NotImplementedError


class LinearRampWindow(ElectricFieldWindow):
    def __init__(self, ramp_on_time = 0 * asec, ramp_time = 50 * asec):
        self.ramp_on_time = ramp_on_time
        self.ramp_time = ramp_time

        # TODO: ramp_from and ramp_to

        super(LinearRampWindow, self).__init__()

    def __str__(self):
        return '{}(ramp on at = {} as, ramp time = {} as)'.format(self.__class__.__name__,
                                                                  uround(self.ramp_on_time, asec, 3),
                                                                  uround(self.ramp_time, asec, 3))

    def __repr__(self):
        return '{}(ramp_on_time = {}, ramp_time = {})'.format(self.__class__.__name__,
                                                              self.ramp_on_time,
                                                              self.ramp_time)

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


class SymmetricExponentialWindow(ElectricFieldWindow):
    def __init__(self, window_time = 500 * asec, window_width = 10 * asec):
        self.window_time = window_time
        self.window_width = window_width

        super(SymmetricExponentialWindow, self).__init__()

    def __str__(self):
        return '{}(window time = {} as, window width = {} as)'.format(self.__class__.__name__,
                                                                      uround(self.window_time, asec, 3),
                                                                      uround(self.window_width, asec, 3))

    def __repr__(self):
        return '{}(window_time = {}, window_width = {})'.format(self.__class__.__name__,
                                                                self.window_time,
                                                                self.window_width)

    def __call__(self, t):
        return np.abs(1 / (1 + np.exp(-(t + self.window_time) / self.window_width)) - 1 / (1 + np.exp(-(t - self.window_time) / self.window_width)))


class UniformLinearlyPolarizedElectricField(Potential):
    def __init__(self, window = None):
        super(UniformLinearlyPolarizedElectricField, self).__init__()

        self.window = window

    def __str__(self):
        if self.window:
            return ' with {}'.format(self.window)
        else:
            return ' with no window'

    def get_amplitude(self, t):
        if self.window is not None:
            return self.window(t)
        else:
            return 1

    def __call__(self, *, t, distance_along_polarization, test_charge, **kwargs):
        return distance_along_polarization * test_charge * self.get_amplitude(t)

    def get_total_electric_field(self, times):
        raise NotImplementedError

    def get_fluence(self, times):
        raise NotImplementedError


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

        return out + super(Rectangle, self).__str__()

    def __repr__(self):
        out = '{}(start_time = {}, end_time = {}, amplitude = {}, window = {})'.format(self.__class__.__name__,
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

        return out


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

        return out + super(SineWave, self).__str__()

    def __repr__(self):
        out = '{}(omega = {}, amplitude = {}, phase = {}, window = {})'.format(self.__class__.__name__,
                                                                               self.omega,
                                                                               self.amplitude,
                                                                               self.phase,
                                                                               repr(self.window))

        return out

    @classmethod
    def from_frequency(cls, frequency, amplitude = 1 * atomic_electric_field, phase = 0, **kwargs):
        return cls(frequency * twopi, amplitude = amplitude, phase = phase, **kwargs)

    @classmethod
    def from_photon_energy(cls, photon_energy, amplitude = 1 * atomic_electric_field, phase = 0, **kwargs):
        return cls.from_frequency(photon_energy / h, amplitude = amplitude, phase = phase, **kwargs)

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
        return np.sin((self.omega * t) + self.phase) * self.amplitude * super(SineWave, self).get_amplitude(t)

    def get_peak_amplitude(self):
        return self.amplitude

    def get_peak_power_density(self):
        return 0.5 * c * epsilon_0 * (np.abs(self.amplitude) ** 2)  # TODO: check factor of 1/2 here


class SincPulse(UniformLinearlyPolarizedElectricField):
    def __init__(self, pulse_width = 100 * asec, fluence = 1 * J / (cm ** 2), phase = 'cos', pulse_center = 0 * asec, **kwargs):
        super(SincPulse, self).__init__(**kwargs)

        self.pulse_width = pulse_width

        if phase not in ('cos', 'sin'):
            raise TypeError("SincPulse phase must be 'cos' or 'sin'")
        self.phase = phase

        self.fluence = fluence
        self.pulse_center = pulse_center

        self.omega_cutoff = twopi / self.pulse_width
        self.amplitude_density = np.sqrt(self.fluence / (2 * epsilon_0 * c * self.omega_cutoff))
        self.amplitude_prefactor = np.sqrt(2 / pi) * self.amplitude_density

    @property
    def largest_photon_energy(self):
        return hbar * self.omega_cutoff

    def __str__(self):
        out = '{}(pulse width = {} as, pulse center = {} as, fluence = {} J/cm^2, phase = {}, largest photon energy = {} eV)'.format(self.__class__.__name__,
                                                                                                                                     uround(self.pulse_width, asec, 3),
                                                                                                                                     uround(self.pulse_center, asec, 3),
                                                                                                                                     uround(self.fluence, J / (cm ** 2), 3),
                                                                                                                                     self.phase,
                                                                                                                                     uround(self.largest_photon_energy, eV, 3))

        return out + super(SincPulse, self).__str__()

    def __repr__(self):
        out = '{}(pulse width = {}, pulse center = {}, fluence = {}, phase = {}, window = {})'.format(self.__class__.__name__,
                                                                                                      self.pulse_width,
                                                                                                      self.pulse_center,
                                                                                                      self.fluence,
                                                                                                      self.phase,
                                                                                                      repr(self.window))

        return out

    def get_amplitude(self, t):
        if self.phase == 'cos':
            amp = np.where(np.not_equal(t, 0),
                           np.sin(self.omega_cutoff * (t - self.pulse_center)) / (t - self.pulse_center),
                           self.omega_cutoff)
        elif self.phase == 'sin':
            amp = np.where(np.not_equal(t, 0),
                           (np.cos(self.omega_cutoff * (t - self.pulse_center)) - 1) / (t - self.pulse_center),
                           0)

        return amp * self.amplitude_prefactor * super(SincPulse, self).get_amplitude(t)


class RandomizedSincPulse(UniformLinearlyPolarizedElectricField):
    def __init__(self, pulse_width = 100 * asec, fluence = 1 * J / (cm ** 2), divisions = 100, **kwargs):
        super(RandomizedSincPulse, self).__init__(**kwargs)

        self.pulse_width = pulse_width

        self.fluence = fluence
        self.divisions = divisions
        self.phases = twopi * np.random.rand(divisions)

        self.omega_cutoff = twopi / self.pulse_width
        self.amplitude_density = np.sqrt(self.fluence / (2 * epsilon_0 * c * self.omega_cutoff))
        self.amplitude_prefactor = np.sqrt(2 / pi) * self.amplitude_density

    @property
    def largest_photon_energy(self):
        return hbar * self.omega_cutoff

    def __str__(self):
        out = '{}(pulse width = {} as, pulse center = {} as, fluence = {} J/cm^2, phase = {}, largest photon energy = {} eV)'.format(self.__class__.__name__,
                                                                                                                                     uround(self.pulse_width, asec, 3),
                                                                                                                                     uround(self.pulse_center, asec, 3),
                                                                                                                                     uround(self.fluence, J / (cm ** 2), 3),
                                                                                                                                     self.phase,
                                                                                                                                     uround(self.largest_photon_energy, eV, 3))

        return out + super(RandomizedSincPulse, self).__str__()

    def __repr__(self):
        out = '{}(pulse width = {}, pulse center = {}, fluence = {}, phase = {}, window = {})'.format(self.__class__.__name__,
                                                                                                      self.pulse_width,
                                                                                                      self.pulse_center,
                                                                                                      self.fluence,
                                                                                                      self.phase,
                                                                                                      repr(self.window))

        return out

    def get_amplitude(self, t):
        raise NotImplementedError

        return amp * self.amplitude_prefactor * super(RandomizedSincPulse, self).get_amplitude(t)


class RadialCosineMask(Potential):
    def __init__(self, inner_radius = 50 * bohr_radius, outer_radius = 100 * bohr_radius, smoothness = 8):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.smoothness = smoothness

    def __str__(self):
        return '{}(inner radius = {} Bohr radii, outer radius = {} Bohr radii, smoothness = {})'.format(self.__class__.__name__,
                                                                                                        uround(self.inner_radius, bohr_radius, 3),
                                                                                                        uround(self.outer_radius, bohr_radius, 3),
                                                                                                        self.smoothness)

    def __repr__(self):
        return '{}(inner_radius = {}, outer_radius = {}, smoothness = {})'.format(self.__class__.__name__,
                                                                                  self.inner_radius,
                                                                                  self.outer_radius,
                                                                                  self.smoothness)

    def __call__(self, *, r, **kwargs):
        return np.where(np.greater_equal(r, self.inner_radius) * np.less_equal(r, self.outer_radius),
                        np.abs(np.cos(0.5 * pi * (r - self.inner_radius) / np.abs(self.outer_radius - self.inner_radius))) ** (1 / self.smoothness),
                        np.where(np.greater_equal(r, self.outer_radius), 0, 1))
