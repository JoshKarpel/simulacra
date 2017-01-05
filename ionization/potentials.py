import logging

import numpy as np
import scipy.integrate as integ

import compy as cp
from compy.units import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PotentialEnergy(cp.Summand):
    def __init__(self, *args, **kwargs):
        super(PotentialEnergy, self).__init__()
        self.summation_class = PotentialEnergySum


class PotentialEnergySum(cp.Sum, PotentialEnergy):
    container_name = 'potentials'


class NoPotentialEnergy(PotentialEnergy):
    def __call__(self, *args, **kwargs):
        return 0


class TimeWindow(cp.Summand):
    def __init__(self):
        super(TimeWindow, self).__init__()
        self.summation_class = TimeWindowSum


class TimeWindowSum(cp.Sum, TimeWindow):
    container_name = 'windows'


class NoTimeWindow(TimeWindow):
    def __call__(self, t):
        return 1


class Mask(cp.Summand):
    def __init__(self):
        super(Mask, self).__init__()
        self.summation_class = MaskSum


class MaskSum(cp.Sum, Mask):
    container_name = 'masks'


class NoMask(Mask):
    def __call__(self, *args, **kwargs):
        return 1


class Coulomb(PotentialEnergy):
    """A PotentialEnergy representing the electric potential energy of the nucleus of a hydrogenic atom."""

    def __init__(self, charge = 1 * proton_charge):
        """Construct a Coulomb object with the given charge."""
        super(Coulomb, self).__init__()

        self.charge = charge

    def __str__(self):
        return cp.utils.field_str(self, ('charge', 'proton_charge'))

    def __repr__(self):
        return cp.utils.field_str(self, 'charge')

    def __call__(self, *, r, test_charge, **kwargs):
        """Return the Coulomb potential energy evaluated at radial distance r for charge test_charge."""
        return coulomb_force_constant * self.charge * test_charge / r


class HarmonicOscillator(PotentialEnergy):
    """A PotentialEnergy representing the potential energy of a harmonic oscillator."""

    def __init__(self, spring_constant = 4.20521 * N / m, center = 0 * nm):
        """Construct a HarmonicOscillator object with the given spring constant and center position."""
        self.spring_constant = spring_constant
        self.center = center

        super(HarmonicOscillator, self).__init__()

    @classmethod
    def from_frequency_and_mass(cls, omega = 1.5192675e15 * Hz, mass = electron_mass):
        """Return a HarmonicOscillator constructed from the given angular frequency and mass."""
        return cls(spring_constant = mass * (omega ** 2))

    @classmethod
    def from_ground_state_energy_and_mass(cls, ground_state_energy = 0.5 * eV, mass = electron_mass):
        """
        Return a HarmonicOscillator constructed from the given ground state energy and mass.

        Note: the ground state energy is half of the energy spacing of the oscillator.
        """
        return cls.from_frequency_and_mass(omega = 2 * ground_state_energy / hbar, mass = mass)

    @classmethod
    def from_energy_spacing_and_mass(cls, energy_spacing = 1 * eV, mass = electron_mass):
        """
        Return a HarmonicOscillator constructed from the given state energy spacing and mass.

        Note: the ground state energy is half of the energy spacing of the oscillator.
        """
        return cls.from_frequency_and_mass(omega = energy_spacing / hbar, mass = mass)

    def __call__(self, *, distance, **kwargs):
        """Return the HarmonicOscillator potential energy evaluated at position distance."""
        return 0.5 * self.spring_constant * ((distance - self.center) ** 2)

    def omega(self, mass):
        """Return the angular frequency for this potential for the given mass."""
        return np.sqrt(self.spring_constant / mass)

    def frequency(self, mass):
        """Return the cyclic frequency for this potential for the given mass."""
        return self.omega(mass) / twopi

    def __str__(self):
        return '{}(spring_constant = {} N/m, center = {} nm'.format(self.__class__.__name__, self.spring_constant, uround(self.center, nm, 3))

    def __repr__(self):
        return '{}(spring_constant = {}, center = {}'.format(self.__class__.__name__, self.spring_constant, self.center)


class RadialImaginary(PotentialEnergy):
    def __init__(self, center = 20 * bohr_radius, width = 2 * bohr_radius, decay_time = 100 * asec):
        """
        Construct a RadialImaginary. The potential is shaped like a Gaussian and has an imaginary amplitude.

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

        super(RadialImaginary, self).__init__()

    def __repr__(self):
        return '{}(center = {}, width = {}, decay_time = {})'.format(self.__class__.__name__, self.center, self.width, self.decay_time)

    def __str__(self):
        return '{}(center = {} Bohr radii, width = {} Bohr radii, decay time = {} as)'.format(self.__class__.__name__,
                                                                                              uround(self.center, bohr_radius, 3),
                                                                                              uround(self.width, bohr_radius, 3),
                                                                                              uround(self.decay_time, asec, 3))

    def __call__(self, *, r, **kwargs):
        return self.prefactor * np.exp(-(((r - self.center) / self.width) ** 2))


class UniformLinearlyPolarizedElectricField(PotentialEnergy):
    def __init__(self, window = NoTimeWindow()):
        super(UniformLinearlyPolarizedElectricField, self).__init__()

        self.window = window

    def __str__(self):
        return ' with {}'.format(self.window)

    def get_electric_field_amplitude(self, t):
        return self.window(t)

    def __call__(self, *, t, distance_along_polarization, test_charge, **kwargs):
        return distance_along_polarization * test_charge * self.get_electric_field_amplitude(t)

    def get_total_electric_field_numeric(self, times):
        """Return the integral of the electric field amplitude from the start of times for each interval in times."""
        return np.cumsum(self.get_electric_field_amplitude(times)) * np.abs(times[1] - times[0])

    def get_fluence_numeric(self, times):
        raise NotImplementedError


class NoElectricField(UniformLinearlyPolarizedElectricField):
    def __str__(self):
        return self.__class__.__name__ + super(NoElectricField, self).__str__()

    def get_electric_field_amplitude(self, t):
        return 0 * super(NoElectricField, self).get_electric_field_amplitude(t)


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

    def get_electric_field_amplitude(self, t):
        cond = np.greater_equal(t, self.start_time) * np.less_equal(t, self.end_time)
        on = np.ones(np.shape(t))
        off = np.zeros(np.shape(t))

        out = np.where(cond, on, off) * self.amplitude * super(Rectangle, self).get_electric_field_amplitude(t)

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

    def get_electric_field_amplitude(self, t):
        return np.sin((self.omega * t) + self.phase) * self.amplitude * super(SineWave, self).get_electric_field_amplitude(t)

    def get_peak_amplitude(self):
        return self.amplitude

    def get_peak_power_density(self):
        return 0.5 * c * epsilon_0 * (np.abs(self.amplitude) ** 2)  # TODO: check factor of 1/2 here


class SincPulse(UniformLinearlyPolarizedElectricField):
    def __init__(self, pulse_width = 200 * asec, fluence = 1 * J / (cm ** 2), phase = 'cos', pulse_center = 0 * asec,
                 dc_correction_time = None,
                 **kwargs):
        """

        :param pulse_width:
        :param fluence:
        :param phase:
        :param pulse_center:
        :param dc_correction_time: if given, the dc field component for cosine phase will be equalized at this time
        :param kwargs:
        """
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

        self.dc_correction_time = dc_correction_time

    @classmethod
    def from_amplitude_density(cls, pulse_width = 100 * asec, amplitude_density = 7.7432868731566454e-06, phase = 'cos', pulse_center = 0 * asec, **kwargs):
        omega_cutoff = twopi / pulse_width
        fluence = (amplitude_density ** 2) * (2 * epsilon_0 * c * omega_cutoff)
        return cls(pulse_width = pulse_width, fluence = fluence, phase = phase, pulse_center = pulse_center, **kwargs)

    @property
    def largest_photon_energy(self):
        return hbar * self.omega_cutoff

    @property
    def frequency_cutoff(self):
        return self.omega_cutoff / twopi

    def __str__(self):
        out = cp.utils.field_str(self,
                                 ('pulse_width', 'asec'),
                                 ('pulse_center', 'asec'),
                                 ('fluence', 'J/cm^2'),
                                 'phase',
                                 ('largest_photon_energy', 'eV'),
                                 ('dc_correction_time', 'asec'))

        return out + super(SincPulse, self).__str__()

    def __repr__(self):
        return cp.utils.field_str(self,
                                  'pulse_width',
                                  'pulse_center',
                                  'fluence',
                                  'phase',
                                  'largest_photon_energy',
                                  'dc_correction_time')

    def get_electric_field_amplitude(self, t):
        if self.phase == 'cos':
            amp = np.where(np.not_equal(t, 0),
                           (np.sin(self.omega_cutoff * (t - self.pulse_center)) / (t - self.pulse_center)),
                           self.omega_cutoff)

            if self.dc_correction_time is not None:
                amp -= pi / (2 * self.dc_correction_time)
        elif self.phase == 'sin':
            amp = np.where(np.not_equal(t, 0),
                           (np.cos(self.omega_cutoff * (t - self.pulse_center)) - 1) / (t - self.pulse_center),
                           0)

        return amp * self.amplitude_prefactor * super(SincPulse, self).get_electric_field_amplitude(t)


# class RandomizedSincPulse(UniformLinearlyPolarizedElectricField):
#     def __init__(self, pulse_width = 100 * asec, fluence = 1 * J / (cm ** 2), divisions = 100, **kwargs):
#         super(RandomizedSincPulse, self).__init__(**kwargs)
#
#         self.pulse_width = pulse_width
#
#         self.fluence = fluence
#         self.divisions = divisions
#         self.phases = twopi * np.random.rand(divisions)
#
#         self.omega_cutoff = twopi / self.pulse_width
#         self.amplitude_density = np.sqrt(self.fluence / (2 * epsilon_0 * c * self.omega_cutoff))
#         self.amplitude_prefactor = np.sqrt(2 / pi) * self.amplitude_density
#
#     @property
#     def largest_photon_energy(self):
#         return hbar * self.omega_cutoff
#
#     def __str__(self):
#         out = '{}(pulse width = {} as, pulse center = {} as, fluence = {} J/cm^2, phase = {}, largest photon energy = {} eV)'.format(self.__class__.__name__,
#                                                                                                                                      uround(self.pulse_width, asec, 3),
#                                                                                                                                      uround(self.pulse_center, asec, 3),
#                                                                                                                                      uround(self.fluence, J / (cm ** 2), 3),
#                                                                                                                                      self.phase,
#                                                                                                                                      uround(self.largest_photon_energy, eV, 3))
#
#         return out + super(RandomizedSincPulse, self).__str__()
#
#     def __repr__(self):
#         out = '{}(pulse width = {}, pulse center = {}, fluence = {}, phase = {}, window = {})'.format(self.__class__.__name__,
#                                                                                                       self.pulse_width,
#                                                                                                       self.pulse_center,
#                                                                                                       self.fluence,
#                                                                                                       self.phase,
#                                                                                                       repr(self.window))
#
#         return out
#
#     def get_electric_field_amplitude(self, t):
#         raise NotImplementedError
#
#         # return amp * self.amplitude_prefactor * super(RandomizedSincPulse, self).get_amplitude(t)


class LinearRampTimeWindow(TimeWindow):
    def __init__(self, ramp_on_time = 0 * asec, ramp_time = 50 * asec):
        self.ramp_on_time = ramp_on_time
        self.ramp_time = ramp_time

        # TODO: ramp_from and ramp_to

        super(LinearRampTimeWindow, self).__init__()

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


class SymmetricExponentialTimeWindow(TimeWindow):
    def __init__(self, window_time = 500 * asec, window_width = 10 * asec):
        self.window_time = window_time
        self.window_width = window_width

        super(SymmetricExponentialTimeWindow, self).__init__()

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


class RadialCosineMask(Mask):
    def __init__(self, inner_radius = 50 * bohr_radius, outer_radius = 100 * bohr_radius, smoothness = 8):
        super(RadialCosineMask, self).__init__()
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
