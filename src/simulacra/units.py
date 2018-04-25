from typing import Union, NewType, Optional, Tuple

import numpy as _np

UNIT_NAME_TO_VALUE = {None: 1, '': 1}
UNIT_NAME_TO_LATEX = {None: '', '': ''}

Unit = NewType('Unit', Union[float, int, str])
TeXString = NewType('TeXString', str)


def get_unit_value_and_latex_from_unit(unit: Optional[Unit]) -> Tuple[float, TeXString]:
    """Return the numerical value of the unit and its LaTeX representation from a unit name."""
    if unit is None:
        unit = 1

    if type(unit) == str:
        unit_value = UNIT_NAME_TO_VALUE[unit]
        unit_latex = UNIT_NAME_TO_LATEX[unit]
    else:
        unit_value = unit
        unit_latex = ''

    return unit_value, unit_latex


def uround(value, unit: Optional[Unit] = None, digits: int = 3):
    """Round value to the number of digits, represented in the given units (by name or value)."""
    unit_value, _ = get_unit_value_and_latex_from_unit(unit)

    return _np.around(value / unit_value, digits)


# dimensionless constants
alpha = 7.2973525664e-3
pi = _np.pi
twopi = 2 * _np.pi
e = _np.e

UNIT_NAME_TO_VALUE.update({
    'alpha': alpha,
    'pi': pi,
    'twopi': twopi,
    'e': e
})
UNIT_NAME_TO_LATEX.update({
    'alpha': r'\alpha',
    'pi': '\pi',
    'twopi': r'2\pi',
    'e': 'e'
})

# base units
m = 1
s = sec = 1
kg = 1
A = 1
K = 1
rad = 1
deg = twopi / 360

UNIT_NAME_TO_VALUE.update({
    'm': m,
    's': s,
    'kg': kg,
    'A': A,
    'K': K,
    'rad': rad,
    'radian': rad,
    'deg': deg,
    'degrees': deg,
})
UNIT_NAME_TO_LATEX.update({
    'm': r'\mathrm{m}',
    's': r'\mathrm{s}',
    'kg': r'\mathrm{kg}',
    'A': r'\mathrm{A}',
    'K': r'\mathrm{K}',
    'rad': r'\mathrm{rad}',
    'radian': r'\mathrm{rad}',
    'deg': r'\mathrm{deg}',
    'degrees': r'\mathrm{deg}',
})

# time
msec = 1e-3 * s
usec = 1e-6 * s
nsec = 1e-9 * s
psec = 1e-12 * s
fsec = 1e-15 * s
asec = 1e-18 * s
minute = 60 * s
hours = hour = 60 * minute
days = day = 24 * hour
weeks = week = 7 * day
years = year = 365 * day

UNIT_NAME_TO_VALUE.update({
    'ms': msec,
    'msec': msec,
    'us': usec,
    'usec': usec,
    'ns': nsec,
    'nsec': nsec,
    'ps': psec,
    'psec': psec,
    'fs': fsec,
    'fsec': fsec,
    'as': asec,
    'asec': asec,
    'minute': minute,
    'minutes': minute,
    'hour': hour,
    'hours': hour,
    'day': day,
    'days': day,
    'week': week,
    'weeks': week,
    'year': year,
    'years': year
})
UNIT_NAME_TO_LATEX.update({
    'ms': r'\mathrm{ms}',
    'msec': r'\mathrm{ms}',
    'us': r'\mathrm{us}',
    'usec': r'\mathrm{us}',
    'ns': r'\mathrm{ns}',
    'nsec': r'\mathrm{ns}',
    'ps': r'\mathrm{ps}',
    'psec': r'\mathrm{ps}',
    'fs': r'\mathrm{fs}',
    'fsec': r'\mathrm{fs}',
    'as': r'\mathrm{as}',
    'asec': r'\mathrm{as}',
    'minute': '\mathrm{minutes}',
    'minutes': '\mathrm{minutes}',
    'hour': '\mathrm{hours}',
    'hours': '\mathrm{hours}',
    'day': '\mathrm{days}',
    'days': '\mathrm{days}',
    'week': '\mathrm{weeks}',
    'weeks': '\mathrm{weeks}',
    'year': '\mathrm{years}',
    'years': '\mathrm{years}'
})

# distance
cm = 1e-2 * m
mm = 1e-3 * m
um = 1e-6 * m
nm = 1e-9 * m
per_nm = 1 / nm
pm = 1e-12 * m
fm = 1e-15 * m
km = 1e3 * m
Mm = 1e6 * m
Gm = 1e9 * m
Tm = 1e12 * m
Pm = 1e15 * m
angstrom = 1e-10 * m
bohr_radius = 5.2917721067e-11 * m
inch = 2.54 * cm

UNIT_NAME_TO_VALUE.update({
    'cm': cm,
    'mm': mm,
    'um': um,
    'nm': nm,
    'pm': pm,
    'fm': fm,
    'km': km,
    'Mm': Mm,
    'Gm': Gm,
    'Tm': Tm,
    'Pm': Pm,
    'angstrom': angstrom,
    'bohr_radius': bohr_radius,
    'inch': inch,
    'per_nm': per_nm
})
UNIT_NAME_TO_LATEX.update({
    'cm': r'\mathrm{cm}',
    'centimeter': r'\mathrm{cm}',
    'centimeters': r'\mathrm{cm}',
    'mm': r'\mathrm{mm}',
    'millimeter': r'\mathrm{mm}',
    'millimeters': r'\mathrm{mm}',
    'um': r'\mathrm{um}',
    'micrometer': r'\mathrm{um}',
    'micrometers': r'\mathrm{um}',
    'micron': r'\mathrm{um}',
    'microns': r'\mathrm{um}',
    'nm': r'\mathrm{nm}',
    'nanometer': r'\mathrm{nm}',
    'nanometers': r'\mathrm{nm}',
    'pm': r'\mathrm{pm}',
    'picometer': r'\mathrm{pm}',
    'picometers': r'\mathrm{pm}',
    'fm': r'\mathrm{fm}',
    'femtometer': r'\mathrm{fm}',
    'femtometers': r'\mathrm{fm}',
    'km': r'\mathrm{km}',
    'kilometer': r'\mathrm{km}',
    'kilometers': r'\mathrm{km}',
    'Mm': r'\mathrm{Mm}',
    'Gm': r'\mathrm{Gm}',
    'Tm': r'\mathrm{Tm}',
    'Pm': r'\mathrm{Pm}',
    'angstrom': r'\mathrm{\AA}',
    'bohr_radius': r'a_0',
    'inch': r'\mathrm{in}',
    'per_nm': r'\mathrm{nm^{-1}}'
})

# mass
g = 1e-3 * kg
mg = 1e-3 * g
ug = 1e-6 * g
ng = 1e-9 * g
pg = 1e-12 * g
fg = 1e-15 * g
proton_mass = 1.672621898e-27 * kg
neutron_mass = 1.674927471e-27 * kg
electron_mass = 9.10938356e-31 * kg
electron_mass_reduced = proton_mass * electron_mass / (proton_mass + electron_mass)

UNIT_NAME_TO_VALUE.update({
    'g': g,
    'mg': mg,
    'ug': ug,
    'ng': ng,
    'pg': pg,
    'fg': fg,
    'proton_mass': proton_mass,
    'neutron_mass': neutron_mass,
    'electron_mass': electron_mass,
    'electron_mass_reduced': electron_mass_reduced
})
UNIT_NAME_TO_LATEX.update({
    'g': r'\mathrm{g}',
    'mg': r'\mathrm{mg}',
    'ug': r'\mathrm{ug}',
    'ng': r'\mathrm{ng}',
    'pg': r'\mathrm{pg}',
    'fg': r'\mathrm{fg}',
    'proton_mass': r'm_p',
    'neutron_mass': r'm_n',
    'electron_mass': r'm_e',
    'electron_mass_reduced': r'\mu_e'
})

# frequency
Hz = 1 / s
mHz = 1e-3 * Hz
uHz = 1e-6 * Hz
nHz = 1e-9 * Hz
pHz = 1e-12 * Hz
fHz = 1e-15 * Hz
kHz = 1e3 * Hz
MHz = 1e6 * Hz
GHz = 1e9 * Hz
THz = 1e12 * Hz
PHz = 1e15 * Hz

UNIT_NAME_TO_VALUE.update({
    'Hz': Hz,
    'mHz': mHz,
    'uHz': uHz,
    'nHz': nHz,
    'pHz': pHz,
    'fHz': fHz,
    'kHz': kHz,
    'MHz': MHz,
    'GHz': GHz,
    'THz': THz,
    'PHz': PHz
})
UNIT_NAME_TO_LATEX.update({
    'Hz': r'\mathrm{Hz}',
    'mHz': r'\mathrm{mHz}',
    'uHz': r'\mathrm{uHz}',
    'nHz': r'\mathrm{nHz}',
    'pHz': r'\mathrm{pHz}',
    'fHz': r'\mathrm{fHz}',
    'kHz': r'\mathrm{kHz}',
    'MHz': r'\mathrm{MHz}',
    'GHz': r'\mathrm{GHz}',
    'THz': r'\mathrm{THz}',
    'PHz': r'\mathrm{PHz}'
})

# frequency, again
per_sec = 1 / sec
per_msec = 1 / msec
per_usec = 1 / usec
per_nsec = 1 / nsec
per_psec = 1 / psec
per_fsec = 1 / fsec
per_asec = 1 / asec
per_minute = 1 / minute
per_hour = 1 / hour
per_day = 1 / day
per_week = 1 / week
per_year = 1 / year

UNIT_NAME_TO_VALUE.update({
    'per_s': per_sec,
    'per_sec': per_sec,
    'per_ms': per_msec,
    'per_msec': per_msec,
    'per_us': per_usec,
    'per_usec': per_usec,
    'per_ns': per_nsec,
    'per_nsec': per_nsec,
    'per_ps': per_psec,
    'per_psec': per_psec,
    'per_fs': per_fsec,
    'per_fsec': per_fsec,
    'per_as': per_asec,
    'per_asec': per_asec,
    'per_minute': per_minute,
    'per_hour': per_hour,
    'per_day': per_day,
    'per_week': per_week,
    'per_year': per_year,
})
UNIT_NAME_TO_LATEX.update({
    'per_s': r'\mathrm{s^{-1}}',
    'per_sec': r'\mathrm{s^{-1}}',
    'per_ms': r'\mathrm{ms^{-1}}',
    'per_msec': r'\mathrm{ms^{-1}}',
    'per_us': r'\mathrm{us^{-1}}',
    'per_usec': r'\mathrm{us^{-1}}',
    'per_ns': r'\mathrm{ns^{-1}}',
    'per_nsec': r'\mathrm{ns^{-1}}',
    'per_ps': r'\mathrm{ps^{-1}}',
    'per_psec': r'\mathrm{ps^{-1}}',
    'per_fs': r'\mathrm{fs^{-1}}',
    'per_fsec': r'\mathrm{fs^{-1}}',
    'per_as': r'\mathrm{as^{-1}}',
    'per_asec': r'\mathrm{as^{-1}}',
    'per_minute': '\mathrm{minute^{-1}}',
    'per_hour': '\mathrm{hour^{-1}}',
    'per_day': '\mathrm{day^{-1}}',
    'per_week': '\mathrm{week^{-1}}',
    'per_year': '\mathrm{year^{-1}}',
})

# electric charge
C = 1 * A * s
mC = 1e-3 * C
uC = 1e-6 * C
nC = 1e-9 * C
pC = 1e-12 * C
fC = 1e-15 * C
kC = 1e3 * C
MC = 1e6 * C
GC = 1e9 * C
TC = 1e12 * C
PC = 1e15 * C
proton_charge = 1.6021766208e-19 * C
electron_charge = -proton_charge

UNIT_NAME_TO_VALUE.update({
    'C': C,
    'mC': mC,
    'uC': uC,
    'nC': nC,
    'pC': pC,
    'fC': fC,
    'kC': kC,
    'MC': MC,
    'GC': GC,
    'TC': TC,
    'PC': PC,
    'proton_charge': proton_charge,
    'electron_charge': electron_charge
})
UNIT_NAME_TO_LATEX.update({
    'C': r'\mathrm{C}',
    'mC': r'\mathrm{mC}',
    'uC': r'\mathrm{uC}',
    'nC': r'\mathrm{nC}',
    'pC': r'\mathrm{pC}',
    'fC': r'\mathrm{fC}',
    'kC': r'\mathrm{kC}',
    'MC': r'\mathrm{MC}',
    'GC': r'\mathrm{GC}',
    'TC': r'\mathrm{TC}',
    'PC': r'\mathrm{PC}',
    'proton_charge': r'e',
    'electron_charge': r'-e'
})

# energy
J = 1 * kg * ((m / s) ** 2)
mJ = 1e-3 * J
uJ = 1e-6 * J
nJ = 1e-9 * J
pJ = 1e-12 * J
fJ = 1e-15 * J
kJ = 1e3 * J
MJ = 1e6 * J
GJ = 1e9 * J
TJ = 1e12 * J
PJ = 1e15 * J
Jcm2 = J / (cm ** 2)

UNIT_NAME_TO_VALUE.update({
    'J': J,
    'mJ': mJ,
    'uJ': uJ,
    'nJ': nJ,
    'pJ': pJ,
    'fJ': fJ,
    'kJ': kJ,
    'MJ': MJ,
    'GJ': GJ,
    'TJ': TJ,
    'PJ': PJ,
    'J/cm^2': Jcm2,
    'Jcm2': Jcm2,
})
UNIT_NAME_TO_LATEX.update({
    'J': r'\mathrm{J}',
    'mJ': r'\mathrm{mJ}',
    'uJ': r'\mathrm{uJ}',
    'nJ': r'\mathrm{nJ}',
    'pJ': r'\mathrm{pJ}',
    'fJ': r'\mathrm{fJ}',
    'kJ': r'\mathrm{kJ}',
    'MJ': r'\mathrm{MJ}',
    'GJ': r'\mathrm{GJ}',
    'TJ': r'\mathrm{TJ}',
    'PJ': r'\mathrm{PJ}',
    'J/cm^2': r'\mathrm{J/cm^2}',
    'Jcm2': r'\mathrm{J/cm^2}'
})

# voltage
V = 1 * J / C
mV = 1e-3 * V
uV = 1e-6 * V
nV = 1e-9 * V
pV = 1e-12 * V
fV = 1e-15 * V
kV = 1e3 * V
MV = 1e6 * V
GV = 1e9 * V
TV = 1e12 * V
PV = 1e15 * V

UNIT_NAME_TO_VALUE.update({
    'V': V,
    'mV': mV,
    'uV': uV,
    'nV': nV,
    'pV': pV,
    'fV': fV,
    'kV': kV,
    'MV': MV,
    'GV': GV,
    'TV': TV,
    'PV': PV
})
UNIT_NAME_TO_LATEX.update({
    'V': r'\mathrm{V}',
    'mV': r'\mathrm{mV}',
    'uV': r'\mathrm{uV}',
    'nV': r'\mathrm{nV}',
    'pV': r'\mathrm{pV}',
    'fV': r'\mathrm{fV}',
    'kV': r'\mathrm{kV}',
    'MV': r'\mathrm{MV}',
    'GV': r'\mathrm{GV}',
    'TV': r'\mathrm{TV}',
    'PV': r'\mathrm{PV}'
})

# electric field strength
V_per_m = V / m

UNIT_NAME_TO_VALUE.update({
    'V_per_m': V_per_m,
    'V/m': V_per_m,
})
UNIT_NAME_TO_LATEX.update({
    'V_per_m': r'\mathrm{V/m}',
    'V/m': r'\mathrm{V/m}',
})

# magnetic field strength
T = V * s / (m ** 2)
mT = 1e-3 * T
uT = 1e-6 * T
nT = 1e-9 * T
pT = 1e-12 * T
fT = 1e-15 * T
kT = 1e3 * T
MT = 1e6 * T
GT = 1e9 * T
TT = 1e12 * T
PT = 1e15 * T

UNIT_NAME_TO_VALUE.update({
    'T': T,
    'mT': mT,
    'uT': uT,
    'nT': nT,
    'pT': pT,
    'fT': fT,
    'kT': kT,
    'MT': MT,
    'GT': GT,
    'TT': TT,
    'PT': PT
})
UNIT_NAME_TO_LATEX.update({
    'T': r'\mathrm{T}',
    'mT': r'\mathrm{mT}',
    'uT': r'\mathrm{uT}',
    'nT': r'\mathrm{nT}',
    'pT': r'\mathrm{pT}',
    'fT': r'\mathrm{fT}',
    'kT': r'\mathrm{kT}',
    'MT': r'\mathrm{MT}',
    'GT': r'\mathrm{GT}',
    'TT': r'\mathrm{TT}',
    'PT': r'\mathrm{PT}'
})

# energies in electron-volts
eV = proton_charge * V
keV = 1e3 * eV
MeV = 1e6 * eV
GeV = 1e9 * eV
TeV = 1e12 * eV
PeV = 1e15 * eV

UNIT_NAME_TO_VALUE.update({
    'eV': eV,
    'keV': keV,
    'MeV': MeV,
    'GeV': GeV,
    'TeV': TeV,
    'PeV': PeV
})
UNIT_NAME_TO_LATEX.update({
    'eV': r'\mathrm{eV}',
    'keV': r'\mathrm{keV}',
    'MeV': r'\mathrm{MeV}',
    'GeV': r'\mathrm{GeV}',
    'TeV': r'\mathrm{TeV}',
    'PeV': r'\mathrm{PeV}'
})

# force
N = 1 * m * kg / (s ** 2)
mN = 1e-3 * N
uN = 1e-6 * N
nN = 1e-9 * N
pN = 1e-12 * N
fN = 1e-15 * N
kN = 1e3 * N
MN = 1e6 * N
GN = 1e9 * N
TN = 1e12 * N
PN = 1e15 * N

UNIT_NAME_TO_VALUE.update({
    'N': N,
    'mN': mN,
    'uN': uN,
    'nN': nN,
    'pN': pN,
    'fN': fN,
    'kN': kN,
    'MN': MN,
    'GN': GN,
    'TN': TN,
    'PN': PN
})
UNIT_NAME_TO_LATEX.update({
    'N': r'\mathrm{N}',
    'mN': r'\mathrm{mN}',
    'uN': r'\mathrm{uN}',
    'nN': r'\mathrm{nN}',
    'pN': r'\mathrm{pN}',
    'fN': r'\mathrm{fN}',
    'kN': r'\mathrm{kN}',
    'MN': r'\mathrm{MN}',
    'GN': r'\mathrm{GN}',
    'TN': r'\mathrm{TN}',
    'PN': r'\mathrm{PN}'
})

# spring constant
N_per_m = N / m

UNIT_NAME_TO_VALUE.update({
    'N_per_m': N_per_m,
    'N/m': N_per_m,
})
UNIT_NAME_TO_LATEX.update({
    'N_per_m': r'\mathrm{N/m}',
    'N/m': r'\mathrm{N/m}',
})

# power
W = 1 * J / s
mW = 1e-3 * W
uW = 1e-6 * W
nW = 1e-9 * W
pW = 1e-12 * W
fW = 1e-15 * W
kW = 1e3 * W
MW = 1e6 * W
GW = 1e9 * W
TW = 1e12 * W
PW = 1e15 * W
Wcm2 = W / (cm ** 2)
TWcm2 = TW / (cm ** 2)

UNIT_NAME_TO_VALUE.update({
    'W': W,
    'mW': mW,
    'uW': uW,
    'nW': nW,
    'pW': pW,
    'fW': fW,
    'kW': kW,
    'MW': MW,
    'GW': GW,
    'TW': TW,
    'PW': PW,
    'Wcm2': Wcm2,
    'TWcm2': TWcm2,
})
UNIT_NAME_TO_LATEX.update({
    'W': r'\mathrm{W}',
    'mW': r'\mathrm{mW}',
    'uW': r'\mathrm{uW}',
    'nW': r'\mathrm{nW}',
    'pW': r'\mathrm{pW}',
    'fW': r'\mathrm{fW}',
    'kW': r'\mathrm{kW}',
    'MW': r'\mathrm{MW}',
    'GW': r'\mathrm{GW}',
    'TW': r'\mathrm{TW}',
    'PW': r'\mathrm{PW}',
    'Wcm2': r'\mathrm{W / cm^2}',
    'TWcm2': r'\mathrm{TW / cm^2}',
})

# current
A = 1 * C / s
mA = 1e-3 * A
uA = 1e-6 * A
nA = 1e-9 * A
pA = 1e-12 * A
fA = 1e-15 * A
kA = 1e3 * A
MA = 1e6 * A
GA = 1e9 * A
TA = 1e12 * A
PA = 1e15 * A

UNIT_NAME_TO_VALUE.update({
    'A': A,
    'mA': mA,
    'uA': uA,
    'nA': nA,
    'pA': pA,
    'fA': fA,
    'kA': kA,
    'MA': MA,
    'GA': GA,
    'TA': TA,
    'PA': PA
})
UNIT_NAME_TO_LATEX.update({
    'A': r'\mathrm{A}',
    'mA': r'\mathrm{mA}',
    'uA': r'\mathrm{uA}',
    'nA': r'\mathrm{nA}',
    'pA': r'\mathrm{pA}',
    'fA': r'\mathrm{fA}',
    'kA': r'\mathrm{kA}',
    'MA': r'\mathrm{MA}',
    'GA': r'\mathrm{GA}',
    'TA': r'\mathrm{TA}',
    'PA': r'\mathrm{PA}'
})

# thermodynamics
k_B = 1.38064852e-23 * J / K
boltzmann_constant = k_B

UNIT_NAME_TO_VALUE.update({
    'k_B': k_B,
    'boltzmann_constant': k_B,
})
UNIT_NAME_TO_LATEX.update({
    'k_B': r'k_B',
    'boltzmann_constant': r'k_B',
})

# speed of light and E&M
c = speed_of_light = 299792458 * m / s
mu_0 = vacuum_permeability = pi * 4e-7 * N / (A ** 2)
epsilon_0 = vacuum_permittivity = 1 / (mu_0 * (c ** 2))
k_e = coulomb_constant = 1 / (4 * pi * epsilon_0)
n_vacuum = 1

UNIT_NAME_TO_VALUE.update({
    'c': c,
    'speed_of_light': c,
    'mu_0': mu_0,
    'vacuum_permeability': mu_0,
    'epsilon_0': epsilon_0,
    'vacuum_permittivity': epsilon_0,
    'coulomb_constant': coulomb_constant,
    'coulomb_force_constant': coulomb_constant,
    'n_vacuum': n_vacuum
})
UNIT_NAME_TO_LATEX.update({
    'c': r'c',
    'speed_of_light': r'c',
    'mu_0': r'\mu_0',
    'vacuum_permeability': r'\mu_0',
    'epsilon_0': r'\epsilon_0',
    'vacuum_permittivity': r'\epsilon_0',
    'coulomb_constant': r'k_e',
    'coulomb_force_constant': r'k_e',
    'k_e': r'k_e',
    'n_vacuum': r'n_{\mathrm{vac}}'
})

# quantum mechanics
h = 6.626_070_040e-34 * J * s  # Planck's constant
hbar = h / twopi
rydberg = 13.605_693_009 * eV
hartree = 2 * rydberg

UNIT_NAME_TO_VALUE.update({
    'h': h,
    'hbar': hbar,
    'rydberg': rydberg,
    'hartree': hartree
})
UNIT_NAME_TO_LATEX.update({
    'h': r'h',
    'hbar': r'\hbar',
    'rydberg': r'\mathrm{Ry}',
    'hartree': r'\mathrm{Ha}',
    'atomic_energy': r'\mathrm{Ha}',
})

atomic_energy = hartree
atomic_electric_field = coulomb_constant * proton_charge / (bohr_radius ** 2)
atomic_magnetic_field = hbar / (proton_charge * (bohr_radius ** 2))
atomic_intensity = .5 * epsilon_0 * c * (atomic_electric_field ** 2)
atomic_electric_potential = coulomb_constant * proton_charge / bohr_radius
atomic_electric_dipole_moment = proton_charge * bohr_radius
atomic_magnetic_dipole_moment = bohr_magneton = proton_charge * hbar / (2 * electron_mass)
atomic_velocity = alpha * c
atomic_momentum = electron_mass * atomic_velocity
atomic_time = hbar / hartree
per_atomic_time = 1 / atomic_time
atomic_angular_frequency = 1 / atomic_time
atomic_frequency = atomic_angular_frequency / twopi
atomic_force = hartree / bohr_radius
atomic_temperature = hartree / k_B
per_bohr_radius = 1 / bohr_radius

UNIT_NAME_TO_VALUE.update({
    'atomic_energy': atomic_energy,
    'atomic_electric_field': atomic_electric_field,
    'atomic_magnetic_field': atomic_magnetic_field,
    'aef': atomic_electric_field,
    'AEF': atomic_electric_field,
    'atomic_intensity': atomic_intensity,
    'atomic_electric_potential': atomic_electric_potential,
    'atomic_electric_dipole_moment': atomic_electric_dipole_moment,
    'atomic_magnetic_dipole_moment': atomic_magnetic_dipole_moment,
    'bohr_magneton': atomic_magnetic_dipole_moment,
    'atomic_velocity': atomic_velocity,
    'atomic_momentum': atomic_momentum,
    'atomic_time': atomic_time,
    'per_atomic_time': per_atomic_time,
    'atomic_frequency': atomic_frequency,
    'atomic_angular_frequency': atomic_angular_frequency,
    'atomic_force': atomic_force,
    'atomic_temperature': atomic_temperature,
    'per_bohr_radius': per_bohr_radius,
})
UNIT_NAME_TO_LATEX.update({
    'atomic_electric_field': r'\mathrm{a.u.}',
    'atomic_magnetic_field': r'\mathrm{a.u.}',
    'aef': r'\mathrm{a.u.}',
    'AEF': r'\mathrm{a.u.}',
    'atomic_intensity': r'\mathrm{a.u.}',
    'atomic_electric_potential': r'\mathrm{a.u.}',
    'atomic_electric_dipole_moment': r'e \, a_0',
    'atomic_magnetic_dipole_moment': r'\mu_B',
    'bohr_magneton': r'\mu_B',
    'atomic_velocity': r'\mathrm{a.u.}',
    'atomic_momentum': r'\mathrm{a.u.}',
    'atomic_time': r'\mathrm{a.u.}',
    'per_atomic_time': r'\mathrm{a.u.}',
    'atomic_frequency': r'\mathrm{a.u.}',
    'atomic_angular_frequency': r'\mathrm{a.u.}',
    'atomic_force': r'\mathrm{a.u.}',
    'atomic_temperature': r'\mathrm{a.u.}',
    'per_bohr_radius': r'1/a_0',
})

# astronomy
light_year = speed_of_light * year
parsec = 3.085_678e18 * m

UNIT_NAME_TO_VALUE.update({
    'light_year': light_year,
    'light_years': light_year,
    'ly': light_year,
    'parsec': parsec,
    'parsecs': parsec,
    'pc': parsec,
})
