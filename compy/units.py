"""Import this to get access to SI units and fundamental constants, with values from CODATA 2014."""

import numpy as _np

unit_names_to_values = {}
unit_names_to_tex_strings = {}


def uround(value, units = 1, digits = 3):
    if type(units) == str:
        units = unit_names_to_values[units]

    return _np.around(value / units, digits)


# dimensionless constants
alpha = 7.2973525664e-3
pi = _np.pi
twopi = 2 * _np.pi
e = _np.e

unit_names_to_values.update({'alpha': alpha,
                             'pi': pi,
                             'twopi': twopi,
                             'e': e})
unit_names_to_tex_strings.update({'alpha': r'$\alpha$',
                                  'pi': '$\pi$',
                                  'twopi': r'$2\pi$',
                                  'e': '$e$'})

# base units
m = 1
s = 1
kg = 1
A = 1
K = 1
rad = 1
deg = rad * 180 / pi

unit_names_to_values.update({'m': m,
                             's': s,
                             'kg': kg,
                             'A': A,
                             'K': K})
unit_names_to_tex_strings.update({'m': r'$\mathrm{m}$',
                                  's': r'$\mathrm{s}$',
                                  'kg': r'$\mathrm{kg}$',
                                  'A': r'$\mathrm{A}$',
                                  'K': r'$\mathrm{K}$'})

# distance
cm = 1e-2 * m
mm = 1e-3 * m
um = 1e-6 * m
nm = 1e-9 * m
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

unit_names_to_values.update({'cm': cm,
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
                             'inch': inch})
unit_names_to_tex_strings.update({'cm': r'$\mathrm{cm}$',
                                  'cetimeter': r'$\mathrm{cm}$',
                                  'cetimeters': r'$\mathrm{cm}$',
                                  'mm': r'$\mathrm{mm}$',
                                  'millimeter': r'$\mathrm{mm}$',
                                  'millimeters': r'$\mathrm{mm}$',
                                  'um': r'$\mathrm{um}$',
                                  'micrometer': r'$\mathrm{um}$',
                                  'micrometers': r'$\mathrm{um}$',
                                  'micron': r'$\mathrm{um}$',
                                  'microns': r'$\mathrm{um}$',
                                  'nm': r'$\mathrm{nm}$',
                                  'nanometer': r'$\mathrm{nm}$',
                                  'nanometers': r'$\mathrm{nm}$',
                                  'pm': r'$\mathrm{pm}$',
                                  'picometer': r'$\mathrm{pm}$',
                                  'picometers': r'$\mathrm{pm}$',
                                  'fm': r'$\mathrm{fm}$',
                                  'femtometer': r'$\mathrm{fm}$',
                                  'femtometers': r'$\mathrm{fm}$',
                                  'km': r'$\mathrm{km}$',
                                  'kilometer': r'$\mathrm{km}$',
                                  'kilometers': r'$\mathrm{km}$',
                                  'Mm': r'$\mathrm{Mm}$',
                                  'Gm': r'$\mathrm{Gm}$',
                                  'Tm': r'$\mathrm{Tm}$',
                                  'Pm': r'$\mathrm{Pm}$',
                                  'angstrom': r'$\mathrm{\AA}$',
                                  'bohr_radius': r'$a_0$',
                                  'inch': r'$\mathrm{in}$'})

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

unit_names_to_values.update({'ms': msec,
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
                             'years': year})
unit_names_to_tex_strings.update({'ms': r'$\mathrm{ms}$',
                                  'msec': r'$\mathrm{ms}$',
                                  'us': r'$\mathrm{us}$',
                                  'usec': r'$\mathrm{us}$',
                                  'ns': r'$\mathrm{ns}$',
                                  'nsec': r'$\mathrm{ns}$',
                                  'ps': r'$\mathrm{ps}$',
                                  'psec': r'$\mathrm{ps}$',
                                  'fs': r'$\mathrm{fs}$',
                                  'fsec': r'$\mathrm{fs}$',
                                  'as': r'$\mathrm{as}$',
                                  'asec': r'$\mathrm{as}$',
                                  'minute': '$\mathrm{minutes}$',
                                  'minutes': '$\mathrm{minutes}$',
                                  'hour': '$\mathrm{hours}$',
                                  'hours': '$\mathrm{hours}$',
                                  'day': '$\mathrm{days}$',
                                  'days': '$\mathrm{days}$',
                                  'week': '$\mathrm{weeks}$',
                                  'weeks': '$\mathrm{weeks}$',
                                  'year': '$\mathrm{years}$',
                                  'years': '$\mathrm{years}$'})

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

unit_names_to_values.update({'g': g,
                             'mg': mg,
                             'ug': ug,
                             'ng': ng,
                             'pg': pg,
                             'fg': fg,
                             'proton_mass': proton_mass,
                             'neutron_mass': neutron_mass,
                             'electron_mass': electron_mass,
                             'electron_mass_reduced': electron_mass_reduced})
unit_names_to_tex_strings.update({'g': r'$\mathrm{g}$',
                                  'mg': r'$\mathrm{mg}$',
                                  'ug': r'$\mathrm{ug}$',
                                  'ng': r'$\mathrm{ng}$',
                                  'pg': r'$\mathrm{pg}$',
                                  'fg': r'$\mathrm{fg}$',
                                  'proton_mass': r'$m_p$',
                                  'neutron_mass': r'$m_n$',
                                  'electron_mass': r'$m_e$',
                                  'electron_mass_reduced': r'$\mu_e$'})

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

unit_names_to_values.update({'Hz': Hz,
                             'mHz': mHz,
                             'uHz': uHz,
                             'nHz': nHz,
                             'pHz': pHz,
                             'fHz': fHz,
                             'kHz': kHz,
                             'MHz': MHz,
                             'GHz': GHz,
                             'THz': THz,
                             'PHz': PHz})
unit_names_to_tex_strings.update({'Hz': r'$\mathrm{Hz}$',
                                  'mHz': r'$\mathrm{mHz}$',
                                  'uHz': r'$\mathrm{uHz}$',
                                  'nHz': r'$\mathrm{nHz}$',
                                  'pHz': r'$\mathrm{pHz}$',
                                  'fHz': r'$\mathrm{fHz}$',
                                  'kHz': r'$\mathrm{kHz}$',
                                  'MHz': r'$\mathrm{MHz}$',
                                  'GHz': r'$\mathrm{GHz}$',
                                  'THz': r'$\mathrm{THz}$',
                                  'PHz': r'$\mathrm{PHz}$'})

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

unit_names_to_values.update({'C': C,
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
                             'electron_charge': electron_charge})
unit_names_to_tex_strings.update({'C': r'$\mathrm{C}$',
                                  'mC': r'$\mathrm{mC}$',
                                  'uC': r'$\mathrm{uC}$',
                                  'nC': r'$\mathrm{nC}$',
                                  'pC': r'$\mathrm{pC}$',
                                  'fC': r'$\mathrm{fC}$',
                                  'kC': r'$\mathrm{kC}$',
                                  'MC': r'$\mathrm{MC}$',
                                  'GC': r'$\mathrm{GC}$',
                                  'TC': r'$\mathrm{TC}$',
                                  'PC': r'$\mathrm{PC}$',
                                  'proton_charge': r'$e$',
                                  'electron_charge': r'$-e$'})

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

unit_names_to_values.update({
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
unit_names_to_tex_strings.update({
    'J': r'$\mathrm{J}$',
    'mJ': r'$\mathrm{mJ}$',
    'uJ': r'$\mathrm{uJ}$',
    'nJ': r'$\mathrm{nJ}$',
    'pJ': r'$\mathrm{pJ}$',
    'fJ': r'$\mathrm{fJ}$',
    'kJ': r'$\mathrm{kJ}$',
    'MJ': r'$\mathrm{MJ}$',
    'GJ': r'$\mathrm{GJ}$',
    'TJ': r'$\mathrm{TJ}$',
    'PJ': r'$\mathrm{PJ}$',
    'J/cm^2': r'$\mathrm{J/cm^2}$',
    'Jcm2': r'$\mathrm{J/cm^2}$'
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

unit_names_to_values.update({'V': V,
                             'mV': mV,
                             'uV': uV,
                             'nV': nV,
                             'pV': pV,
                             'fV': fV,
                             'kV': kV,
                             'MV': MV,
                             'GV': GV,
                             'TV': TV,
                             'PV': PV})
unit_names_to_tex_strings.update({'V': r'$\mathrm{V}$',
                                  'mV': r'$\mathrm{mV}$',
                                  'uV': r'$\mathrm{uV}$',
                                  'nV': r'$\mathrm{nV}$',
                                  'pV': r'$\mathrm{pV}$',
                                  'fV': r'$\mathrm{fV}$',
                                  'kV': r'$\mathrm{kV}$',
                                  'MV': r'$\mathrm{MV}$',
                                  'GV': r'$\mathrm{GV}$',
                                  'TV': r'$\mathrm{TV}$',
                                  'PV': r'$\mathrm{PV}$'})

# energies in electron-volts
eV = proton_charge * V
keV = 1e3 * eV
MeV = 1e6 * eV
GeV = 1e9 * eV
TeV = 1e12 * eV
PeV = 1e15 * eV

unit_names_to_values.update({'eV': eV,
                             'keV': keV,
                             'MeV': MeV,
                             'GeV': GeV,
                             'TeV': TeV,
                             'PeV': PeV})
unit_names_to_tex_strings.update({'eV': r'$\mathrm{eV}$',
                                  'keV': r'$\mathrm{keV}$',
                                  'MeV': r'$\mathrm{MeV}$',
                                  'GeV': r'$\mathrm{GeV}$',
                                  'TeV': r'$\mathrm{TeV}$',
                                  'PeV': r'$\mathrm{PeV}$'})

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

unit_names_to_values.update({'N': N,
                             'mN': mN,
                             'uN': uN,
                             'nN': nN,
                             'pN': pN,
                             'fN': fN,
                             'kN': kN,
                             'MN': MN,
                             'GN': GN,
                             'TN': TN,
                             'PN': PN})
unit_names_to_tex_strings.update({'N': r'$\mathrm{N}$',
                                  'mN': r'$\mathrm{mN}$',
                                  'uN': r'$\mathrm{uN}$',
                                  'nN': r'$\mathrm{nN}$',
                                  'pN': r'$\mathrm{pN}$',
                                  'fN': r'$\mathrm{fN}$',
                                  'kN': r'$\mathrm{kN}$',
                                  'MN': r'$\mathrm{MN}$',
                                  'GN': r'$\mathrm{GN}$',
                                  'TN': r'$\mathrm{TN}$',
                                  'PN': r'$\mathrm{PN}$'})

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

unit_names_to_values.update({'W': W,
                             'mW': mW,
                             'uW': uW,
                             'nW': nW,
                             'pW': pW,
                             'fW': fW,
                             'kW': kW,
                             'MW': MW,
                             'GW': GW,
                             'TW': TW,
                             'PW': PW})
unit_names_to_tex_strings.update({'W': r'$\mathrm{W}$',
                                  'mW': r'$\mathrm{mW}$',
                                  'uW': r'$\mathrm{uW}$',
                                  'nW': r'$\mathrm{nW}$',
                                  'pW': r'$\mathrm{pW}$',
                                  'fW': r'$\mathrm{fW}$',
                                  'kW': r'$\mathrm{kW}$',
                                  'MW': r'$\mathrm{MW}$',
                                  'GW': r'$\mathrm{GW}$',
                                  'TW': r'$\mathrm{TW}$',
                                  'PW': r'$\mathrm{PW}$'})

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

unit_names_to_values.update({'A': A,
                             'mA': mA,
                             'uA': uA,
                             'nA': nA,
                             'pA': pA,
                             'fA': fA,
                             'kA': kA,
                             'MA': MA,
                             'GA': GA,
                             'TA': TA,
                             'PA': PA})
unit_names_to_tex_strings.update({'A': r'$\mathrm{A}$',
                                  'mA': r'$\mathrm{mA}$',
                                  'uA': r'$\mathrm{uA}$',
                                  'nA': r'$\mathrm{nA}$',
                                  'pA': r'$\mathrm{pA}$',
                                  'fA': r'$\mathrm{fA}$',
                                  'kA': r'$\mathrm{kA}$',
                                  'MA': r'$\mathrm{MA}$',
                                  'GA': r'$\mathrm{GA}$',
                                  'TA': r'$\mathrm{TA}$',
                                  'PA': r'$\mathrm{PA}$'})

# quantum mechanics
h = 6.626070040e-34 * J * s  # Planck's constant
hbar = h / twopi
rydberg = 13.605693009 * eV
hartree = 27.21138602 * eV

unit_names_to_values.update({'h': h,
                             'hbar': hbar,
                             'rydberg': rydberg,
                             'hartree': hartree})
unit_names_to_tex_strings.update({'h': r'$h$',
                                  'hbar': r'$\hbar$',
                                  'rydberg': r'$\mathrm{Ry}',
                                  'hartree': r'$\mathrm{Ha}'})

# speed of light and EM
c = 299792458 * m / s
mu_0 = pi * 4e-7 * N / (A ** 2)
epsilon_0 = 1 / (mu_0 * (c ** 2))
coulomb_force_constant = 1 / (4 * pi * epsilon_0)
n_vacuum = 1

unit_names_to_values.update({'c': c,
                             'mu_0': mu_0,
                             'epsilon_0': epsilon_0,
                             'coulomb_force_constant': coulomb_force_constant,
                             'n_vacuum': n_vacuum})
unit_names_to_tex_strings.update({'c': r'$c$',
                                  'mu_0': r'$\mu_0$',
                                  'epsilon_0': r'$\epsilon_0$',
                                  'coulomb_force_constant': r'$k_e$',
                                  'n_vacuum': r'$n_{\mathrm{vac}}$'})

atomic_energy = hartree
atomic_electric_field = coulomb_force_constant * proton_charge / (bohr_radius ** 2)
atomic_electric_potential = coulomb_force_constant * proton_charge / bohr_radius
atomic_electric_dipole = proton_charge * bohr_radius  # TODO: check whether + or -
atomic_velocity = alpha * c
atomic_momentum = electron_mass * atomic_velocity
atomic_time = hbar / hartree

unit_names_to_values.update({'atomic_electric_field': atomic_electric_field,
                             'AEF': atomic_electric_field,
                             'atomic_electric_potential': atomic_electric_potential,
                             'atomic_electric_dipole': atomic_electric_dipole,
                             'atomic_velocity': atomic_velocity,
                             'atomic_momentum': atomic_momentum})
unit_names_to_tex_strings.update({'atomic_electric_field': r'a.u.',  # TODO: add these
                                  'atomic_electric_potential': r'a.u.',
                                  'atomic_electric_dipole': r'$e \, a_0$',
                                  'atomic_velocity': r'a.u.',
                                  'atomic_momentum': r'a.u.'})
