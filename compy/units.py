"""Import this as get access to SI units and fundamental constants, with values from CODATA 2014."""

import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# dimensionless constants
alpha = 7.2973525664e-3
pi = np.pi
twopi = 2 * np.pi
e = np.e

# base units
m = 1
s = 1
kg = 1
A = 1
K = 1

# distance
cm = 1e-2 * m
mm = 1e-3 * m
um = 1e-6 * m
nm = 1e-9 * m
angstrom = 1e-10 * m
pm = 1e-12 * m
fm = 1e-15 * m
bohr_radius = 5.2917721067e-11 * m

# time
ms = 1e-3 * s
us = 1e-6 * s
ns = 1e-9 * s
ps = 1e-12 * s
fs = 1e-15 * s
minute = 60 * s
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365 * day

# mass
g = 1e-3 * kg
mg = 1e-3 * g
ug = 1e-6 * g
ng = 1e-9 * g
pg = 1e-12 * g
fg = 1e-15 * g
proton_mass = 1.672621898e-27 * kg
electron_mass = 9.10938356e-31 * kg
electron_mass_reduced = proton_mass * electron_mass / (proton_mass + electron_mass)

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

# energies in electron-volts
eV = proton_charge * V
keV = 1e3 * eV
MeV = 1e6 * eV
GeV = 1e9 * eV
TeV = 1e12 * eV
PeV = 1e15 * eV

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

# quantum mechanics
h = 6.626070040e-34 * J * s  # Planck's constant
hbar = h / (2 * pi)
rydberg = 13.605693009 * eV
hartree = 27.21138602 * eV

# speed of light and EM
c = 299792458 * m / s
mu_0 = pi * 4e-7 * N / (A ** 2)
epsilon_0 = 1 / (mu_0 * (c ** 2))
coulomb_force_constant = 1 / (4 * pi * epsilon_0)
n_vacuum = 1

logger.debug('Units initialized')
