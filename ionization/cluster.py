import logging

import compy as cp
from compy.units import *
from ionization import core

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ask_mesh_type():
    mesh_specification = {}

    mesh_type = cp.cluster.ask_for_input('Mesh Type (cyl | sph | harm)', default = 'cyl', cast_to = str)

    if mesh_type == 'cyl':
        param_type = core.CylindricalSliceSpecification

        mesh_specification['z_bound'] = bohr_radius * cp.cluster.ask_for_input('Z Bound (Bohr radii)', default = 30, cast_to = float)
        mesh_specification['rho_bound'] = bohr_radius * cp.cluster.ask_for_input('Rho Bound (Bohr radii)', default = 30, cast_to = float)
        mesh_specification['z_points'] = 2 * (mesh_specification['z_bound'] / bohr_radius) * cp.cluster.ask_for_input('Z Points per Bohr Radii', default = 20, cast_to = int)
        mesh_specification['rho_points'] = (mesh_specification['rho_bound'] / bohr_radius) * cp.cluster.ask_for_input('Rho Points per Bohr Radii', default = 20, cast_to = int)

    elif mesh_type == 'sph':
        param_type = core.SphericalSliceSpecification

        mesh_specification['r_bound'] = cp.cluster.ask_for_input('R Bound (Bohr radii)', default = 30, cast_to = float)
        mesh_specification['r_points'] = (mesh_specification['r_bound'] / bohr_radius) * ask_for_input('R Points per Bohr Radii', default = 40, cast_to = int)
        mesh_specification['theta_points'] = cp.cluster.ask_for_input('Theta Points', default = 500, cast_to = int)

    elif mesh_type == 'harm':
        param_type = core.SphericalHarmonicSpecification

        mesh_specification['r_bound'] = cp.cluster.ask_for_input('R Bound (Bohr radii)', default = 30, cast_to = float)
        mesh_specification['r_points'] = (mesh_specification['r_bound'] / bohr_radius) * ask_for_input('R Points per Bohr Radii', default = 40, cast_to = int)
        mesh_specification['spherical_harmonics'] = cp.cluster.ask_for_input('Spherical Harmonics', default = 500, cast_to = int)

    else:
        exception_text = 'Mesh type {} not found!'.format(mesh_type)
        raise Exception(exception_text)

    return param_type, mesh_specification


class IonizationJobProcessor(cp.core.JobProcessor):
    raise NotImplementedError
