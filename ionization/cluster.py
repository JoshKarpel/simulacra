import logging

import compy as cp
from compy.cluster import *
from compy.units import *
from ionization import core

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ask_mesh_type():
    """
    :return: spec_type, mesh_kwargs
    """
    mesh_kwargs = {}

    mesh_type = cp.cluster.ask_for_input('Mesh Type (cyl | sph | harm)', default = 'harm', cast_to = str)

    try:
        if mesh_type == 'cyl':
            spec_type = core.CylindricalSliceSpecification

            mesh_kwargs['z_bound'] = bohr_radius * cp.cluster.ask_for_input('Z Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['rho_bound'] = bohr_radius * cp.cluster.ask_for_input('Rho Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['z_points'] = 2 * (mesh_kwargs['z_bound'] / bohr_radius) * cp.cluster.ask_for_input('Z Points per Bohr Radii', default = 20, cast_to = int)
            mesh_kwargs['rho_points'] = (mesh_kwargs['rho_bound'] / bohr_radius) * cp.cluster.ask_for_input('Rho Points per Bohr Radii', default = 20, cast_to = int)

            mesh_kwargs['outer_radius'] = max(mesh_kwargs['z_bound'], mesh_kwargs['rho_bound'])

            memory_estimate = (128 / 8) * mesh_kwargs['z_points'] * mesh_kwargs['rho_points']

        elif mesh_type == 'sph':
            spec_type = core.SphericalSliceSpecification

            mesh_kwargs['r_bound'] = bohr_radius * cp.cluster.ask_for_input('R Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['r_points'] = (mesh_kwargs['r_bound'] / bohr_radius) * cp.cluster.ask_for_input('R Points per Bohr Radii', default = 40, cast_to = int)
            mesh_kwargs['theta_points'] = cp.cluster.ask_for_input('Theta Points', default = 100, cast_to = int)

            mesh_kwargs['outer_radius'] = mesh_kwargs['r_bound']

            memory_estimate = (128 / 8) * mesh_kwargs['r_points'] * mesh_kwargs['theta_points']

        elif mesh_type == 'harm':
            spec_type = core.SphericalHarmonicSpecification

            mesh_kwargs['r_bound'] = bohr_radius * cp.cluster.ask_for_input('R Bound (Bohr radii)', default = 250, cast_to = float)
            mesh_kwargs['r_points'] = (mesh_kwargs['r_bound'] / bohr_radius) * cp.cluster.ask_for_input('R Points per Bohr Radii', default = 4, cast_to = int)
            mesh_kwargs['l_points'] = cp.cluster.ask_for_input('l points', default = 100, cast_to = int)

            mesh_kwargs['outer_radius'] = mesh_kwargs['r_bound']

            memory_estimate = (128 / 8) * mesh_kwargs['r_points'] * mesh_kwargs['l_points']

        else:
            raise ValueError('Mesh type {} not found!'.format(mesh_type))

        logger.warning('Predicted memory usage per Simulation is >{}'.format(utils.convert_bytes(memory_estimate)))

        return spec_type, mesh_kwargs
    except ValueError:
        ask_mesh_type()


class ElectricFieldJobProcessor(cp.cluster.JobProcessor):
    def __init__(self, job_name, job_dir_path):
        super(ElectricFieldJobProcessor, self).__init__(job_name, job_dir_path, core.ElectricFieldSimulation)

    def collect_data_from_sim(self, sim_name, sim):
        super(ElectricFieldJobProcessor, self).collect_data_from_sim(sim_name, sim)

        self.data[sim_name].update({
            'electric_potential': sim.spec.electric_potential,
            'final_norm': sim.norm_vs_time[-1],
            'final_inner_products': sim.inner_products_vs_time[-1],
            'final_state_overlaps': sim.state_overlaps_vs_time[-1]
        })
