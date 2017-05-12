import os

import numpy as np

import simulacra as si
import ionization as ion
from . import tests_simulacra


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DIR = os.path.join(THIS_DIR, 'temp-unit-testing')


class TestElectricFieldSpecification:
    spec_type = ion.ElectricFieldSpecification

    def setUp(self):
        self.obj = self.spec_type('bar')
        self.obj_name = 'bar'
        self.target_name = 'bar.spec'
        si.utils.ensure_dir_exists(TEST_DIR)


class TestLineSpecification(TestElectricFieldSpecification, tests_simulacra.TestBeet):
    spec_type = ion.LineSpecification


class TestCylindricalSliceSpecification(TestElectricFieldSpecification, tests_simulacra.TestBeet):
    spec_type = ion.CylindricalSliceSpecification


class TestSphericalSliceSpecification(TestElectricFieldSpecification, tests_simulacra.TestBeet):
    spec_type = ion.SphericalSliceSpecification


class TestSphericalHarmonicSpecification(TestElectricFieldSpecification, tests_simulacra.TestBeet):
    spec_type = ion.SphericalHarmonicSpecification


class TestElectricFieldSimulation:
    spec_type = ion.ElectricFieldSpecification

    def setUp(self):
        self.obj = self.spec_type('baz').to_simulation()
        self.obj_name = 'baz'
        self.target_name = 'baz.sim'
        si.utils.ensure_dir_exists(TEST_DIR)

    def test_save_load__save_mesh(self):
        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = True)
        loaded = ion.ElectricFieldSimulation.load(path, initialize_mesh = False)

        self.assertEqual(loaded.mesh, pre_save_mesh)  # the new mesh should be equal
        self.assertIsNot(loaded.mesh, pre_save_mesh)  # the new mesh will not be the same object, because it it gets swapped with a copy during the save

    def test_save_load__no_save_mesh(self):
        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = False)
        loaded = ion.ElectricFieldSimulation.load(path, initialize_mesh = False)

        self.assertNotEqual(loaded.mesh, pre_save_mesh)  # pre and post should not be equal
        self.assertIsNone(loaded.mesh)  # in fact, the loaded mesh shouldn't even exist

    def test_save_load__save_mesh__reinitialize(self):
        self.obj.mesh.g_mesh = np.ones(1000)  # replace g_mesh with dummy entry

        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = True)
        loaded = ion.ElectricFieldSimulation.load(path, initialize_mesh = True)

        self.assertIsNotNone(loaded.mesh)
        self.assertNotEqual(loaded.mesh, pre_save_mesh)  # even though we saved the mesh, we reinitialized, so it shouldn't have the same entries
        self.assertIsNot(loaded.mesh, pre_save_mesh)  # the new mesh will not be the same object, because it it gets swapped with a copy during the save

    def test_save_load__no_save_mesh__reinitialize(self):
        self.obj.mesh.g_mesh = np.ones(1000)  # replace g_mesh with dummy entry

        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = False)
        loaded = ion.ElectricFieldSimulation.load(path, initialize_mesh = True)

        self.assertIsNotNone(loaded.mesh)
        self.assertNotEqual(loaded.mesh, pre_save_mesh)  # pre and post should not be equal

    def test_initial_norm(self):
        self.assertAlmostEqual(self.obj.mesh.norm(), 1)

    def test_initial_state_overlap(self):
        ip = self.obj.mesh.inner_product(self.obj.spec.initial_state)
        self.assertAlmostEqual(np.abs(ip) ** 2, 1)


class TestLineSimulation(TestElectricFieldSimulation, tests_simulacra.TestBeet):
    spec_type = ion.LineSpecification


class TestCylindricalSliceSimulation(TestElectricFieldSimulation, tests_simulacra.TestBeet):
    spec_type = ion.CylindricalSliceSpecification


class TestSphericalSliceSimulation(TestElectricFieldSimulation, tests_simulacra.TestBeet):
    spec_type = ion.SphericalSliceSpecification


class TestSphericalHarmonicSimulation(TestElectricFieldSimulation, tests_simulacra.TestBeet):
    spec_type = ion.SphericalHarmonicSpecification
