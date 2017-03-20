import os
import sys
import unittest
import shutil

import numpy as np

import compy as cp
from . import core, cluster, potentials, states, integrodiff

TEST_DIR = os.path.join(os.getcwd(), 'temp__unit_testing')


class TestElectricFieldSpecification:
    spec_type = core.ElectricFieldSpecification

    def setUp(self):
        self.obj = self.spec_type('bar')
        self.obj_name = 'bar'
        self.target_name = 'bar.spec'
        cp.utils.ensure_dir_exists(TEST_DIR)


class TestLineSpecification(TestElectricFieldSpecification, cp.tests.TestBeet):
    spec_type = core.LineSpecification


class TestCylindricalSliceSpecification(TestElectricFieldSpecification, cp.tests.TestBeet):
    spec_type = core.CylindricalSliceSpecification


class TestSphericalSliceSpecification(TestElectricFieldSpecification, cp.tests.TestBeet):
    spec_type = core.SphericalSliceSpecification


class TestSphericalHarmonicSpecification(TestElectricFieldSpecification, cp.tests.TestBeet):
    spec_type = core.SphericalHarmonicSpecification


class TestElectricFieldSimulation:
    spec_type = core.ElectricFieldSpecification

    def setUp(self):
        self.obj = self.spec_type('baz').to_simulation()
        self.obj_name = 'baz'
        self.target_name = 'baz.sim'
        cp.utils.ensure_dir_exists(TEST_DIR)

    def test_save_load__save_mesh(self):
        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = True)
        loaded = core.ElectricFieldSimulation.load(path, initialize_mesh = False)

        self.assertEqual(loaded.mesh, pre_save_mesh)  # the new mesh should be equal
        self.assertIsNot(loaded.mesh, pre_save_mesh)  # the new mesh will not be the same object, because it it gets swapped with a copy during the save

    def test_save_load__no_save_mesh(self):
        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = False)
        loaded = core.ElectricFieldSimulation.load(path, initialize_mesh = False)

        self.assertNotEqual(loaded.mesh, pre_save_mesh)  # pre and post should not be equal
        self.assertIsNone(loaded.mesh)  # in fact, the loaded mesh shouldn't even exist

    def test_save_load__save_mesh__reinitialize(self):
        self.obj.mesh.g_mesh = np.ones(1000)  # replace g_mesh with dummy entry

        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = True)
        loaded = core.ElectricFieldSimulation.load(path, initialize_mesh = True)

        self.assertIsNotNone(loaded.mesh)
        self.assertNotEqual(loaded.mesh, pre_save_mesh)  # even though we saved the mesh, we reinitialized, so it shouldn't have the same entries
        self.assertIsNot(loaded.mesh, pre_save_mesh)  # the new mesh will not be the same object, because it it gets swapped with a copy during the save

    def test_save_load__no_save_mesh__reinitialize(self):
        self.obj.mesh.g_mesh = np.ones(1000)  # replace g_mesh with dummy entry

        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = False)
        loaded = core.ElectricFieldSimulation.load(path, initialize_mesh = True)

        self.assertIsNotNone(loaded.mesh)
        self.assertNotEqual(loaded.mesh, pre_save_mesh)  # pre and post should not be equal


class TestLineSimulation(TestElectricFieldSimulation, cp.tests.TestBeet):
    spec_type = core.LineSpecification


class TestCylindricalSliceSimulation(TestElectricFieldSimulation, cp.tests.TestBeet):
    spec_type = core.CylindricalSliceSpecification


class TestSphericalSliceSimulation(TestElectricFieldSimulation, cp.tests.TestBeet):
    spec_type = core.SphericalSliceSpecification


class TestSphericalHarmonicSimulation(TestElectricFieldSimulation, cp.tests.TestBeet):
    spec_type = core.SphericalHarmonicSpecification
