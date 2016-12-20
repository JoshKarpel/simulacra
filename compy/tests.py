import os
import sys
import unittest
import shutil

from . import core, utils, math, cluster

TEST_DIR = os.path.join(os.getcwd(), 'temp__unit_testing')


class TestEnsureDirExists(unittest.TestCase):
    def setUp(self):
        self.dirname = 'foo'
        self.filename = 'foo/bar.py'
        self.target_name = os.path.join(TEST_DIR, 'foo')

    def test_ensure_dir_from_dirname(self):
        utils.ensure_dir_exists(os.path.join(TEST_DIR, self.dirname))
        self.assertTrue(os.path.exists(self.target_name))
        os.rmdir(self.target_name)  # necessary cleanup to run other tests in this case

    def test_ensure_dir_from_filename(self):
        utils.ensure_dir_exists(os.path.join(TEST_DIR, self.filename))
        self.assertTrue(os.path.exists(self.target_name))
        self.assertFalse(os.path.exists(os.path.join(self.target_name, 'kappa')))  # didn't accidentally create a path with the name of the file
        os.rmdir(self.target_name)  # necessary cleanup to run other tests in this TestCase

    def tearDown(self):
        shutil.rmtree(TEST_DIR)


class TestBeet(unittest.TestCase):
    def setUp(self):
        self.obj = utils.Beet('foo')
        self.obj_name = 'foo'
        self.target_name = 'foo.beet'
        utils.ensure_dir_exists(TEST_DIR)

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_beet_names(self):
        self.assertEqual(self.obj.name, self.obj_name)
        self.assertEqual(self.obj.file_name, self.obj_name)

    def test_save_load(self):
        path = self.obj.save(target_dir = TEST_DIR)
        self.assertEqual(path, os.path.join(TEST_DIR, self.target_name))  # test if path was constructed correctly
        self.assertTrue(os.path.exists(path))  # path should actually exist on the system
        loaded = utils.Beet.load(path)
        self.assertEqual(loaded, self.obj)  # beets should be equal, but NOT the same object
        self.assertEqual(loaded.uid, self.obj.uid)  # beets should have the same uid
        self.assertEqual(hash(loaded), hash(self.obj))  # beets should have the same hash
        self.assertIsNot(loaded, self.obj)  # beets should NOT be the same object


class TestSpecification(TestBeet):
    def setUp(self):
        self.obj = core.Specification('bar')
        self.obj_name = 'bar'
        self.target_name = 'bar.spec'
        utils.ensure_dir_exists(TEST_DIR)


class TestSimulation(TestBeet):
    def setUp(self):
        self.obj = core.Simulation(core.Specification('baz'))
        self.obj_name = 'baz'
        self.target_name = 'baz.sim'
        utils.ensure_dir_exists(TEST_DIR)


class TestFibonnaci(unittest.TestCase):
    def test_fibonnaci_value(self):
        self.assertEqual(math.fibonacci(99), 218922995834555169026)

    def test_fibonnaci_exception(self):
        with self.assertRaises(TypeError):
            math.fibonacci('foo')


class TestPrimeFinders(unittest.TestCase):
    def test_is_prime(self):
        for prime in [2, 3, 5, 7, 11, 17]:
            with self.subTest(p = prime):
                self.assertTrue(math.is_prime(prime))

        for not_prime in [1, 4, 6, 15]:
            with self.subTest(np = not_prime):
                self.assertFalse(math.is_prime(not_prime))
