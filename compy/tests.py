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
        self.assertFalse(os.path.exists(os.path.join(self.target_name, 'bar')))  # didn't accidentally create a path with the name of the file
        os.rmdir(self.target_name)  # necessary cleanup to run other tests in this TestCase

    def tearDown(self):
        shutil.rmtree(TEST_DIR)


class TestBeet(unittest.TestCase):
    def setUp(self):
        self.beet = utils.Beet('foo')
        utils.ensure_dir_exists(TEST_DIR)

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_beet_names(self):
        self.assertEqual(self.beet.name, 'foo')
        self.assertEqual(self.beet.file_name, 'foo')

    def test_save_load(self):
        path = self.beet.save(target_dir = TEST_DIR)
        self.assertEqual(path, os.path.join(TEST_DIR, 'foo.beet'))  # test if path was constructed correctly
        self.assertTrue(os.path.exists(path))  # path should actually exist on the system
        loaded = utils.Beet.load(path)
        self.assertEqual(loaded, self.beet)  # beets should be equal, but NOT the same object
        self.assertEqual(loaded.uid, self.beet.uid)  # beets should have the same uid
        self.assertEqual(hash(loaded), hash(self.beet))  # beets should have the same hash
        self.assertIsNot(loaded, self.beet)  # beets should NOT be the same object


class TestSpecification(unittest.TestCase):
    pass


class TestSimulation(unittest.TestCase):
    pass


class TestFibonnaci(unittest.TestCase):
    def test_fibonnaci_value(self):
        self.assertEqual(math.fibonacci(99), 218922995834555169026)

    def test_fibonnaci_expception(self):
        with self.assertRaises(ValueError):
            math.fibonacci('foo')
