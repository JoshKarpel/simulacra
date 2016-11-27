import os
import sys
import unittest
import shutil

from . import core

from .core import *
from .utils import *
from .cluster import *
from .math import *

TEST_DIR = os.path.join(os.getcwd(), 'temp__unit_testing')
print(TEST_DIR)


class TestEnsureDirExists(unittest.TestCase):
    def setUp(self):
        self.dirname = 'foo'
        self.filename = 'foo/bar.py'
        self.target_name = os.path.join(TEST_DIR, 'foo')

    def test_ensure_dir_from_dirname(self):
        ensure_dir_exists(os.path.join(TEST_DIR, self.dirname))
        self.assertTrue(os.path.exists(self.target_name))
        os.rmdir(self.target_name)  # necessary cleanup to run other tests in this case

    def test_ensure_dir_from_filename(self):
        ensure_dir_exists(os.path.join(TEST_DIR, self.filename))
        self.assertTrue(os.path.exists(self.target_name))
        self.assertFalse(os.path.exists(os.path.join(self.target_name, 'bar')))
        os.rmdir(self.target_name)  # necessary cleanup to run other tests in this case

    def tearDown(self):
        shutil.rmtree(TEST_DIR)


class TestBeet(unittest.TestCase):
    def setUp(self):
        self.beet = Beet('foo')
        ensure_dir_exists(TEST_DIR)

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_beet_names(self):
        self.assertEqual(self.beet.name, 'foo')
        self.assertEqual(self.beet.file_name, 'foo')

    def test_save_load(self):
        path = self.beet.save(target_dir = TEST_DIR)
        self.assertEqual(path, os.path.join(TEST_DIR, 'foo.beet'))
        loaded = Beet.load(path)
        self.assertEqual(loaded, self.beet)  # beets should be equal, but NOT the same object
        self.assertEqual(hash(loaded), hash(self.beet))  # beets should be equal, but NOT the same object
        self.assertIsNot(loaded, self.beet)
