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

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_ensure_dir_from_dirname(self):
        utils.ensure_dir_exists(os.path.join(TEST_DIR, self.dirname))
        self.assertTrue(os.path.exists(self.target_name))

    def test_ensure_dir_from_filename(self):
        utils.ensure_dir_exists(os.path.join(TEST_DIR, self.filename))
        self.assertTrue(os.path.exists(self.target_name))
        self.assertFalse(os.path.exists(os.path.join(self.target_name, 'kappa')))  # didn't accidentally create a path with the name of the file


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


class TestSumming(unittest.TestCase):
    def setUp(self):
        self.summand_one = utils.Summand()
        self.summand_two = utils.Summand()
        self.sum = self.summand_one + self.summand_two

    def test_is(self):
        self.assertFalse(self.summand_one is self.summand_two)

    def test_equality(self):
        self.assertFalse(self.summand_one == self.summand_two)

    def test_instance_of(self):
        self.assertTrue(isinstance(self.summand_one, utils.Summand))
        self.assertTrue(isinstance(self.summand_two, utils.Summand))
        self.assertTrue(isinstance(self.sum, utils.Summand))
        self.assertTrue(isinstance(self.sum, utils.Sum))

        self.assertFalse(isinstance(self.summand_one, utils.Sum))
        self.assertFalse(isinstance(self.summand_two, utils.Sum))

    def test_container(self):
        self.assertTrue(self.summand_one in self.sum.summands)
        self.assertTrue(self.summand_two in self.sum.summands)

        self.assertTrue(self.summand_one in self.sum)
        self.assertTrue(self.summand_two in self.sum)


class TestSummingSubclassing(unittest.TestCase):
    def setUp(self):
        class Fruit(utils.Summand):
            def __init__(self):
                super().__init__()
                self.summation_class = FruitBasket

        class FruitBasket(utils.Sum, Fruit):
            container_name = 'basket'

        class Apple(Fruit):
            pass

        class Banana(Fruit):
            pass

        self.Fruit = Fruit
        self.FruitBasket = FruitBasket
        self.Apple = Apple
        self.Banana = Banana

        self.apple = self.Apple()
        self.banana = self.Banana()
        self.fruit_basket = self.apple + self.banana

    def test_instance_of_bases(self):
        self.assertTrue(isinstance(self.apple, utils.Summand))
        self.assertTrue(isinstance(self.banana, utils.Summand))
        self.assertTrue(isinstance(self.fruit_basket, utils.Summand))
        self.assertTrue(isinstance(self.fruit_basket, utils.Sum))

        self.assertFalse(isinstance(self.apple, utils.Sum))
        self.assertFalse(isinstance(self.banana, utils.Sum))

    def test_instance_of_subclasses(self):
        self.assertTrue(isinstance(self.fruit_basket, self.Fruit))
        self.assertTrue(isinstance(self.fruit_basket, self.FruitBasket))

        self.assertFalse(isinstance(self.fruit_basket, self.Apple))
        self.assertFalse(isinstance(self.fruit_basket, self.Banana))

    def test_container(self):
        self.assertTrue(self.apple in self.fruit_basket.basket)
        self.assertTrue(self.banana in self.fruit_basket.basket)

        self.assertTrue(self.apple in self.fruit_basket)
        self.assertTrue(self.banana in self.fruit_basket)


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


class TestRestrictedValues(unittest.TestCase):
    legal = ('a', 5, (4, 5, 6))
    illegal = ('foo', 3, (1, 2, 3))
    attr = utils.RestrictedValues('attr', legal)

    def test_legal_assignments(self):
        for x in self.legal:
            with self.subTest(x = x):
                self.attr = x

    def test_illegal_assignments(self):
        for x in self.illegal:
            with self.subTest(x = x):
                with self.assertRaises(ValueError):
                    self.attr = x
