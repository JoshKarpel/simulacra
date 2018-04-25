import os
import unittest
import shutil

import simulacra as si

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DIR = os.path.join(THIS_DIR, 'temp-unit-testing')


class TestBeet(unittest.TestCase):
    def setUp(self):
        self.obj = si.Beet('foo')
        self.obj_name = 'foo'
        self.target_name = 'foo.beet'
        si.utils.ensure_parents_exist(TEST_DIR)

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_beet_names(self):
        self.assertEqual(self.obj.name, self.obj_name)
        self.assertEqual(self.obj.file_name, self.obj_name)

    def test_save_load(self):
        path = self.obj.save(target_dir = TEST_DIR)
        self.assertEqual(path, os.path.join(TEST_DIR, self.target_name))  # test if path was constructed correctly
        self.assertTrue(os.path.exists(path))  # path should actually exist on the system
        loaded = si.Beet.load(path)
        self.assertEqual(loaded, self.obj)  # beets should be equal, but NOT the same object
        self.assertEqual(loaded.uuid, self.obj.uuid)  # beets should have the same uid
        self.assertEqual(hash(loaded), hash(self.obj))  # beets should have the same hash
        self.assertIsNot(loaded, self.obj)  # beets should NOT be the same object


class TestSpecification(TestBeet):
    def setUp(self):
        self.obj = si.Specification('bar')
        self.obj_name = 'bar'
        self.target_name = 'bar.spec'
        si.utils.ensure_parents_exist(TEST_DIR)


class TestSimulation(TestBeet):
    def setUp(self):
        self.obj = si.Simulation(si.Specification('baz'))
        self.obj_name = 'baz'
        self.target_name = 'baz.sim'
        si.utils.ensure_parents_exist(TEST_DIR)

    def testStatus(self):
        passes = (si.Status.INITIALIZED, si.Status.RUNNING, si.Status.RUNNING, si.Status.FINISHED, si.Status.PAUSED)
        fails = ('foo', 'foobar', 5, 10, None)

        for status in passes:
            with self.subTest(x = status):
                self.obj.status = status

        for status in fails:
            with self.subTest(x = status):
                with self.assertRaises(TypeError):
                    self.obj.status = status


class TestSumming(unittest.TestCase):
    def setUp(self):
        self.summand_one = si.Summand()
        self.summand_two = si.Summand()
        self.sum = self.summand_one + self.summand_two

    def test_is(self):
        self.assertFalse(self.summand_one is self.summand_two)

    def test_equality(self):
        self.assertFalse(self.summand_one == self.summand_two)

    def test_instance_of(self):
        self.assertTrue(isinstance(self.summand_one, si.Summand))
        self.assertTrue(isinstance(self.summand_two, si.Summand))
        self.assertTrue(isinstance(self.sum, si.Summand))
        self.assertTrue(isinstance(self.sum, si.Sum))

        self.assertFalse(isinstance(self.summand_one, si.Sum))
        self.assertFalse(isinstance(self.summand_two, si.Sum))

    def test_container(self):
        self.assertTrue(self.summand_one in self.sum.summands)
        self.assertTrue(self.summand_two in self.sum.summands)

        self.assertTrue(self.summand_one in self.sum)
        self.assertTrue(self.summand_two in self.sum)


class TestSummingSubclassing(unittest.TestCase):
    def setUp(self):
        class Fruit(si.Summand):
            def __init__(self):
                super().__init__()
                self.summation_class = FruitBasket

        class FruitBasket(si.Sum, Fruit):
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
        self.assertTrue(isinstance(self.apple, si.Summand))
        self.assertTrue(isinstance(self.banana, si.Summand))
        self.assertTrue(isinstance(self.fruit_basket, si.Summand))
        self.assertTrue(isinstance(self.fruit_basket, si.Sum))

        self.assertFalse(isinstance(self.apple, si.Sum))
        self.assertFalse(isinstance(self.banana, si.Sum))

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
