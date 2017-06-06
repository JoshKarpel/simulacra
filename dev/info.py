import logging
import os

import simulacra as si


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


class Foo(si.Beet):
    def __init__(self, name, a = 5):
        self.a = a

        super().__init__(name)

    def info(self):
        info = super().info()

        foo_info = si.Info(header = 'foo info')
        foo_info.children['a'] = self.a
        # sup_info.fields['FOO INFO'] = foo_info

        info.add_info(foo_info)

        return info


class BarSim(si.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        self.car = spec.foo * spec.bar

    def info(self):
        info = super().info()

        info.add_field('car', self.car)

        return info


class Bat:
    def __init__(self, species = 'brown'):
        self.species = species

    def info(self):
        info = si.Info(header = 'Bat')
        info.add_field('species', self.species)

        return info


class BarSpec(si.Specification):
    def __init__(self, name, foo = 5, bar = 3, bat = Bat(), pot = si.Summand(), **kwargs):
        self.foo = foo
        self.bar = bar
        self.bat = bat
        self.pot = pot

        super().__init__(name, simulation_type = BarSim, **kwargs)

    def info(self):
        info = super().info()

        info.add_field('foo', self.foo)
        info.add_field('car', self.bar)

        info.add_info(self.bat.info())
        info.add_info(self.pot.info())

        return info

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        foo = Foo('foo', a = 6)

        print(foo.info())
        print()

        spec = BarSpec('test', file_name = 'foo', pot = si.Summand() + si.Summand())
        sim = spec.to_simulation()

        print(spec.info())

        print()
        print('-' * 80)
        print()

        print(sim.info())
        # print(sim.info().fields)
        print()
