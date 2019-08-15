import logging
import os

import simulacra as si
import simulacra.info

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)


class Foo(si.Beet):
    def __init__(self, name, a=5):
        self.a = a

        super().__init__(name)

    def info(self) -> si.Info:
        info = super().info()

        foo_info = simulacra.info.Info(header="foo info")
        foo_info._children["a"] = self.a
        # sup_info.fields['FOO INFO'] = foo_info

        info.add_info(foo_info)

        return info


class BarSim(si.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        self.car = spec.foo * spec.bar

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("car", self.car)

        return info

    def run(self):
        pass


class Bat:
    def __init__(self, species="brown"):
        self.species = species

    def info(self) -> si.Info:
        info = simulacra.info.Info(header="Bat")
        info.add_field("species", self.species)
        info.add_field("favorite pancake", "blueberry")

        return info


class BarSpec(si.Specification):
    simulation_type = BarSim

    def __init__(self, name, foo=5, bar=3, bat=Bat(), **kwargs):
        self.foo = foo
        self.bar = bar
        self.bat = bat

        super().__init__(name, **kwargs)

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("foo", self.foo)
        info.add_field("car", self.bar)

        info.add_info(self.bat.info())
        info.add_field("wassup", "nuthin")

        info.add_field("yo", "dawg")

        return info


if __name__ == "__main__":
    foo = Foo("foo", a=6)

    print(foo.info())
    print()

    spec = BarSpec("test", file_name="foo")
    sim = spec.to_sim()

    print(spec.info())

    print()
    print("-" * 80)
    print()

    print(sim.info())
    # print(sim.info().fields)
    print()

    # sim.info().log()

    # info = si.Info(header = 'top')
    # info.add_field('foo', 'bar')
    #
    # subinfo = si.Info(header = 'middle')
    # subinfo.add_field('gaz', 'baz')
    # info.add_info(subinfo)
    #
    # info.add_field('bar', 'foo')
    #
    # p = str(info)
    # print('\n------------------------------------------\n')
    # print(p)
