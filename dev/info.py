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
        sup_info = super().info()
        foo_info = si.Info(header = 'foo info')
        foo_info.fields['a'] = self.a
        sup_info.fields['FOO INFO'] = foo_info

        return sup_info

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        foo = Foo('foo', a = 6)

        print(foo.info())
        print()

        spec = si.Specification('test')
        sim = spec.to_simulation()

        print(spec.info())
        print()

        print(sim.info())
        print(sim.info().fields)
        print()
