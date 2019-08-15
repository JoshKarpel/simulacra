import simulacra as si


class Foo(si.utils.StrEnum):
    BAR = "bar"
    BAZ = "baz"


f = Foo.BAR

print(f)
print(repr(f))

import typing

A = typing.NewType("A", int)

print(A)
print(repr(A))
