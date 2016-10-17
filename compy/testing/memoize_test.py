import time

from compy.utils import Timer, memoize


def fib(n):
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)


@memoize()
def foo(a = 5):
    time.sleep(1)
    return a + 1


if __name__ == '__main__':
    test = 25

    with Timer() as t_bare:
        print([(n, fib(n)) for n in range(test)])

    print('no memo:', t_bare)

    fib = memoize()(fib)
    with Timer() as t_memo:
        print([(n, fib(n)) for n in range(test)])

    print('memo:', t_memo)

    with Timer() as t_foo_bare:
        print([(n, foo(a = n)) for n in range(5)])

    print('foo', t_foo_bare)

    with Timer() as t_foo_memo:
        print([(n, foo(a = n)) for n in range(5)])

    print('foo', t_foo_memo)
    print(foo.memo)


