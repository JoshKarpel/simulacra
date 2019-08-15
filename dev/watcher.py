import simulacra as si


def not_watched_func(x):
    print("not_watched_func ran")
    return x + 1


class Foo:
    def __init__(self):
        self.w = 0

    @si.utils.watched_memoize(lambda s: s.w)
    def watched_method(self, x):
        print("watched_method ran")
        return x + 1

    @si.utils.watched_memoize(lambda s: s.w)
    def watched_method_no_args(self):
        print("watched_method_no_args ran")
        return "foo"


if __name__ == "__main__":
    print(not_watched_func(1))
    print(not_watched_func(2))
    print(not_watched_func(3))

    print()

    f = Foo()
    print(f.watched_method(1))
    print(f.watched_method(1))
    print(f.watched_method(1))
    f.w = 1
    print(f.watched_method(2))
    print(f.watched_method(3))
    print(f.watched_method(4))

    print()

    print(f.watched_method_no_args())
    print(f.watched_method_no_args())
    print(f.watched_method_no_args())
    f.w = 2
    print(f.watched_method_no_args())
    print(f.watched_method_no_args())
    print(f.watched_method_no_args())
