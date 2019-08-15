import os

import core


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)


class Fruit(core.Summand):
    def __init__(self):
        super(Fruit, self).__init__()
        self.sum_target = FruitBasket


class FruitBasket(core.Sum, Fruit):
    container_name = "basket"


class Apple(Fruit):
    pass


class Banana(Fruit):
    pass


if __name__ == "__main__":
    a = Apple()
    b = Banana()

    c = a + b
    print(c)
    print(repr(c))
    print(c.basket)
    print(vars(c))

    print(c, "is instance of FruitBasket?", isinstance(c, FruitBasket))
    print(c, "is instance of Fruit?", isinstance(c, Fruit))
    print(c, "is instance of Summand?", isinstance(c, core.Summand))
    print(c, "is instance of Sum?", isinstance(c, core.Sum))
    print(c, "is instance of Apple?", isinstance(c, Apple))
    print(c, "is instance of Banana?", isinstance(c, Banana))
    print(c, "is instance of int?", isinstance(c, int))
