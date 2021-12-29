import enum
import math
import csv

from scipy.stats import studentized_range
import scipy.integrate as integrate
import numpy as np
import mpmath
import sympy
from tabulate import tabulate


class ShewhartConstants:
    """
    n to 343
    # https://stackoverflow.com/questions/64490723/math-gamma-limited-by-float-64-bit-range-any-way-to-assign-more-bits
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """singleton mode"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):  # noqa NOSONAR
        pass

    def d2(self, n: int):
        def func_d2(x):
            return 1 - studentized_range.cdf(x, n, np.inf)

        return integrate.quad(func_d2, 0, np.inf)[0]

    def d3(self, n: int):
        # n=3 result will be complex number 🙄
        if n == 3:
            return 0.888368

        def func_d3(x):
            return x * (1 - studentized_range.cdf(x, n, np.inf))

        quad_func_d3 = integrate.quad(func_d3, 0, np.inf)[0] * 2
        d2 = self.d2(n) ** 2

        return (quad_func_d3 - d2) ** 0.5

    def c4(self, n: int):
        return math.gamma(n / 2) * (((2 / (n - 1)) ** 0.5) / math.gamma((n - 1) / 2))

    def A2(self, n):  # noqa NOSONAR
        return 3 / (self.d2(n) * (n ** 0.5))

    def A3(self, n):  # noqa NOSONAR
        return 3 / (self.c4(n) * (n ** 0.5))

    def D3(self, n):  # noqa NOSONAR
        r = 1 - 3 * (self.d3(n) / self.d2(n))
        if isinstance(r, complex):
            return r
        else:
            return max(0.0, r)

    def D4(self, n):  # noqa NOSONAR
        return 1 + 3 * (self.d3(n) / self.d2(n))

    def B3(self, n):  # noqa NOSONAR
        r = 1 - 3 * ((1 - (self.c4(n) ** 2)) ** 0.5) / self.c4(n)
        if isinstance(r, complex):
            return r
        else:
            return max(0.0, r)

    def B4(self, n):  # noqa NOSONAR
        return 1 + 3 * ((1 - (self.c4(n) ** 2)) ** 0.5) / self.c4(n)


class ShewhartConstantsFix(ShewhartConstants):
    """
    n to 343
    fix math-gamma-limited-by-float-64-bit-range
    # https://stackoverflow.com/questions/64490723/math-gamma-limited-by-float-64-bit-range-any-way-to-assign-more-bits
    """

    def c4(self, n: int):
        return sympy.Float(mpmath.gamma(n / 2) * (((2 / (n - 1)) ** 0.5) / mpmath.gamma((n - 1) / 2)), 6)


class SCTitle(str, enum.Enum):
    n = "n"
    d2 = "d2"
    d3 = "d3"
    c4 = "c4"
    A2 = "A2"  # noqa NOSONAR
    A3 = "A3"  # noqa NOSONAR
    D3 = "D3"  # noqa NOSONAR
    D4 = "D4"  # noqa NOSONAR
    B3 = "B3"  # noqa NOSONAR
    B4 = "B4"  # noqa NOSONAR


def print_to_csv(n):
    csv_header = [SCTitle.n,
                  SCTitle.d2,
                  SCTitle.d3,
                  SCTitle.c4,
                  SCTitle.A2,
                  SCTitle.A3,
                  SCTitle.D3,
                  SCTitle.D4,
                  SCTitle.B3,
                  SCTitle.B4]
    with open(f'shewhart_constants_{n}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for n in range(2, n + 1):
            writer.writerow(
                [n, sc.d2(n), sc.d3(n), sc.c4(n), sc.A2(n), sc.A3(n), sc.D3(n), sc.D4(n), sc.B3(n), sc.B4(n)])


if __name__ == '__main__':
    sc = ShewhartConstantsFix()
    table_header = [SCTitle.n.value,
                    SCTitle.d2.value,
                    SCTitle.d3.value,
                    SCTitle.c4.value,
                    SCTitle.A2.value,
                    SCTitle.A3.value,
                    SCTitle.D3.value,
                    SCTitle.D4.value,
                    SCTitle.B3.value,
                    SCTitle.B4.value]
    print(
        tabulate(
            [
                (n, sc.d2(n), sc.d3(n), sc.c4(n), sc.A2(n), sc.A3(n), sc.D3(n), sc.D4(n), sc.B3(n), sc.B4(n),)
                for n in range(2, 30)
            ], table_header, tablefmt="pretty")
    )
