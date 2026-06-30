"""Pake powder lineshape model (numpy)."""

from __future__ import annotations

import numpy as np

g = 0.05
s = 0.04
bigy = np.sqrt(3 - s)


def Lineshape(x, eps):
    def cosal(x, eps):
        return (1 - eps * x - s) / bigxsquare(x, eps)

    def bigxsquare(x, eps):
        return np.sqrt(g**2 + (1 - eps * x - s) ** 2)

    def mult_term(x, eps):
        return 1 / (2 * np.pi * np.sqrt(bigxsquare(x, eps)))

    def cosaltwo(x, eps):
        return np.sqrt((1 + cosal(x, eps)) / 2)

    def sinaltwo(x, eps):
        return np.sqrt((1 - cosal(x, eps)) / 2)

    def termone(x, eps):
        return np.pi / 2 + np.arctan(
            (bigy**2 - bigxsquare(x, eps))
            / (2 * bigy * np.sqrt(bigxsquare(x, eps)) * sinaltwo(x, eps))
        )

    def termtwo(x, eps):
        return np.log(
            (bigy**2 + bigxsquare(x, eps) + 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps))
            / (bigy**2 + bigxsquare(x, eps) - 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps))
        )

    def icurve(x, eps):
        return mult_term(x, eps) * (2 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))

    return icurve(x, eps) / 10


def GenerateVectorLineshape(P, x):
  r = (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)

  i_plus_sign = 1
  i_minus_sign = 1
  if P < 0:
    r = 1 / r
    i_plus_sign = -1
    i_minus_sign = -1

  Iplus = i_plus_sign * r * Lineshape(x, 1)
  Iminus = i_minus_sign * Lineshape(x, -1)

  p_summed = np.sum(Iplus + Iminus)
  if p_summed == 0:
    delta_p = 0.0
  else:
    delta_p = P / p_summed
  Iplus = Iplus * delta_p
  Iminus = Iminus * delta_p
  signal = Iplus + Iminus

  return signal, Iplus, Iminus