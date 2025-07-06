import math
import numpy as np


def mW_to_dBm(mW):
    return W_to_dBm(mW / 1000)


def W_to_dBm(W):
    return 10 * math.log10(W) + 30


def dBm_to_mW(dBm):
    return math.pow(10, dBm / 10)


def dBm_to_W(dBm):
    return dBm_to_mW(dBm) / 1000


def W_to_dBW(W):
    return 10 * math.log10(W)


def dBW_to_W(dBW):
    return math.pow(10, 0.1 * dBW)


def dBW_to_dBm(dBW):
    return dBW + 30


def dBm_to_dBW(dBm):
    return dBm - 30


def get_dB(a, base):
    return 10 * math.log10(a / base)


def dB_to(dB, base):
    return 10 ** (dB / 10) * base


if __name__ == "__main__":
    # print(dBm_to_W(40))
    # print(dBm_to_W(33))
    # print(dBm_to_W(50))
    # print(W_to_dBm(100))
    # print(mW_to_dBm(100))
    # print(dBW_to_W(10))
    # print(W_to_dBW(10))
    # print(dBm_to_W(51))
    # print(dBm_to_W(20))
    # print(dBm_to_W(2))
    # print(dBm_to_W(10))
    # print(dBm_to_W(27))
    # print(dBm_to_W(40))
    print(W_to_dBW(205621768679.41006))
