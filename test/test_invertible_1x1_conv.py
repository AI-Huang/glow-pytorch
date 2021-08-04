#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-04-21 21:43
# @Author  : Kelly Kan HUANG (kan.huang@connect.ust.hk)

import torch
import torch.nn.functional as F
from models.glow.invertible_1x1_conv import Invertible_1x1_Conv


def invertible_1x1_conv_test():
    """
    Test cases:
        conv1x1 = Invertible_1x1_Conv(
        in_channels=in_channels, out_channels=out_channels, LU_decomposed=False)
    """
    in_channels = 3
    # out_channels = 16 # in_channels and out_channels must equal
    out_channels = 3
    conv1x1 = Invertible_1x1_Conv(
        in_channels=in_channels, out_channels=out_channels, LU_decomposed=False)

    x = torch.randn([64, in_channels, 32, 32])
    print(x.shape)
    print(f"x.sum(): {x.sum()}")

    logdet_init = 0

    # Forward obversely
    z, logdet = conv1x1(x, logdet_init)
    print(z.shape)
    print(f"z.sum(): {z.sum()}")

    # Forward reversely
    x_hat, logdet_final = conv1x1(z, logdet, reverse=True)
    print(x_hat.shape)
    print(f"x_hat.sum(): {x_hat.sum()}")

    print(f"F.l1_loss(x, x_hat): {F.l1_loss(x, x_hat)}")

    assert F.l1_loss(x, x_hat) < 1


def main():
    invertible_1x1_conv_test()


if __name__ == "__main__":
    main()
