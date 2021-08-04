#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-14-21 16:11
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Invertible_1x1_Conv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039).

    Does not support LU-decomposed version YET.

    Reference: https://github.com/openai/glow/blob/master/model.py#L438

    Args:
        num_channels (int): Number of channels in the input and output.
    """

    def __init__(self, in_channels: int, out_channels: int, LU_decomposed: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Invertible_1x1_Conv, self).__init__()
        self.in_channels = in_channels  # Store
        self.out_channels = out_channels  # Store
        self.kernel_channel = [out_channels, in_channels]  # [C_out , C_in]
        self.weight = Parameter(torch.empty(
            self.kernel_channel, **factory_kwargs))
        self.reset_parameters(LU_decomposed)

    def reset_parameters(self, LU_decomposed) -> None:
        if not LU_decomposed:

            w_shape = self.kernel_channel
            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(
                *w_shape))[0].astype('float32')
            self.weight = Parameter(
                torch.from_numpy(w_init).reshape([*w_shape, 1, 1]))  # 1x1 Conv
        else:

            # From https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
            # [Warning!] NOT sure.
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, z, logdet, reverse=False, LU_decomposed=False):
        """Invertible 1x1 conv, port from https://github.com/openai/glow/blob/master/model.py

        Args:
            z (torch.Tensor): (N, C, H, W)
            LU_decomposed (bool): default "False", set to "True" to use the LU-decomposed version
        """
        if not LU_decomposed:

            w = self.weight

            dlogdet = w.double().det().abs().log().float() *\
                z.size(2) * z.size(3)  # Multiplied with [H, W]

            if not reverse:

                z = F.conv2d(z, w)
                logdet += dlogdet

                return z, logdet
            else:

                w_shape = self.kernel_channel
                _w = w.reshape(w_shape).inverse()
                _w = _w.reshape(w_shape+[1, 1])
                z = F.conv2d(z, _w)
                logdet -= dlogdet

                return z, logdet

        else:

            # LU-decomposed version
            shape = Z.int_shape(z)
            with tf.variable_scope(name):

                dtype = 'float64'

                # Random orthogonal matrix:
                import scipy
                np_w = scipy.linalg.qr(np.random.randn(shape[3], shape[3]))[
                    0].astype('float32')

                np_p, np_l, np_u = scipy.linalg.lu(np_w)
                np_s = np.diag(np_u)
                np_sign_s = np.sign(np_s)
                np_log_s = np.log(abs(np_s))
                np_u = np.triu(np_u, k=1)

                p = tf.get_variable("P", initializer=np_p, trainable=False)
                l = tf.get_variable("L", initializer=np_l)
                sign_s = tf.get_variable(
                    "sign_S", initializer=np_sign_s, trainable=False)
                log_s = tf.get_variable("log_S", initializer=np_log_s)
                # S = tf.get_variable("S", initializer=np_s)
                u = tf.get_variable("U", initializer=np_u)

                p = tf.cast(p, dtype)
                l = tf.cast(l, dtype)
                sign_s = tf.cast(sign_s, dtype)
                log_s = tf.cast(log_s, dtype)
                u = tf.cast(u, dtype)

                w_shape = [shape[3], shape[3]]

                l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
                l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
                u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
                w = tf.matmul(p, tf.matmul(l, u))

                if True:
                    u_inv = tf.matrix_inverse(u)
                    l_inv = tf.matrix_inverse(l)
                    p_inv = tf.matrix_inverse(p)
                    w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
                else:
                    w_inv = tf.matrix_inverse(w)

                w = tf.cast(w, tf.float32)
                w_inv = tf.cast(w_inv, tf.float32)
                log_s = tf.cast(log_s, tf.float32)

                if not reverse:

                    w = tf.reshape(w, [1, 1] + w_shape)
                    z = tf.nn.conv2d(z, w, [1, 1, 1, 1],
                                     'SAME', data_format='NHWC')
                    logdet += tf.reduce_sum(log_s) * (shape[1]*shape[2])

                    return z, logdet
                else:

                    w_inv = tf.reshape(w_inv, [1, 1]+w_shape)
                    z = tf.nn.conv2d(
                        z, w_inv, [1, 1, 1, 1], 'SAME', data_format='NHWC')
                    logdet -= tf.reduce_sum(log_s) * (shape[1]*shape[2])

                    return z, logdet
