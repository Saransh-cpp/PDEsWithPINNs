"""Backend supported: tensorflow.compat.v1
Implementation of the wave propagation example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""
import sys
sys.path.insert(0, "../utils/")

import numpy as np
import pandas as pd
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt
from plot import scatter_plot_3D
from dat_to_csv import dat_to_csv


A = 2
c = 10
L = 1
C = 1
n = 1
m = 1


def pde(x, y):
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tt - c ** 2 * dy_xx


def sol(x):
    x, t = np.split(x, 2, axis=1)
    return C * np.sin(n * np.pi * x / L) * np.cos(m * np.pi * t * c / L)


def boundary_initial(x, _):
    return np.isclose(x[1], 0)


def get_initial_loss(model):
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic_1 = dde.IC(
    geomtime,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
    lambda _, on_initial: on_initial,
)
ic_2 = dde.NeumannBC(geomtime, lambda x: 0, boundary_initial)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic_1, ic_2],
    num_domain=360,
    num_boundary=360,
    num_initial=360,
    num_test=10000,
    solution=sol,
)

layer_size = [2] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.STMsFFN(
    layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, 10]
)
net.apply_feature_transform(lambda x: (x - 0.5) * 2 * np.sqrt(3))

model = dde.Model(data, net)
initial_losses = get_initial_loss(model)
loss_weights = 5 / (initial_losses + 1e-3)
losshistory, train_state = model.train(0)
model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
    decay=("inverse time", 2000, 0.9),
    loss_weights=loss_weights,
)
pde_residual_resampler = dde.callbacks.PDEResidualResampler(period=1)
losshistory, train_state = model.train(epochs=10000, callbacks=[pde_residual_resampler])

dde.saveplot(
    losshistory,
    train_state,
    issave=True,
    isplot=True,
    test_fname="../dat_data/wave_1D.dat",
)

dat_to_csv(
    "../dat_data/wave_1D.dat", "../csv_data/wave_1D.csv", ["x", "t", "u_true", "u_pred"]
)
scatter_plot_3D(
    csv_file_name="../csv_data/wave_1D.csv",
    columns=["x", "t", "u_true", "u_pred"],
    x_axis="x",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["x", "u_true / u_pred", "t"],
)
