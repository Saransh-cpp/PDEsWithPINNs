import sys

sys.path.insert(0, "../utils/")

import numpy as np
import pandas as pd
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt
from plot import plot_3D, plot_2D
from dat_to_csv import dat_to_csv


c = 10  # wave equation constant
C = 4 / np.pi  # Fourier constant
n = 1
m = 1

# dimensions
a = 1
b = 1


def pde(x, u):
    u_tt = dde.grad.hessian(u, x, i=2, j=2)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    return u_tt - ((c ** 2) * (u_xx + u_yy))


def sol(x):
    return (
        C
        * np.sin(m * np.pi * x[:, 0:1] / a)
        * np.sin(n * np.pi * x[:, 1:2] / b)
        * np.cos(np.pi * c * x[:, 2:3])  # (m^2/a^2  +  n^2/b^2) = 1
    )


def boundary_init(x, _):
    return np.isclose(x[-1], 0)


def get_initial_loss(model):
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]


spatial_domain = dde.geometry.Rectangle(xmin=[0, 0], xmax=[a, b])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

d_bc = dde.DirichletBC(
    spatio_temporal_domain, lambda x: 0, lambda _, on_boundary: on_boundary
)
ic = dde.IC(
    spatio_temporal_domain,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / a),
    lambda _, on_initial: on_initial,
)
ic_2 = dde.OperatorBC(
    spatio_temporal_domain,
    lambda x, u, _: dde.grad.jacobian(u, x, i=0, j=1),
    boundary_init,
)

data = dde.data.TimePDE(
    spatio_temporal_domain,
    pde,
    [d_bc, ic, ic_2],
    num_domain=360,
    num_boundary=360,
    num_initial=360,
    num_test=10000,
    solution=sol,
)

net = dde.nn.STMsFFN(
    [3] + [100] * 3 + [300] * 2 + [100] * 2 + [1],
    "tanh",
    "Glorot uniform",
    sigmas_x=[50, 100, 200],
    sigmas_t=[50, 100, 200],
)

net.apply_feature_transform(lambda x: (x - 0.5) * 2 * np.sqrt(3))

model = dde.Model(data, net)
initial_losses = get_initial_loss(model)
loss_weights = 1 / (2 * initial_losses)
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
    test_fname="../dat_data/wave_2D.dat",
)

dat_to_csv(
    dat_file_name="../dat_data/wave_2D.dat",
    csv_file_name="../csv_data/wave_2D.csv",
    columns=["x", "y", "t", "u_true", "u_pred"],
)
plot_3D(
    csv_file_name="../csv_data/wave_2D.csv",
    columns=["y", "t", "u_true", "u_pred"],
    x_axis="y",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["y", "u_true / u_pred", "t"],
)
plot_3D(
    csv_file_name="../csv_data/wave_2D.csv",
    columns=["x", "t", "u_true", "u_pred"],
    x_axis="x",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["x", "u_true / u_pred", "t"],
)
