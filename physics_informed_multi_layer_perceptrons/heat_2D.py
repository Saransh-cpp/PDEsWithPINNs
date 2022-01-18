import sys

sys.path.insert(0, "../utils/")

import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt
from plot import scatter_plot_3D
from dat_to_csv import dat_to_csv


k = 0.4

n = 1
m = 1
a = 1
b = 1
C = 4 / np.pi


def pde(x, u):
    u_t = dde.grad.jacobian(u, x, j=2)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    return u_t - k * (u_xx + u_yy)


def func(x):
    return (
        C
        * np.sin(m * np.pi * x[:, 0:1] / a)
        * np.sin(n * np.pi * x[:, 1:2] / b)
        * np.exp(-k * np.pi ** 2 * x[:, 2:3])  # (m^2/a^2  +  n^2/b^2) = 1
    )


spatial_domain = dde.geometry.Rectangle(xmin=[0, 0], xmax=[a, b])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

bc = dde.DirichletBC(
    spatio_temporal_domain, lambda x: 0, lambda _, on_boundary: on_boundary
)
ic = dde.IC(
    spatio_temporal_domain,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / a),
    lambda _, on_initial: on_initial,
)


data = dde.data.TimePDE(
    spatio_temporal_domain,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=500,
    num_initial=500,
    num_test=10000,
    solution=func,
)

layer_size = [3] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(
    "adam", lr=0.001, metrics=["l2 relative error"],
)
model.train(epochs=10000)
model.compile(
    "L-BFGS", metrics=["l2 relative error"],
)
losshistory, train_state = model.train()

dde.saveplot(
    losshistory,
    train_state,
    issave=True,
    isplot=True,
    test_fname="../dat_data/heat_2D.dat",
)

dat_to_csv(
    dat_file_name="../dat_data/heat_2D.dat",
    csv_file_name="../csv_data/heat_2D.csv",
    columns=["x", "y", "t", "u_true", "u_pred"],
)
scatter_plot_3D(
    csv_file_name="../csv_data/heat_2D.csv",
    columns=["y", "t", "u_true", "u_pred"],
    x_axis="y",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["y", "u_true / u_pred", "t"],
)
scatter_plot_3D(
    csv_file_name="../csv_data/heat_2D.csv",
    columns=["x", "t", "u_true", "u_pred"],
    x_axis="x",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["x", "u_true / u_pred", "t"],
)
