import sys

sys.path.insert(0, "../utils/")

import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt
from plot import plot_3D, plot_2D
from dat_to_csv import dat_to_csv


k = 0.4

l = 1
n = 1
m = 1
q = 1

a = 1
b = 1
c = 1
d = 1

C = 1


def pde(x, u):
    u_t = dde.grad.jacobian(u, x, i=0, j=4)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    u_zz = dde.grad.hessian(u, x, i=2, j=2)
    u_ww = dde.grad.hessian(u, x, i=3, j=3)
    return u_t - k * (u_xx + u_yy + u_zz + u_ww)


def func(x):
    return (
        C
        * np.sin(m * np.pi * x[:, 0:1] / a)
        * np.sin(n * np.pi * x[:, 1:2] / b)
        * np.sin(q * np.pi * x[:, 2:3] / c)
        * np.sin(l * np.pi * x[:, 3:4] / d)
        * np.exp(
            -(4 * k * np.pi ** 2 * x[:, 4:5])
        )  # (m^2/a^2  +  n^2/b^2  +  q^2/c^2  +  l^2/d^2) = 4
    )


spatial_domain = dde.geometry.Hypercube(xmin=[0, 0, 0, 0], xmax=[a, b, c, d])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

d_bc = dde.DirichletBC(
    spatio_temporal_domain, lambda x: 0, lambda _, on_boundary: on_boundary
)
ic = dde.IC(
    spatio_temporal_domain,
    lambda x: np.sin(np.pi * x[:, 0:1])
    * np.sin(np.pi * x[:, 1:2])
    * np.sin(np.pi * x[:, 2:3])
    * np.sin(np.pi * x[:, 3:4]),
    lambda _, on_initial: on_initial,
)

data = dde.data.TimePDE(
    spatio_temporal_domain,
    pde,
    [d_bc, ic],
    num_domain=2540,
    num_boundary=360,
    num_initial=360,
    num_test=10000,
    solution=func,
)

layer_size = [5] + [100] * 3 + [1]
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
    test_fname="../dat_data/heat_4D.dat",
)

dat_to_csv(
    dat_file_name="../dat_data/heat_4D.dat",
    csv_file_name="../csv_data/heat_4D.csv",
    columns=["x", "y", "z", "w", "t", "u_true", "u_pred"],
)

plot_3D(
    csv_file_name="../csv_data/heat_4D.csv",
    columns=["y", "t", "u_true", "u_pred"],
    x_axis="y",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["y", "u_true / u_pred", "t"],
)
plot_3D(
    csv_file_name="../csv_data/heat_4D.csv",
    columns=["x", "t", "u_true", "u_pred"],
    x_axis="x",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["x", "u_true / u_pred", "t"],
)
plot_3D(
    csv_file_name="../csv_data/heat_4D.csv",
    columns=["z", "t", "u_true", "u_pred"],
    x_axis="z",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["z", "u_true / u_pred", "t"],
)
plot_3D(
    csv_file_name="../csv_data/heat_4D.csv",
    columns=["w", "t", "u_true", "u_pred"],
    x_axis="w",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["w", "u_true / u_pred", "t"],
)
