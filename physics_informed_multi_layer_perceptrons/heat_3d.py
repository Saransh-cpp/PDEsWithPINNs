import sys

sys.path.insert(0, "../utils/")

import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt
from plot import plot_3D, plot_2D
from dat_to_csv import dat_to_csv


k = 0.4

n = 1
m = 1
q = 1

a = 1
b = 1
c = 1

C = 16 / (np.pi ** 2)


def pde(x, u):
    u_t = dde.grad.jacobian(u, x, i=0, j=3)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    u_zz = dde.grad.hessian(u, x, i=2, j=2)
    return u_t - k * (u_xx + u_yy + u_zz)


def func(x):
    return (
        C
        * np.sin(m * np.pi * x[:, 0:1] / a)
        * np.sin(n * np.pi * x[:, 1:2] / b)
        * np.sin(q * np.pi * x[:, 2:3] / c)
        * np.exp(-k * np.pi ** 2 * x[:, 3:4])  # (m^2/a^2  +  n^2/b^2  +  q^2/c^2) = 1
    )


def boundary_up_and_bottom(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 1) or np.isclose(x[1], 0))


def boundary_right_and_left(x, on_boundary):
    return on_boundary and (np.isclose(x[0], 0) or np.isclose(x[0], 1))


def boundary_front_and_back(x, on_boundary):
    return on_boundary and (np.isclose(x[2], 0) or np.isclose(x[2], 1))


spatial_domain = dde.geometry.Cuboid(xmin=[0, 0, 0], xmax=[a, b, c])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

d_bc_u_and_b = dde.DirichletBC(
    spatio_temporal_domain, lambda x: 0, boundary_up_and_bottom
)
d_bc_f_and_b = dde.DirichletBC(
    spatio_temporal_domain, lambda x: 0, boundary_front_and_back
)
d_bc_r_and_l = dde.DirichletBC(
    spatio_temporal_domain, lambda X: 0, boundary_right_and_left
)

ic = dde.IC(
    spatio_temporal_domain,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / a),
    lambda _, on_initial: on_initial,
)

data = dde.data.TimePDE(
    spatio_temporal_domain,
    pde,
    [d_bc_u_and_b, d_bc_r_and_l, d_bc_f_and_b, ic],
    num_domain=2540,
    num_boundary=1000,
    num_initial=1000,
    num_test=2540,
    solution=func,
)

layer_size = [4] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(
    "adam", lr=0.001, metrics=["l2 relative error"],
)
model.train(epochs=20000)
model.compile(
    "L-BFGS", metrics=["l2 relative error"],
)
losshistory, train_state = model.train()

dde.saveplot(
    losshistory,
    train_state,
    issave=True,
    isplot=True,
    test_fname="../dat_data/heat_3D.dat",
)

dat_to_csv(
    dat_file_name="../dat_data/heat_3D.dat",
    csv_file_name="../csv_data/heat_3D.csv",
    columns=["x", "y", "z", "t", "u_true", "u_pred"],
)

plot_3D(
    csv_file_name="../csv_data/heat_3D.csv",
    columns=["y", "t", "u_true", "u_pred"],
    x_axis="y",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["y", "u_true / u_pred", "t"],
)
plot_3D(
    csv_file_name="../csv_data/heat_3D.csv",
    columns=["x", "t", "u_true", "u_pred"],
    x_axis="x",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["x", "u_true / u_pred", "t"],
)
plot_3D(
    csv_file_name="../csv_data/heat_3D.csv",
    columns=["z", "t", "u_true", "u_pred"],
    x_axis="z",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["z", "u_true / u_pred", "t"],
)
