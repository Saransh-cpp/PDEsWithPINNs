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
        * np.exp(-2 * k * np.pi ** 2 * x[:, 2:3])  # (m^2/a^2  +  n^2/b^2) = 1
    )


def boundary_u(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)


def boundary_b(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)


def boundary_r_and_l(x, on_boundary):
    return on_boundary and (np.isclose(x[0], 0) or np.isclose(x[0], 1))


spatial_domain = dde.geometry.Rectangle(xmin=[0, 0], xmax=[a, b])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

d_bc_b = dde.DirichletBC(spatio_temporal_domain, lambda x: 0, boundary_b)
d_bc_u = dde.DirichletBC(spatio_temporal_domain, lambda x: 0, boundary_u)
d_bc_rl = dde.DirichletBC(spatio_temporal_domain, lambda x: 0, boundary_r_and_l)
ic = dde.IC(
    spatio_temporal_domain,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / a),
    lambda _, on_initial: on_initial,
)


data = dde.data.TimePDE(
    spatio_temporal_domain,
    pde,
    [d_bc_b, d_bc_u, d_bc_rl, ic],
    num_domain=2540,
    num_boundary=1000,
    num_initial=1000,
    num_test=2540,
    solution=func,
)

layer_size = [3] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
)
model.train(epochs=20000)
model.compile(
    "L-BFGS",
    metrics=["l2 relative error"],
)
losshistory, train_state = model.train()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dat_to_csv("test.dat", "heat_2D_test.csv", ["x", "y", "t", "u_true", "u_pred"])
scatter_plot_3D(
    "heat_2D_test.csv", ["y", "t", "u_true", "u_pred"], "y", "t", "u_true", "u_pred", labels=["y", "u_true / u_pred", "t"]
)