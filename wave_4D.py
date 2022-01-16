import numpy as np

import deepxde as dde
from deepxde.backend import tf


nu_ref = 0.1

n = 1
L = 1


def pde(x, u):
    u_tt = dde.grad.hessian(u, x, i=4, j=4)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    u_zz = dde.grad.hessian(u, x, i=2, j=2)
    u_ww = dde.grad.hessian(u, x, i=3, j=3)
    return u_tt - nu_ref * (u_xx + u_yy + u_zz + u_ww)


# def func(x):
#     return (
#         np.sin(np.pi * x[:, 0:1])
#         * np.sin(np.pi * x[:, 1:2])
#         * np.exp(-2 * nu_ref * np.pi ** 2 * x[:, 2:3])
#     )


def boundary_u(x, on_boundary):
    return on_boundary and np.isclose(x[1], 1)


def boundary_b(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)


def boundary_r_and_l(x, on_boundary):
    return on_boundary and (np.isclose(x[0], 0) or np.isclose(x[0], 1))


spatial_domain = dde.geometry.Hypercube(xmin=[0, 0, 0, 0], xmax=[1, 1, 1, 1])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

d_bc_b = dde.DirichletBC(
    spatio_temporal_domain, lambda x: np.sin(n * np.pi * x[:, 0:1] / L), boundary_b
)
d_bc_u = dde.DirichletBC(spatio_temporal_domain, lambda x: 0, boundary_u)
n_bc = dde.NeumannBC(spatio_temporal_domain, lambda X: 0, boundary_r_and_l)
ic = dde.IC(
    spatio_temporal_domain,
    lambda x: 0,
    lambda _, on_initial: on_initial,
)

data = dde.data.TimePDE(
    spatio_temporal_domain,
    pde,
    [d_bc_u, d_bc_b, n_bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)

net = dde.nn.STMsFFN(
    [5] + [20] * 3 + [1], "tanh", "Glorot uniform", sigmas_x=[1], sigmas_t=[1, 10]
)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
model.train(epochs=20000)
model.compile("L-BFGS")
losshistory, train_state = model.train()

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
