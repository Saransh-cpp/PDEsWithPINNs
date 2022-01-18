import sys

sys.path.insert(0, "../utils/")

import numpy as np
import pandas as pd
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt
from dat_to_csv import dat_to_csv
from plot import scatter_plot_3D, plot_2D


a = 0.4  # Thermal diffusivity
L = 1  # Length of the bar
n = 1  # Frequency of the sinusoidal initial conditions
C = 1  # Fourier constant


def func(x):
    """
    Returns the exact solution for a given x and t (for sinusoidal initial conditions).
    """
    x, t = np.split(x, 2, axis=1)

    # C is 1 -> Fourier sine series
    # check for "-" sign
    return (
        C
        * np.exp(-(n ** 2 * np.pi ** 2 * a * t) / (L ** 2))
        * np.sin(n * np.pi * x / L)
    )


def pde(x, y):
    """
    Expresses the PDE residual of the heat equation.
    """
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a * dy_xx


# Computational geometry:
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Initial and boundary conditions:
bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.IC(
    geomtime,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
    lambda _, on_initial: on_initial,
)

# Define the PDE problem and configurations of the network:
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
    solution=func,
)
net = dde.nn.FNN([2] + [16] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Build and train the model:
model.compile(
    "adam", lr=1e-3, metrics=["l2 relative error"],
)
model.train(epochs=10000)
model.compile(
    "L-BFGS", metrics=["l2 relative error"],
)
losshistory, train_state = model.train()

# Plot/print the results
dde.saveplot(
    losshistory,
    train_state,
    issave=True,
    isplot=True,
    test_fname="../dat_data/heat_1D.dat",
)

dat_to_csv(
    "../dat_data/heat_1D.dat", "../csv_data/heat_1D.csv", ["x", "t", "u_true", "u_pred"]
)
scatter_plot_3D(
    csv_file_name="../csv_data/heat_1D.csv",
    columns=["x", "t", "u_true", "u_pred"],
    x_axis="x",
    z_axis="t",
    u_true="u_true",
    u_pred="u_pred",
    labels=["x", "u_true / u_pred", "t"],
)
plot_2D(
    csv_file_name="../csv_data/heat_1D.csv",
    columns=["x", "t", "u_true", "u_pred"],
    x_axis="x",
    u_true="u_true",
    u_pred="u_pred",
    t_values=[
        0,
        0.100000001490116,
        0.200000002980232,
        0.300000011920928,
        0.400000005960464,
        0.5,
        0.600000023841857,
        0.699999988079071,
        0.800000011920928,
        0.899999976158142,
        1,
    ],
)
