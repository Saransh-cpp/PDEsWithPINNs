"""Backend supported: tensorflow.compat.v1
Implementation of the wave propagation example in paper https://arxiv.org/abs/2012.10047.
References:
    https://github.com/PredictiveIntelligenceLab/MultiscalePINNs.
"""
import deepxde as dde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dat_to_csv import dat_to_csv


A = 2
C = 10


def pde(x, y):
    dy_tt = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_tt - C ** 2 * dy_xx


def func(x):
    x, t = np.split(x, 2, axis=1)
    return np.sin(np.pi * x) * np.cos(C * np.pi * t) + np.sin(A * np.pi * x) * np.cos(
        A * C * np.pi * t
    )


def plot():
    df = pd.read_csv("wave_1D_test.csv", usecols=["x", "t", "y_true", "y_pred"])
    x = df["x"]
    t = df["t"]
    y_true = df["y_true"]
    y_pred = df["y_pred"]

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter(x, y_pred, t, "green")

    ax.scatter(x, y_true, t, cmap="viridis")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel(r"$t$")

    plt.show()


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic_1 = dde.IC(geomtime, func, lambda _, on_initial: on_initial)
# do not use dde.NeumannBC here, since `normal_derivative` does not work with temporal coordinate.
ic_2 = dde.OperatorBC(
    geomtime,
    lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
    lambda x, _: np.isclose(x[1], 0),
)
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic_1, ic_2],
    num_domain=2540,
    num_boundary=1000,
    num_initial=1000,
    num_test=10000,
    solution=func
)

layer_size = [2] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.STMsFFN(
    layer_size, activation, initializer, sigmas_x=[1], sigmas_t=[1, 10]
)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"],)
losshistory, train_state = model.train(0)
model.compile(
    "adam",
    lr=0.001,
    metrics=["l2 relative error"],
)
losshistory, train_state = model.train(
    epochs=10000,
)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

dat_to_csv("test.dat", "wave_1D_test.csv", ["x", "t", "y_true", "y_pred"])
plot()
