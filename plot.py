import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scatter_plot_3D(
    csv_file_name, columns, x_axis, z_axis, u_true, u_pred, labels=["x", "y", "z"]
):
    df = pd.read_csv(csv_file_name, usecols=columns)
    x = df[x_axis]
    z = df[z_axis]
    u_true = df[u_true]
    u_pred = df[u_pred]

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter(x, u_pred, z, "green")

    ax.scatter(x, u_true, z, cmap="viridis")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    plt.show()
