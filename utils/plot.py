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

    _ = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter(x, u_pred, z, "green")

    ax.scatter(x, u_true, z, cmap="viridis")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    plt.show()


def plot_2D(
    csv_file_name,
    columns,
    x_axis,
    u_true,
    u_pred,
    t_values,
    labels=["u_pred", "u_true"],
    y_lim=(0, 1),
    plt_type="scatter",
    subplots_dim=(2, 5),
    figsize=(15, 8),
):
    df = pd.read_csv(csv_file_name, usecols=columns)
    dfs = []
    xs = []
    u_trues = []
    u_preds = []

    for i in range(len(t_values)):
        dfs.append(df[np.isclose(df.t, t_values[i])])
        xs.append(dfs[i][x_axis])
        u_trues.append(dfs[i][u_true])
        u_preds.append(dfs[i][u_pred])

    fig, axes = plt.subplots(subplots_dim[0], subplots_dim[1], figsize=figsize)

    for x, u_true, u_pred, ax in zip(xs, u_trues, u_preds, axes.flat):
        if plt_type == "scatter":
            ax.scatter(x, u_pred)
            ax.scatter(x, u_true)
        elif plt_type == "interpolated":
            ax.plot(x, u_pred, "--")
            ax.plot(x, u_true, "**")
        ax.set_ylim(y_lim[0], y_lim[1])

    fig.tight_layout()
    fig.legend(labels, loc="lower right")
    plt.show()
