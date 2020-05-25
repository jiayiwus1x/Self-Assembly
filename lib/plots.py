from __future__ import division

import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lib import func as func


def sphere(r, x0, y0, z0):
    '''
    making sphere plot based on center of the sphere and radius
    :param r: radius of sphere
    :param x0: x coordinate of center
    :param y0: y coordinate of center
    :param z0: z coordinate of center
    :return: x,y,z of surface of the sphere for plotting purpose
    '''
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v)) + x0
    y = r * np.outer(np.sin(u), np.sin(v)) + y0
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z0
    return x, y, z


def plot(snap, bs, datapath, int_mat, N, type_names=[]):
    '''
    plotting structure create by the individual spherical particles
    '''
    positions = snap.particles.position

    for i in range(3):

        if max(positions[:, i]) - min(positions[:, i]) > bs / 2:
            positions[:, i] = (positions[:, i] + bs) % bs

    cm = func.con_matrix(positions)
    cm_w_n = func.con_matrix_test(positions)

    plotpath_2 = datapath + '/plots_part/'
    if not os.path.exists(plotpath_2): os.mkdir(plotpath_2)
    timenow = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    np.savetxt(plotpath_2 + '/' + timenow + 'con_matrix.txt', cm_w_n)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221, projection='3d')
    cmap = matplotlib.cm.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, N))

    for i in range(N):
        x, y, z = sphere(0.5, positions[i, 0], positions[i, 1], positions[i, 2])

        ax.plot_surface(x, y, z, color=colors[i])

    ax.scatter(np.mean(positions[:, 0]), np.mean(positions[:, 1]), np.mean(positions[:, 2]), s=10)

    ax.set_axis_off()
    ax.set_aspect('equal')

    ax1 = fig.add_subplot(223, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=100, color=colors)
    for i in range(N):
        for j in range(i + 1, N):
            if cm[i, j] != 0:
                x = [positions[i][0], positions[j][0]]
                y = [positions[i][1], positions[j][1]]
                z = [positions[i][2], positions[j][2]]
                plt.plot(x, y, z, color='pink', alpha=0.8)
    ax1.set_axis_off()
    ax1.set_aspect('equal')

    ax1.set_xlim(-1. + np.mean(positions[:, 0]), 1 + np.mean(positions[:, 0]))
    ax1.set_ylim(-1. + np.mean(positions[:, 1]), 1 + np.mean(positions[:, 1]))
    ax1.set_zlim(-1. + np.mean(positions[:, 2]), 1 + np.mean(positions[:, 2]))

    plt.subplot(222)

    plt.imshow(cm)
    plt.xticks(range(N), type_names)
    plt.yticks(range(N), type_names)

    plt.title('contact matrix', fontsize=20)

    plt.subplot(224)
    plt.imshow(int_mat, cmap='coolwarm')
    plt.xticks(range(N), type_names)
    plt.yticks(range(N), type_names)
    plt.title('interaction matrix', fontsize=20)
    plt.colorbar()

    plt.savefig(plotpath_2 + '/' + timenow + '_final_config.png')
    plt.close()

def plot_con(cx, position, savedir, N, type_names=['red', 'purple', 'green', 'blue', 'orange', 'yellow']):
    '''
    plot the contact matrix of the structure
    '''
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(position[:, 0], position[:, 1], position[:, 2], s=100, color=type_names)
    for i in range(N):
        for j in range(i + 1, N):
            if cx[i, j] != 0:
                x = [position[i][0], position[j][0]]
                y = [position[i][1], position[j][1]]
                z = [position[i][2], position[j][2]]
                plt.plot(x, y, z, color='pink', alpha=0.8)
    ax.set_axis_off()
    ax.set_aspect('equal')

    plt.savefig(savedir + 'positions.png')
    plt.close()
