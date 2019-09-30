from __future__ import division

import numpy as np


def make_tar(N, type_names, lists):
    '''

    :param N: numbers of particles
    :param type_names: names of particles
    :param lists: lists of bonding particle
    :return: contact matrix
    '''
    tar_con_matrix1 = np.zeros((N, N))

    for i in range(N - 1):
        tar_con_matrix1[i, lists[i]] = 1
        tar_con_matrix1[lists[i], i] = 1
    #
    # plt.imshow(tar_con_matrix1)
    # plt.xticks(range(N), type_names)
    # plt.yticks(range(N), type_names)

    return tar_con_matrix1


def dis_int(r, rmin, rmax, epsilon):
    '''
    # lj with chop(when epsilon <0) interaction

    :param r: distance between two particles that are interacting
    :param rmin: min distance to interact
    :param rmax: max distance to interact
    :param epsilon: energy of LJ potential
    :return: (V,F) potential and force
    '''

    sigma = 1.

    if epsilon > 0:
        eps = epsilon

        V = eps * ((sigma / r) ** 12 - 2 * (sigma / r) ** 6)
        F = eps / r * (12 * (sigma / r) ** 12 - 2 * 6 * (sigma / r) ** 6)  # F = -grad(V)
    else:
        eps = -epsilon + 10
        V = eps * (sigma / r) ** 12
        F = eps / r * (12 * (sigma / r) ** 12)  # F = -grad(V)

    return (V, F)


def con_matrix(x):
    '''

    :param x: based on interaction list
    :return: get contact matrix
    '''
    N = len(x)
    b = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(x[i] - x[j], 2) <= 1.01:
                b[i, j] = 1
                b[j, i] = 1
                # if np.linalg.norm(x[i] - x[j], 2) < 0.7:
                #    print('particle overlap at', np.linalg.norm(x[i] - x[j], 2))
                # sys.exit()
    return b


def con_matrix_test(x):
    '''

    :param x: based on interaction list
    :return:
    '''
    N = len(x)
    b = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            b[i, j] = np.round(np.linalg.norm(x[i] - x[j], 2), 1)
            b[j, i] = np.round(np.linalg.norm(x[i] - x[j], 2), 1)

    return b


def int_matrix(x, N=10):
    '''
    :param x:  based on interaction list
    :return: get interaction matrix
    '''

    b = np.zeros((N, N))
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            b[i, j] = x[k]
            b[j, i] = x[k]
            k += 1
    return b
