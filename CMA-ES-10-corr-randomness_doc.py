from __future__ import division

import multiprocessing as mp
import os
from datetime import datetime

import cma
import matplotlib.pyplot as plt
import numpy as np

import func as func
import make_target as mt
import run_hoomd

Timenow = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
Datapath = '/*where you save the data*' + str(Timenow)  # where to save the data
if not os.path.exists(Datapath): os.mkdir(Datapath)  # make the path if it's not exists
print(Datapath)


def use_CMA(x0, max_iter, datapath,
            strs=str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))):
    d0 = function_2(x0)

    print('initial condition', x0)
    print('initial function value ', d0)
    print('')
    es = cma.CMAEvolutionStrategy(x0, 2, {'verb_disp': 1})
    i = 0
    num = 28
    f_eval_list = []
    all_para = []
    best_solu = []

    # while not es.stop():
    while i < max_iter:

        solutions = es.ask(number=num)

        """ parallelizing function evaluations """

        # define output name
        proc_name = range(num)
        output = mp.Queue()
        seed = np.random.randint(0, 10e8, size=num)
        processes = [mp.Process(target=function_2,
                                args=(solutions[i], output, proc_name[i], seed[i])) for i in range(num)]

        # Run processes
        for p in processes:
            p.start()
            # p.join()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        f_eval_unsort = [output.get() for p in processes]
        f_unsort = np.array(f_eval_unsort)

        seq = list(map(int, f_unsort[:, 1]))
        # print(seq)
        f_eval = f_unsort[:, 0][np.argsort(seq)]

        best_solu.append(solutions[np.argmin(f_eval)])

        f_eval_list.append(f_eval)
        all_para.append(solutions)

        es.tell(solutions, f_eval)
        es.logger.add()
        es.disp()

        i += 1

    es.result_pretty()

    np.save(datapath + '/all_func_val' + strs, f_eval_list)
    np.save(datapath + '/all_para' + strs, all_para)

    np.save(datapath + '/para_w_min_func_val' + strs, best_solu)

    plt.scatter(range(len(f_eval_list)), np.min(f_eval_list, axis=1))
    plt.plot(range(len(f_eval_list)), np.min(f_eval_list, axis=1))  # fs[:,5] is the minimum function val of each

    plt.savefig(datapath + '/' + strs + 'evolution.png')
    plt.close()

    return es.result, best_solu, np.min(f_eval_list, axis=1)


def function_2(x0, output=[], names='0', seed=np.random.randint(0, 10e8)):
    # parameters to run hoomd

    eps = x0.copy()
    eps = eps * factor * temperature

    frames = run_hoomd.run_hoomd(eps, temperature, rmin, rmax, bs, type_names, datapath, names, seed, N=N)

    lhf = len(frames)
    meas = np.zeros(int(lhf / 4))
    k = 0

    for i in range(int(0.75 * lhf), lhf):

        snap = frames[i]

        positions = snap.particles.position
        for j in range(3):
            if max(positions[:, j]) - min(positions[:, j]) > bs / 2:
                positions[:, j] = (positions[:, j] + bs) % bs

        con_matrices = func.con_matrix(positions)
        meas[k] = np.sum(abs(con_matrices - tar_con_matrix)) / 2
        k += 1

    f = np.mean(meas)

    if output == []:
        return f
    else:
        output.put([f, int(names)])


N = 10
type_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'][:N]

# tar 1
tar_con_matrix = func.make_tar(N, type_names, mt.make_pyramid())

# tar 2
# tar_con_matrix2 = func.make_tar(N, type_names, mt.pen())

max_iter = 45

n = int(N * (N - 1) / 2)
plot_sphere = True
factor = 40

temperature = 2.

positions = np.array([np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2)]).T
print(positions)

'''parameters for hoomd'''
rmin = 0.7
rmax = 1.1
bs = 10

f = open(Datapath + "/parameters_run.txt", "w+")
f.write('max iteration for single goal : \r\n' + str(max_iter) + '\n')
f.write('ratio between parameter and temperature : \r\n' + str(factor) + '\n')
f.write('temperature : \r\n' + str(temperature) + '\n')

f.write('number of parameters:\r\n' + str(n) + '\n')
f.write('parameters for hoomd' + '\n')
f.write('r min : \r\n' + str(rmin) + '\n')
f.write('r max : \r\n' + str(rmax) + '\n')
f.write('box size : \r\n' + str(bs) + '\n')

f.close()

allxs = []

x0 = (np.random.rand(n) - 0.5) * 4
allxs.append(x0)
print(np.mean(abs(x0)))

np.save(Datapath + '/x0', x0)

datapath = Datapath + '/tar1_%04d/'
if not os.path.exists(datapath): os.mkdir(datapath)

results = use_CMA(x0, max_iter, factor)
np.save(datapath + '/results', results)
