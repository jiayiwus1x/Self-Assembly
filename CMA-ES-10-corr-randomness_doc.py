from __future__ import division

import argparse
import multiprocessing as mp
import os
from datetime import datetime

import cma
import matplotlib.pyplot as plt
import numpy as np

import lib.func as func
import lib.run_hoomd as run_hoomd
from lib import make_target as mt

parser = argparse.ArgumentParser(description='use of the Script')
parser.add_argument('-N', '--N', help='Number of particles', type=int, default=10)
parser.add_argument('-type_names', '--type_names', help='name reference of particle', type=list,
                    default=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                             'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
parser.add_argument('-max_iter', '--max_iter', help='maximum number of iterations', type=int, default=45)
parser.add_argument('-plot_sphere', '--plot_sphere', help='boolean whether or not to plot sphere', type=bool,
                    default=True)
parser.add_argument('-factor', '--factor', help='scalar constant for energy calculation', type=int, default=40)
parser.add_argument('-temp', '--temp', help='temperature level of dynamics', type=float, default=2.)
parser.add_argument('-rmin', '--rmin', help='min distance to interact', type=float, default=0.7)
parser.add_argument('-rmax', '--rmax', help='max distance to interact', type=float, default=1.1)
parser.add_argument('-bs', '--bs', help='box size', type=int, default=10)
args = parser.parse_args()

Timenow = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
Datapath = '/*where you save the data*' + str(Timenow)  # where to save the data
if not os.path.exists(Datapath): os.mkdir(Datapath)  # make the path if it's not exists
print(Datapath)


def use_CMA(x0, max_iter, datapath,
            strs=str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))):
    '''
    function wrapper for CMA-ES optimization
    
    :params x0: initial starting point
    :params max_iter: the maximum number of iterations
    :params datapath: datapath to save results
    :params strs: string of time
    :return: results of optimization, the best solution, and minimum function evaluation achieved
    '''
    d0 = cost_function(x0)

    print('initial condition', x0)
    print('initial function value ', d0)
    print('')
    es = cma.CMAEvolutionStrategy(x0, 2, {'verb_disp': 1})
    i = 0
    num = 28
    f_eval_list = []
    all_para = []
    best_solu = []

    while i < max_iter:

        solutions = es.ask(number=num)

        """ parallelizing function evaluations """

        # define output name
        proc_name = range(num)
        output = mp.Queue()
        seed = np.random.randint(0, 10e8, size=num)
        processes = [mp.Process(target=cost_function,
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


def cost_function(x0, output=[], names='0', seed=np.random.randint(0, 10e8)):
    '''
    objective function
    
    :params x0: initial parameters (potentials)
    :params output: results of optimization (function evaluations)
    :params names: processor name
    :params seed: random seed 
    '''
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

    if not output:
        return f
    else:
        output.put([f, int(names)])


if __name__ == "__main__":
    N = args.N
    type_names = args.type_names[:N]

    tar_con_matrix = func.make_tar(N, type_names, mt.make_pyramid())

    max_iter = args.max_iter

    n = int(N * (N - 1) / 2)
    plot_sphere = args.plot_sphere
    factor = args.factor

    temperature = args.temp

    positions = np.array([np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2)]).T
    print(positions)

    '''parameters for hoomd'''
    rmin = args.rmin
    rmax = args.rmax
    bs = args.bs

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
