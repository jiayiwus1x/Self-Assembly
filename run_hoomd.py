import gsd.hoomd
import hoomd
import hoomd.md as md
import numpy as np


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


def run_hoomd(eps, temperature, rmin, rmax, bs, type_names, datapath, pro_name='0', ran_seed=42, run_time=4e6, N=10):
    hoomd.context.initialize("");
    hoomd.option.set_notice_level(0)
    # timenow = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    # ra = generate_str(4)
    # type_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J','K','L','M','N','O']
    positions = np.array([np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2), np.arange(-N / 2, N / 2)]).T

    uc = hoomd.lattice.unitcell(N=len(type_names),
                                a1=[10., 0, 0],
                                a2=[0, 10., 0],
                                a3=[0, 0, 10.],
                                position=positions,
                                dimensions=3,
                                type_name=type_names);

    system = hoomd.init.create_lattice(unitcell=uc, n=[1, 1, 1]);
    nl = hoomd.md.nlist.cell()

    table = md.pair.table(width=1000, nlist=nl)

    # sys.exit()
    num = 0
    k = 0
    for i in type_names:

        num += 1
        table.pair_coeff.set(i, i, func=dis_int, rmin=rmin, rmax=rmax, coeff=dict(epsilon=0))
        for j in type_names[-len(type_names) + num:]:
            if -len(type_names) + num != 0:
                table.pair_coeff.set(i, j, func=dis_int, rmin=rmin, rmax=rmax, coeff=dict(epsilon=eps[k]))
                k += 1
                # print(i,j,k)

    hoomd.update.box_resize(L=bs)
    hoomd.md.integrate.mode_standard(dt=0.001);
    all = hoomd.group.all();

    # print(ran_seed)
    hoomd.md.integrate.langevin(group=all, kT=temperature, seed=ran_seed);

    if pro_name == '0':
        hoomd.analyze.log(filename=datapath + "/log-output.log",
                          quantities=['potential_energy', 'kinetic_energy',
                                      'temperature'],
                          period=int(run_time / 200),
                          overwrite=True);
    hoomd.dump.gsd(datapath + "/trajectory_" + str(pro_name) + ".gsd",
                   period=int(run_time / 200),
                   group=hoomd.group.all(),
                   overwrite=True);
    log = hoomd.analyze.log(filename=None, quantities=['potential_energy', "kinetic_energy"],
                            period=int(run_time / 200))
    hoomd.run(run_time, quiet=True);

    # U = log.query('potential_energy')
    # print(U)

    frames = gsd.hoomd.open(name=datapath + "/trajectory_" + str(pro_name) + ".gsd", mode='rb')

    # find minimum point
    return frames
