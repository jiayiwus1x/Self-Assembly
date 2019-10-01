# Self assembly of a single structure 

<p align="center">
  <img width="650"  src="self_assembly.gif">
</p>

## Description

Ever wonder how the <a href=https://www.youtube.com/watch?v=ep2-W1X65KI>Microbots</a> in Big Hero 6 knew how to assemble into all those different structures? In this project, we start with a bunch of particles in a periodic box that interact with a certain <href=https://en.wikipedia.org/wiki/Lennard-Jones_potential>Lennard-Jones potential</a>. We train the particles to adjust their potential so that they if you shake the box hard enough, they eventually form a Pyramid. You can change the ending structure into anything you want.

## Prerequire package

<i>Shaking the box</i> is really just <a href=https://en.wikipedia.org/wiki/Langevin_dynamics>Langevin dynamics</a>. We use Hoomd to handle the molecular dynamics. Make sure to download and read through a basic tutorial for Hoomd:
[Hoomd-Blue](http://glotzerlab.engin.umich.edu/hoomd-blue/)

<i>Training</i> is done using <a href=https://pypi.org/project/cma/)>CMA-ES</a> which you can install as follows:

```bash
pip install cma
```

## Usage



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Futher reading


