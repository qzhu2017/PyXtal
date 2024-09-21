import os
import time

import numpy as np
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import read

from pyxtal import pyxtal
from pyxtal.util import good_lattice

"""
A script to perform multistages vasp calculation
"""


class VASP:
    """
    This is a calculator to perform structure optimization in GULP
    At the moment, only inorganic crystal is considered

    Args:

    struc: structure object generated by Pyxtal
    ff: path of forcefield lib
    opt: `conv`, `conp`, `single`
    """

    def __init__(self, struc, path="tmp", cmd="mpirun -np 16 vasp_std"):
        if isinstance(struc, pyxtal):
            struc = struc.to_ase()

        if not isinstance(struc, Atoms):
            raise NotImplementedError("only support ASE atoms object")

        self.structure = struc
        self.folder = path
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.pstress = 0.0
        self.energy = None
        self.energy_per_atom = None
        self.stress = None
        self.forces = None
        self.gap = None
        self.cputime = 0
        self.error = True
        self.cmd = cmd

    def set_vasp(self, level=0, pstress=0.0000, setup=None):
        self.pstress = pstress
        default0 = {
            "xc": "pbe",
            "npar": 8,
            "kgamma": True,
            "lcharg": False,
            "lwave": False,
            "ibrion": 2,
            "pstress": pstress * 10,
            "setups": setup,
        }
        if level == 0:
            default1 = {
                "prec": "low",
                "algo": "normal",
                "kspacing": 0.4,
                "isif": 4,
                "ediff": 1e-2,
                "nsw": 10,
                "potim": 0.02,
            }
        elif level == 1:
            default1 = {
                "prec": "normal",
                "algo": "normal",
                "kspacing": 0.3,
                "isif": 3,
                "ediff": 1e-3,
                "nsw": 25,
                "potim": 0.05,
            }
        elif level == 2:
            default1 = {
                "prec": "accurate",
                "kspacing": 0.2,
                "isif": 3,
                "ediff": 1e-3,
                "nsw": 50,
                "potim": 0.1,
            }
        elif level == 3:
            default1 = {
                "prec": "accurate",
                "encut": 600,
                "kspacing": 0.15,
                "isif": 3,
                "ediff": 1e-4,
                "nsw": 50,
            }
        elif level == 4:
            default1 = {
                "prec": "accurate",
                "encut": 600,
                "kspacing": 0.15,
                "isif": 3,
                "ediff": 1e-4,
                "nsw": 0,
            }

        dict_vasp = dict(default0, **default1)
        return Vasp(**dict_vasp)

    def read_OUTCAR(self, path="OUTCAR"):
        """read time and ncores info from OUTCAR"""
        time = 0
        ncore = 0
        for line in open(path):
            if line.rfind("running on  ") > -1:
                ncore = int(line.split()[2])
            elif line.rfind("Elapsed time ") > -1:
                time = float(line.split(":")[-1])
        self.cputime = time
        self.ncore = ncore

    def read_OSZICAR(self, path="OSZICAR"):
        """read the enthalpy from OSZICAR"""
        energy = 100000
        for line in open(path):
            if line.rfind(" F= ") > -1:
                energy = float(line.split()[2])
        self.energy = energy  # this is actually enthalpy

    def read_bandgap(self, path="vasprun.xml"):
        from vasprun import vasprun

        myrun = vasprun(path)
        self.gap = myrun.values["gap"]

    def run(self, setup=None, pstress=0, level=0, clean=True, read_gap=False, walltime=None):
        if walltime is not None:
            os.environ["VASP_COMMAND"] = "timeout " + max_time + " " + self.cmd
        else:
            os.environ["VASP_COMMAND"] = self.cmd
        cwd = os.getcwd()
        setups = self.set_vasp(level, pstress, setup)
        self.structure.set_calculator(setups)
        os.chdir(self.folder)
        try:
            self.structure.get_potential_energy()
            self.error = False
            self.read_OSZICAR()
        except RuntimeError:
            # VASP is not full done
            self.read_OSZICAR()
            if self.energy < 10000:
                self.error = False
        except (IndexError, ValueError, UnboundLocalError):
            print("Error in parsing vasp output or VASP calc is wrong")
            os.system("cp OUTCAR Error-OUTCAR")

        if not self.error:
            try:
                self.forces = self.structure.get_forces()
            except:
                self.forces = np.zeros([len(self.structure), 3])
            self.energy_per_atom = self.energy / len(self.structure)
            self.read_OUTCAR()
            if read_gap:
                self.read_bandgap()
        if clean:
            self.clean()

        os.chdir(cwd)

    def clean(self):
        os.remove("POSCAR")
        os.remove("POTCAR")
        os.remove("INCAR")
        os.remove("OUTCAR")
        if os.path.exists("OSZICAR"):
            os.remove("OSZICAR")
            os.remove("DOSCAR")
            os.remove("EIGENVAL")
            #os.remove("vasprun.xml")
            #os.remove("vasp.out")
            os.remove("ase-sort.dat")

    def to_pymatgen(self):
        from pymatgen.core.structure import Structure

        return Structure(self.lattice.matrix, self.sites, self.frac_coords)

    def to_pyxtal(self):
        struc = pyxtal()
        struc.from_seed(self.structure)
        return struc


def single_optimize(
    struc,
    level,
    pstress,
    setup,
    path,
    clean,
    cmd="mpirun -np 16 vasp_std",
    walltime="30m",
):
    """
    single optmization

    Args:
        struc: pyxtal structure
        level: vasp calc level
        pstress: external pressure
        setup: vasp setup
        path: calculation directory

    Returns:
        the structure, energy and time costs
    """
    calc = VASP(struc, path, cmd=cmd)
    calc.run(setup, pstress, level, clean=clean)
    if calc.error:
        return None, None, 0, True
    else:
        try:
            struc = calc.to_pyxtal()
            struc.optimize_lattice()
            return struc, calc.energy_per_atom, calc.cputime, calc.error
        except:
            return None, None, 0, True


def single_point(struc, setup=None, path=None, clean=True):
    """
    single optmization

    Args:
        struc: pyxtal structure
        level: vasp calc level
        pstress: external pressure
        setup: vasp setup
        path: calculation directory

    Returns:
        the energy and forces
    """
    calc = VASP(struc, path)
    calc.run(setup, level=4, clean=clean)
    return calc.energy, calc.forces, calc.error


def optimize(
    struc,
    path,
    levels=None,
    pstress=0,
    setup=None,
    clean=True,
    cmd="mpirun -np 16 vasp_std",
    walltime="30m",
):
    """
    multi optimization

    Args:
        struc: pyxtal structure
        path: calculation directory
        levels: list of vasp calc levels
        pstress: external pressure
        setup: vasp setup

    Returns:
        list of structure, energies and time costs
    """

    if levels is None:
        levels = [0, 2, 3]
    time_total = 0
    for _i, level in enumerate(levels):
        struc, eng, time, error = single_optimize(struc, level, pstress, setup, path, clean, cmd, walltime)

        time_total += time
        # print(eng, time, time_total, '++++++++++++++++++++++++++++++')
        if error or not good_lattice(struc):
            return None, None, 0, True
    return struc, eng, time_total, error


def VASP_relax(struc, opt_cell=False, step=100, kspacing=0.25, pstress=0, folder="tmp"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    cwd = os.getcwd()
    os.chdir(folder)

    isif = 3 if opt_cell else 2

    calc = Vasp(
        xc="PBE",
        prec="Accurate",
        npar=8,
        kgamma=True,
        lcharg=False,
        lwave=False,
        ibrion=2,
        pstress=pstress,
        kspacing=kspacing,
        nsw=step,
        isif=isif,
        potim=0.05,
        ediff=1e-3,
    )
    struc.set_calculator(calc)
    energy = struc.get_potential_energy()
    struc = read("CONTCAR", format="vasp")
    os.chdir(cwd)
    return struc, energy


if __name__ == "__main__":
    while True:
        struc = pyxtal()
        struc.from_random(3, 19, ["C"], [4])
        if struc.valid:
            break

    # set up the commands
    os.system("source /share/intel/mkl/bin/mklvars.sh intel64")
    cmd = "mpirun -n 4 /share/apps/bin/vasp544-2019u2/vasp_std"

    calc = VASP(struc, path="tmp", cmd=cmd)
    calc.run()
    print("Energy:", calc.energy)
    print("Forces", calc.forces)

    struc, eng, time, _ = optimize(struc, path="tmp", levels=[0, 1, 2], cmd=cmd, walltime="30s")
    print(struc)
    print("Energy:", eng)
    print("Time:", time)

    calc = VASP(struc, path="tmp", cmd=cmd)
    calc.run(level=4, read_gap=True)
    print("Energy:", calc.energy)
    print("Gap:", calc.gap)
