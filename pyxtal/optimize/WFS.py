"""
global optimizer
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

import numpy as np

from numpy.random import Generator
from pymatgen.analysis.structure_matcher import StructureMatcher

from pyxtal.optimize.base import GlobalOptimize
from pyxtal.representation import representation

if TYPE_CHECKING:
    from pyxtal.lattice import Lattice
    from pyxtal.molecule import pyxtal_molecule


class WFS(GlobalOptimize):
    """
    Standard Population algorithm

    Args:
        smiles (str): smiles string
        workdir (str): path of working directory
        sg (int or list): space group number or list of spg numbers
        tag (string): job prefix
        ff_opt (bool): activate on the fly FF mode
        ff_style (str): automated force style (`gaff` or `openff`)
        ff_parameters (str or list): ff parameter xml file or list
        reference_file (str): path of reference xml data for FF training
        N_gen (int): number of generation (default: `10`)
        N_pop (int): number of populations (default: `10`)
        fracs (list): fractions for each variation (default: `[0.5, 0.5, 0.0]`)
        N_cpu (int): number of cpus for parallel calculation (default: `1`)
        cif (str): cif file name to store all structure information
        block: block mode
        num_block: list of blocks
        compositions: list of composition, (default is [1]*Num_mol)
        lattice (bool): whether or not supply the lattice
        torsions: list of torsion angle
        molecules (list): list of pyxtal_molecule objects
        sites (list): list of wp sites, e.g., [['4a']]
        use_hall (bool): whether or not use hall number (default: False)
        skip_ani (bool): whether or not use ani or not (default: True)
        eng_cutoff (float): the cutoff energy for FF training
        E_max (float): maximum energy defined as an invalid structure
        verbose (bool): show more details
        matcher : structurematcher from pymatgen
        early_quit: whether quit the program early when the target is found
    """

    def __init__(
        self,
        smiles: str,
        workdir: str,
        sg: int | list,
        tag: str,
        info: dict[any, any] | None = None,
        ff_opt: bool = False,
        ff_style: str = "openff",
        ff_parameters: str = "parameters.xml",
        reference_file: str = "references.xml",
        ref_criteria: dict[any, any] | None = None,
        N_gen: int = 10,
        N_pop: int = 10,
        fracs: list | None = None,
        N_cpu: int = 1,
        cif: str | None = None,
        block: list[any] | None = None,
        num_block: list[any] | None = None,
        composition: list[any] | None = None,
        lattice: Lattice | None = None,
        torsions: list[any] | None = None,
        molecules: list[pyxtal_molecule] | None = None,
        sites: list[any] | None = None,
        use_hall: bool = False,
        skip_ani: bool = True,
        factor: float = 1.1,
        eng_cutoff: float = 5.0,
        E_max: float = 1e10,
        verbose: bool = False,
        random_state: int | None = None,
        max_time: float | None = None,
        matcher: StructureMatcher | None = None,
        early_quit: bool = True,
        check_stable: bool = False,
    ):
        if isinstance(random_state, Generator):
            self.random_state = random_state.spawn(1)[0]
        else:
            self.random_state = np.random.default_rng(random_state)

        # POPULATION parameters:
        if fracs is None:
            fracs = [0.6, 0.4, 0.0]
        self.N_gen = N_gen
        self.N_pop = N_pop
        self.fracs = np.array(fracs)
        self.verbose = verbose

        # initialize other base parameters
        GlobalOptimize.__init__(
            self,
            smiles,
            workdir,
            sg,
            tag,
            info,
            ff_opt,
            ff_style,
            ff_parameters,
            reference_file,
            ref_criteria,
            N_cpu,
            cif,
            block,
            num_block,
            composition,
            lattice,
            torsions,
            molecules,
            sites,
            use_hall,
            skip_ani,
            factor,
            eng_cutoff,
            E_max,
            random_state,
            max_time,
            matcher,
            early_quit,
            check_stable,
        )

        strs = self.full_str()
        self.logging.info(strs)
        print(strs)

    def full_str(self):
        s = str(self)
        s += "\nMethod    : Stochastic Width First Sampling"
        s += f"\nGeneration: {self.N_gen:4d}"
        s += f"\nPopulation: {self.N_pop:4d}"
        s += "\nFraction  : {:4.2f} {:4.2f} {:4.2f}".format(*self.fracs)
        # The rest base information from now on
        return s

    def run(self, ref_pmg=None, ref_eng=None, ref_pxrd=None):
        """
        The main code to run WFS prediction

        Args:
            ref_pmg: reference pmg structure
            ref_eng: reference energy
            ref_pxrd: reference pxrd profile in 2D array

        Returns:
            success_rate or None
        """
        if ref_pmg is not None:
            ref_pmg.remove_species("H")

        # Related to the FF optimization
        N_added = 0
        success_rate = 0

        for gen in range(self.N_gen):
            print(f"\nGeneration {gen:d} starts")
            self.logging.info(f"Generation {gen:d} starts")
            self.generation = gen + 1
            t0 = time()

            # lists for structure information
            current_reps = [None] * self.N_pop
            current_matches = [False] * self.N_pop if ref_pxrd is None else [0.0] * self.N_pop
            current_engs = [self.E_max] * self.N_pop
            current_xtals = [None] * self.N_pop
            current_tags = ["Random"] * self.N_pop

            # Set up the origin for new structures
            if gen > 0:
                N_pops = [int(self.N_pop * i) for i in self.fracs]
                count = N_pops[0]

                for _sub_pop in range(N_pops[1]):
                    id = self._selTournament(engs)
                    xtal = prev_xtals[id]
                    current_tags[count] = "Mutation"
                    if xtal is None:
                        print(id)
                        print(len(engs))
                        print(engs)
                        print(len(prev_xtals))
                        print(prev_xtals)
                        raise ValueError("Problem in selection")
                    current_xtals[count] = xtal
                    count += 1

                # Not Working
                for _sub_pop in range(N_pops[2]):
                    xtal1 = prev_xtals[self.selTournament(engs)]
                    xtal2 = prev_xtals[self.selTournament(engs)]
                    current_tags[count] = "Crossover"
                    current_xtals[count] = self._crossover(xtal1, xtal2)
                    count += 1

            # Local optimization
            gen_results = self.local_optimization(gen, current_xtals, ref_pmg, ref_pxrd)

            # Summary and Ranking
            for id, res in enumerate(gen_results):
                (xtal, match) = res

                if xtal is not None:
                    current_xtals[id] = xtal
                    rep = xtal.get_1D_representation()
                    current_reps[id] = rep.x
                    current_engs[id] = xtal.energy / sum(xtal.numMols)
                    current_matches[id] = match

                    # Don't write bad structure
                    if self.cif is not None and xtal.energy < 9999:
                        if self.verbose:
                            print("Add qualified structure", id, xtal.energy)
                        with open(self.workdir + "/" + self.cif, "a+") as f:
                            label = self.tag + "-g" + str(gen) + "-p" + str(id)
                            f.writelines(xtal.to_file(header=label))
                        # else:
                        #    print("Neglect bad structure", id, xtal.energy)
                    self.engs.append(xtal.energy / sum(xtal.numMols))
                    # print(output)

            strs = f"Generation {gen:d} finishes"  # ; import sys; sys.exit()
            print(strs)
            self.logging.info(strs)
            t1 = time()

            # Apply Gaussian
            if ref_pxrd is None:
                engs = self._apply_gaussian(current_reps, current_engs)
            else:
                engs = self._apply_gaussian(current_reps, -1 * np.array(current_matches))

            # Store the best structures
            count = 0
            xtals = []
            ids = np.argsort(engs)
            for id in ids:
                xtal = current_xtals[id]
                rep = current_reps[id]
                eng = current_engs[id]
                tag = current_tags[id]
                if self.new_struc(xtal, xtals):
                    xtals.append(xtal)
                    self.best_reps.append(rep)
                    d_rep = representation(rep, self.smiles)
                    try:
                        strs = d_rep.to_string(None, eng, tag)
                        out = f"{gen:3d} {strs:s} Top"
                        if ref_pxrd is not None:
                            out += f" {current_matches[id]:6.3f}"
                        print(out)
                    except:
                        print('Error', xtal)
                    count += 1
                if count == 3:
                    break

            t2 = time()
            gen_out = f"Gen{gen:3d} time usage: "
            gen_out += f"{t1 - t0:5.1f}[Calc] {t2 - t1:5.1f}[Proc]"
            print(gen_out)

            # Save the reps for next move
            prev_xtals = current_xtals  # ; print(self.engs)
            self.min_energy = np.min(np.array(self.engs))
            self.N_struc = len(self.engs)

            # Update the FF parameters if necessary
            if self.ff_opt:
                N_max = min([int(self.N_pop * 0.6), 50])
                ids = np.argsort(engs)
                xtals = self.select_xtals(current_xtals, ids, N_max)
                print("Select Good structures for FF optimization", len(xtals))
                N_added = self.ff_optimization(xtals, N_added)
            else:
                success_rate = self.success_count(gen, current_xtals, current_matches, current_tags, ref_pmg)
                gen_out = f"Success rate at Gen {gen:3d}: {success_rate:7.4f}%"
                self.logging.info(gen_out)
                print(gen_out)

                if self.early_termination(success_rate):
                    return success_rate

        return success_rate

    def _selTournament(self, fitness, factor=0.35):
        """
        Select the best individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        """
        IDs = self.random_state.choice(len(fitness), size=int(len(fitness) * factor), replace=False)
        min_fit = np.argmin(fitness[IDs])
        return IDs[min_fit]

    def _crossover(self, x1, x2):
        """
        How to design this?
        """
        raise NotImplementedError


if __name__ == "__main__":
    import argparse
    import os

    from pyxtal.db import database

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gen",
        dest="gen",
        type=int,
        default=10,
        help="Number of generation, optional",
    )
    parser.add_argument(
        "-p",
        "--pop",
        dest="pop",
        type=int,
        default=10,
        help="Population size, optional",
    )
    parser.add_argument("-n", "--ncpu", dest="ncpu", type=int, default=1, help="cpu number, optional")
    parser.add_argument("--ffopt", action="store_true", help="enable ff optimization")

    options = parser.parse_args()
    gen = options.gen
    pop = options.pop
    ncpu = options.ncpu
    ffopt = options.ffopt
    db_name, name = "pyxtal/database/test.db", "ACSALA"
    wdir = name
    os.makedirs(wdir, exist_ok=True)
    os.makedirs(wdir + "/calc", exist_ok=True)

    db = database(db_name)
    row = db.get_row(name)
    xtal = db.get_pyxtal(name)
    smile, wt, spg = row.mol_smi, row.mol_weight, row.space_group.replace(" ", "")
    chm_info = None
    if not ffopt:
        if "charmm_info" in row.data:
            # prepare charmm input
            chm_info = row.data["charmm_info"]
            with open(wdir + "/calc/pyxtal.prm", "w") as prm:
                prm.write(chm_info["prm"])
            with open(wdir + "/calc/pyxtal.rtf", "w") as rtf:
                rtf.write(chm_info["rtf"])
        else:
            # Make sure we generate the initial guess from ambertools
            if os.path.exists("parameters.xml"):
                os.remove("parameters.xml")
    # load reference xtal
    pmg0 = xtal.to_pymatgen()
    if xtal.has_special_site(): xtal = xtal.to_subgroup()
    N_torsion = xtal.get_num_torsions()

    # GO run
    t0 = time()
    go = WFS(
        smile,
        wdir,
        xtal.group.number,
        name.lower(),
        info=chm_info,
        ff_style="openff",  #'gaff',
        ff_opt=ffopt,
        N_gen=gen,
        N_pop=pop,
        N_cpu=ncpu,
        cif="pyxtal.cif",
    )

    suc_rate = go.run(pmg0)
    print(f"CSD {name:s} in Gen {go.generation:d}")

    if len(go.matches) > 0:
        best_rank = go.print_matches()
        mytag = f"True {best_rank:d}/{go.N_struc:d} Succ_rate: {suc_rate:7.4f}%"
    else:
        mytag = f"False 0/{go.N_struc:d}"

    eng = go.min_energy
    t1 = int((time() - t0)/60)
    strs = "Final {:8s} [{:2d}]{:10s} ".format(name, sum(xtal.numMols), spg)
    strs += "{:3d}m {:2d} {:6.1f}".format(t1, N_torsion, wt)
    strs += "{:12.3f} {:20s} {:s}".format(eng, mytag, smile)
    print(strs)
