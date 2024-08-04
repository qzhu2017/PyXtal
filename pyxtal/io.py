"""
This module handles reading and write crystal files.
"""

import importlib.resources

import numpy as np
from monty.serialization import loadfn
from pymatgen.core.bonds import CovalentBond
from pymatgen.core.structure import Molecule, Structure

from pyxtal.constants import logo
from pyxtal.lattice import Lattice
from pyxtal.molecule import Orientation, compare_mol_connectivity, pyxtal_molecule
from pyxtal.msg import ReadSeedError
from pyxtal.symmetry import Group
from pyxtal.util import get_symmetrized_pmg
from pyxtal.wyckoff_site import atom_site, mol_site

with importlib.resources.as_file(importlib.resources.files("pyxtal") / "database" / "bonds.json") as path:
    bonds = loadfn(path)


def in_merged_coords(wp, pt, pts, cell):
    """
    Whether or not the pt in within the pts
    """
    (c, s) = pt
    for pt0 in pts:
        (c0, s0) = pt0
        if s == s0 and wp.are_equivalent_pts(c, c0, cell):
            # print(c, c0, 'equivalent')
            return True
    return False


def get_cif_str_for_pyxtal(struc, header: str = "", sym_num=None, style: str = "mp"):
    """Get the cif string for a given structure. The default setting for
    _atom_site follows the materials project cif

    TODO make this a method of the pyxtal class

    Args:
        struc: pyxtal structure object
        header: additional information
        sym_num: the number of symmetry operations, None means writing all symops
        style: `icsd` or `mp` (used in pymatgen)
    """
    if struc.molecular:
        sites = struc.mol_sites
        molecule = True
        # special = struc.has_special_site()
        # print('===============================================', struc)
    else:
        sites = struc.atom_sites
        molecule = False

    if sym_num is None:
        l_type = struc.group.lattice_type
        number = struc.group.number
        G1 = struc.group[0]
        symbol = struc.group.symbol if G1.is_standard_setting() else sites[0].wp.get_hm_symbol()

    else:  # P1 symmetry
        l_type = "triclinic"
        symbol = "P1"
        number = 1
        G1 = Group(1).Wyckoff_positions[0]

    lines = logo
    lines += "data_" + header + "\n"
    if hasattr(struc, "energy"):
        eng = struc.energy / sum(struc.numMols) if struc.molecular else struc.energy / sum(struc.numIons)
        lines += f"#Energy: {eng} eV/cell\n"

    lines += f"\n_symmetry_space_group_name_H-M '{symbol:s}'\n"
    lines += f"_symmetry_Int_Tables_number      {number:>15d}\n"
    lines += f"_symmetry_cell_setting           {l_type:>15s}\n"

    a, b, c, alpha, beta, gamma = struc.lattice.get_para(degree=True)
    lines += f"_cell_length_a        {a:12.6f}\n"
    lines += f"_cell_length_b        {b:12.6f}\n"
    lines += f"_cell_length_c        {c:12.6f}\n"
    lines += f"_cell_angle_alpha     {alpha:12.6f}\n"
    lines += f"_cell_angle_beta      {beta:12.6f}\n"
    lines += f"_cell_angle_gamma     {gamma:12.6f}\n"
    lines += f"_cell_volume          {struc.lattice.volume:12.6f}\n"
    # if struc.molecular:
    #    lines += '_cell_formula_units_Z     {:d}\n'.format(sum(struc.numMols))
    # else:
    #    lines += '_cell_formula_units_Z     {:d}\n'.format(sum(struc.numIons))

    lines += "\nloop_\n"
    lines += " _symmetry_equiv_pos_site_id\n"
    lines += " _symmetry_equiv_pos_as_xyz\n"

    for i, op in enumerate(G1):
        lines += f"{i + 1:d} '{op.as_xyz_str():s}'\n"

    lines += "\nloop_\n"
    lines += " _atom_site_label\n"
    lines += " _atom_site_type_symbol\n"
    lines += " _atom_site_symmetry_multiplicity\n"
    if style == "icsd":
        lines += " _atom_site_Wyckoff_symbol\n"
    lines += " _atom_site_fract_x\n"
    lines += " _atom_site_fract_y\n"
    lines += " _atom_site_fract_z\n"
    lines += " _atom_site_occupancy\n"

    for site in sites:
        mul = site.wp.multiplicity
        letter = site.wp.letter
        if molecule:
            if sym_num is None:
                coord0s, specie0s = site._get_coords_and_species(first=True)
                if site.wp.index > 0:
                    # print("#Check if the mul is consistent!", site.wp.index)
                    muls = []
                    coords = []
                    species = []
                    merges = []

                    for coord, specie in zip(coord0s, specie0s):
                        _, wp, _ = G1.merge(coord, struc.lattice.matrix, 0.05)
                        if len(wp) > mul:
                            if not in_merged_coords(G1, [coord, specie], merges, struc.lattice.matrix):
                                # print("General Position", specie, coord)
                                coords.append(coord)
                                species.append(specie)
                                muls.append(len(wp))
                                merges.append((coord, specie))
                        else:
                            # print("Special Position", specie, coord)
                            coords.append(coord)
                            species.append(specie)
                            muls.append(mul)

                else:
                    coords, species = coord0s, specie0s
                    muls = [mul] * len(coords)
            else:
                coords = None
                species = []
                for id in range(sym_num):
                    mol = site.get_mol_object(id)
                    tmp = mol.cart_coords.dot(site.lattice.inv_matrix)
                    coords = tmp if coords is None else np.append(coords, tmp, axis=0)
                    species.extend([s.value for s in mol.species])
                muls = [mul] * len(coords)
                # coords, species = site._get_coords_and_species(ids=sym_num)
        else:
            coords, species, muls = [site.position], [site.specie], [mul]

        for specie, coord, mul in zip(species, coords, muls):
            lines += f"{specie:6s} {specie:6s} {mul:3d} "
            if style != "mp":
                lines += f"{letter:s} "
            lines += "{:12.6f}{:12.6f}{:12.6f} 1\n".format(*coord)
    lines += "#END\n\n"

    return lines


def write_cif(struc, filename=None, header="", permission="w", sym_num=None, style="mp"):
    """
    Export the structure in cif format
    The default setting for _atom_site follows the materials project cif

    Args:
        struc: pyxtal structure object
        filename: path of the structure file
        header: additional information
        permission: write(`w`) or append(`a+`) to the given file
        sym_num: the number of symmetry operations, None means writing all symops
        style: `icsd` or `mp` (used in pymatgen)

    """
    lines = get_cif_str_for_pyxtal(struc, header=header, sym_num=sym_num, style=style)

    if filename is None:
        return lines
    else:
        with open(filename, permission) as f:
            f.write(lines)
        return None


def read_cif(filename):
    """
    read the cif, mainly for pyxtal cif output
    Be cautious in using it to read other cif files

    Args:
        filename: path of the structure file

    Return:
        pyxtal structure
    """
    species = []
    coords = []
    with open(filename) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("_symmetry_Int_Tables_number"):
                sg = int(line.split()[-1])
            elif line.startswith("_cell_length_a"):
                a = float(lines[i].split()[-1])
                b = float(lines[i + 1].split()[-1])
                c = float(lines[i + 2].split()[-1])
                alpha = float(lines[i + 3].split()[-1])
                beta = float(lines[i + 4].split()[-1])
                gamma = float(lines[i + 5].split()[-1])
            elif line.startswith("_symmetry_cell_setting"):
                lat_type = line.split()[-1]
            elif line.startswith("_symmetry_space_group_name_H-M "):
                symbol = line.split()[-1]
                diag = eval(symbol) in ["Pn", "P21/n", "C2/n"]

            elif line.find("_atom_site") >= 0:
                s = i
                while True:
                    s += 1
                    if lines[s].find("_atom_site") >= 0:
                        pass
                    elif len(lines[s].split()) <= 3:
                        break
                    else:
                        tmp = lines[s].split()
                        pos = [float(tmp[-4]), float(tmp[-3]), float(tmp[-2])]
                        species.append(tmp[0])
                        coords.append(pos)
                break

    wp0 = Group(sg)[0]
    lattice = Lattice.from_para(a, b, c, alpha, beta, gamma, lat_type)
    sites = []
    for specie, coord in zip(species, coords):
        pt, wp, _ = wp0.merge(coord, lattice.matrix, tol=0.1)
        sites.append(atom_site(wp, pt, specie, diag))
    return lattice, sites


class structure_from_ext:
    def __init__(self, struc, ref_mols, tol=0.2, ignore_HH=False, add_H=False, hn=None):
        """
        extract the mol_site information from the give cif file
        and reference molecule

        Args:
            struc: cif/poscar file or a Pymatgen Structure object
            ref_mols: a list of reference molecule (xyz file or Pyxtal molecule)
            tol: scale factor for covalent bond distance
            ignore_HH: whether or not ignore short H-H in checking molecule
            add_H: whether or not add the H atoms
        """

        for i, ref_mol in enumerate(ref_mols):
            if isinstance(ref_mol, str):
                ref_mols[i] = pyxtal_molecule(ref_mol, fix=True)
            elif isinstance(ref_mol, pyxtal_molecule):
                continue
            else:
                print(type(ref_mol))
                raise NameError(f"reference molecule of type {type(ref_mol)} cannot be defined")

        if isinstance(struc, str):
            pmg_struc = Structure.from_file(struc)
        elif isinstance(struc, Structure):
            pmg_struc = struc
        else:
            print(type(struc))
            raise NameError("input structure cannot be intepretted")

        # reset the hydrogen position
        if add_H:
            pmg_struc.remove_species("H")

        self.ref_mols = ref_mols
        self.tol = tol
        self.add_H = add_H

        sym_struc, number = get_symmetrized_pmg(pmg_struc, hn=hn)
        group = Group(number) if hn is None else Group(hn, use_hall=True)

        self.group = group
        self.wyc = group[0]

        molecules = search_molecules_in_crystal(sym_struc, self.tol, ignore_HH=ignore_HH)

        self.pmg_struc = sym_struc
        matrix = sym_struc.lattice.matrix
        ltype = group.lattice_type
        self.lattice = Lattice.from_matrix(matrix, ltype=ltype)
        self.resort(molecules)
        if len(self.ids) == 0:
            raise RuntimeError("Cannot extract molecules")

    def resort(self, molecules):
        from pyxtal.operations import apply_ops, find_ids

        # filter out the molecular generators
        inv_lat = self.pmg_struc.lattice.inv_matrix
        new_lat = self.lattice.matrix
        positions = np.zeros([len(molecules), 3])
        for i in range(len(molecules)):
            positions[i] = np.dot(molecules[i].cart_coords.mean(axis=0), inv_lat)

        wps = []
        ids = []  # id for the generator
        visited_ids = []
        for id, pos in enumerate(positions):
            if id not in visited_ids:
                centers = apply_ops(pos, self.wyc)
                tmp_ids = find_ids(centers, positions)
                visited_ids.extend(tmp_ids)
                # print(id, pos, tmp_ids, len(self.wyc), len(molecules[id]))
                if len(tmp_ids) == len(self.wyc):
                    # general position
                    # if len(molecules[id])==1: print("groups", tmp_ids, '\n', centers)
                    wps.append(self.wyc)
                    ids.append(id)
                else:  # special sites
                    for id0 in tmp_ids:
                        p0 = positions[id0]
                        p1, wp, _ = self.wyc.merge(p0, new_lat, 0.1)
                        diff = p1 - p0
                        diff -= np.rint(diff)
                        if np.abs(diff).sum() < 1e-2:  # sort position by mapping
                            wps.append(wp)
                            ids.append(id0)  # find the right ids
                            # print("add special", wp.index, id0)
                            break
        # print("===============================================================", self.wps)

        # add position and molecule, print("ids", ids, mults)
        len(ids)
        self.numMols = [0] * len(self.ref_mols)
        self.positions = []
        self.wps = []
        self.p_mols = []
        self.ids = []
        ids_done = []

        # search for the matched molecules
        for j, mol2_ref in enumerate(self.ref_mols):
            mol2 = mol2_ref.copy()
            if self.add_H:
                mol2.mol.remove_species("H")

            for i, id in enumerate(ids):
                mol1 = molecules[id]
                # print("++++++++++++++++++++++++++", id, ids, len(mol2.mol), len(mol1))
                # print(mol2.mol.to('xyz'))
                if id not in ids_done and len(mol2.mol) == len(mol1):
                    p_mol = mol2_ref.copy()  # create p_mol
                    match, mapping = compare_mol_connectivity(mol2.mol, mol1)
                    if match:
                        if len(mol1) > 1:
                            # rearrange the order
                            order = [mapping[at] for at in range(len(mol1))]
                            xyz = mol1.cart_coords[order]
                            # add hydrogen positions here
                            if self.add_H:
                                # print(mol2.smile)
                                xyz = self.add_Hydrogens(mol2.smile, xyz)
                            # print(xyz)
                            frac = np.dot(xyz, inv_lat)
                            xyz = np.dot(frac, new_lat)
                            center = p_mol.get_center(xyz)
                            p_mol.reset_positions(xyz - center)
                            position = np.dot(center, np.linalg.inv(new_lat))
                        else:
                            xyz = mol1.cart_coords[0]
                            position = np.dot(xyz, inv_lat)
                        position -= np.floor(position)

                        self.positions.append(position)
                        self.p_mols.append(p_mol)
                        self.ids.append(j)
                        ids_done.append(id)

                        self.wps.append(wps[i])
                        self.numMols[j] += len(wps[i])
                        # print("================================================ADDDDDD", id, len(mol1))

        # check if some molecules cannot be matched
        if len(ids_done) < len(ids):
            # print("==========================================================Nonmatched molecules", ids_done, ids)
            for id in ids:
                if id not in ids_done:
                    msg = "This molecule cannot be matched to the reference\n"
                    msg += "Molecules extracted from the structure\n"
                    msg += molecules[id].to(fmt="xyz") + "\n"
                    msg += "Reference molecule from smiles or xyz\n"
                    msg += mol2.mol.to(fmt="xyz")
                    raise ReadSeedError(msg)

    def add_Hydrogens(self, smile, xyz):
        """
        add hydrogen for pymtagen molecule
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Geometry import Point3D

        # print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS", smile)
        m1 = Chem.MolFromSmiles(smile)
        m2 = Chem.AddHs(m1)
        if len(smile) > 100:
            AllChem.EmbedMolecule(m2, randomSeed=3)
        else:
            AllChem.EmbedMolecule(m2, randomSeed=0xF00D)
        m2 = Chem.RemoveHs(m2)
        conf = m2.GetConformer(0)

        for i in range(conf.GetNumAtoms()):
            x, y, z = xyz[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))
        # Chem.MolToPDBFile(m2, 'test.pdb')
        # conf = m2.GetConformer(0); print(conf.GetPositions())

        m1 = Chem.AddHs(m1)
        if len(smile) > 100:
            AllChem.EmbedMolecule(m1, randomSeed=3)
        else:
            AllChem.EmbedMolecule(m1, randomSeed=0xF00D)

        # print(m1.GetNumAtoms(), m2.GetNumAtoms())
        AllChem.UFFOptimizeMolecule(m1)
        try:
            m3 = AllChem.ConstrainedEmbed(m1, m2)
        except ValueError as e:
            raise ReadSeedError(str(e))

        conf = m3.GetConformer(0)  # ; print(conf.GetPositions())

        return conf.GetPositions()

    def make_mol_sites(self):
        """
        generate the molecular wyckoff sites
        """
        ori = Orientation(np.eye(3))
        sites = []
        for id, mol, pos, wp in zip(self.ids, self.p_mols, self.positions, self.wps):
            # print(id, mol.smile, wp.multiplicity)
            site = mol_site(mol, pos, ori, wp, self.lattice)
            site.type = id
            # print(pos)
            # print(self.lattice.matrix)
            # print([a.value for a in site.molecule.mol.species])
            # print(site.molecule.mol.cart_coords)
            # print(site._get_coords_and_species(absolute=True)[0][:10])
            sites.append(site)
        return sites

    def align(self):
        """
        compute the orientation wrt the reference molecule
        """
        try:
            from openbabel import openbabel, pybel
        except:
            import openbabel
            import pybel

        m1 = pybel.readstring("xyz", self.ref_mol.to("xyz"))
        m2 = pybel.readstring("xyz", self.molecule.to("xyz"))
        aligner = openbabel.OBAlign(True, False)
        aligner.SetRefMol(m1.OBMol)
        aligner.SetTargetMol(m2.OBMol)
        aligner.Align()
        print("RMSD: ", aligner.GetRMSD())
        rot = np.zeros([3, 3])
        for i in range(3):
            for j in range(3):
                rot[i, j] = aligner.GetRotMatrix().Get(i, j)
        coord2 = self.molecule.cart_coords
        coord2 -= np.mean(coord2, axis=0)
        coord3 = rot.dot(coord2.T).T + np.mean(self.ref_mol.cart_coords, axis=0)
        self.mol_aligned = Molecule(self.ref_mol.atomic_numbers, coord3)
        self.ori = Orientation(rot)

    def show(self, overlay=True):
        from pyxtal.viz import display_molecules

        if overlay:
            return display_molecules([self.ref_mol, self.mol_aligned])
        else:
            return display_molecules([self.ref_mol, self.molecule])


def search_molecules_in_crystal(struc, tol=0.2, once=False, ignore_HH=True, max_bond_length=None):
    """
    Function to perform to find the molecule in a Pymatgen structure

    Args:
        struc: Pymatgen Structure
        tol: tolerance value to check the connectivity
        once: search only one molecule or all molecules
        ignore_HH: whether or not ignore the short H-H in checking molecule
        max_bond_length: sets maximum bond length if bond length is missing in bond length database

    Returns:
        molecules: list of pymatgen molecules
        positions: list of center positions
    """

    def check_one_layer(struc, sites0, visited):
        new_members = []
        for site0 in sites0:
            sites_add, visited = check_one_site(struc, site0, visited)
            new_members.extend(sites_add)
        return new_members, visited

    def check_one_site(struc, site0, visited, rmax=2.8):
        neigh_sites = struc.get_neighbors(site0, rmax)
        ids = [m.index for m in visited]
        sites_add = []
        ids_add = []
        pbc = isinstance(struc, Structure)

        for site1 in neigh_sites:
            if site1.index not in ids + ids_add:
                try:
                    if CovalentBond.is_bonded(site0, site1, tol):
                        if pbc:
                            (d, image) = site0.distance_and_image(site1)
                        else:
                            d = site0.distance(site1)
                        val1, val2 = site1.specie.value, site0.specie.value
                        key = f"{val1:s}-{val2:s}"

                        # sometime the H-H short distance is not avoidable
                        if key == "H-H":
                            if not ignore_HH:
                                if pbc:
                                    site1.frac_coords += image
                                sites_add.append(site1)
                                ids_add.append(site1.index)
                        else:
                            if d < bonds.get(key,max_bond_length):
                                if pbc:
                                    site1.frac_coords += image
                                sites_add.append(site1)
                                ids_add.append(site1.index)
                except ValueError:
                    # QZ: use our own bond distance lib
                    if pbc:
                        (d, image) = site0.distance_and_image(site1)
                    else:
                        d = site0.distance(site1)

                    val1, val2 = site1.specie.value, site0.specie.value
                    key = f"{val1:s}-{val2:s}"
                    if d < bonds[key]:
                        if pbc:
                            site1.frac_coords += image
                        sites_add.append(site1)
                        ids_add.append(site1.index)

        if len(sites_add) > 0:
            visited.extend(sites_add)

        return sites_add, visited

    molecules = []
    visited_ids = []

    for id, site in enumerate(struc.sites):
        if id not in visited_ids:
            first_site = site
            visited = [first_site]
            first_site.index = id
            n_iter, max_iter = 0, len(struc) - len(visited_ids)
            while n_iter < max_iter:
                if n_iter == 0:
                    new_sites, visited = check_one_site(struc, first_site, visited)
                else:
                    new_sites, visited = check_one_layer(struc, new_sites, visited)
                n_iter += 1
                if len(new_sites) == 0:
                    break

            coords = [s.coords for s in visited]
            coords = np.array(coords)
            numbers = [s.specie.number for s in visited]
            molecules.append(Molecule(numbers, coords))
            visited_ids.extend([s.index for s in visited])
            # print(molecules[-1].to(fmt='xyz')); import sys; sys.exit()
        if once and len(molecules) == 1:
            break

    return molecules


if __name__ == "__main__":
    from pyxtal.database.collection import Collection

    pmg = Structure.from_file("pyxtal/database/cifs/resorcinol.cif")
    mols = search_molecules_in_crystal(pmg, tol=0.2, once=False)
    print(len(mols))

    pmg = Collection("molecules")["xxv"]
    mols = search_molecules_in_crystal(pmg, tol=0.2, once=False)
    print(len(mols))
