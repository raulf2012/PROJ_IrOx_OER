#!/usr/bin/env python

"""
"""

# | - Import Modules
import os

import glob
import filecmp

import numpy as np
import pandas as pd

from ase.atoms import Atoms
from ase.io import read

from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
# __|


def nearest_atom(atoms, position):
    """Returns atom nearest to position"""
    #| - nearest_atom
    position = np.array(position)
    dist_list = []
    for atom in atoms:
        dist = np.linalg.norm(position - atom.position)
        dist_list.append(dist)

    return atoms[np.argmin(dist_list)]
    #__|

def nearest_atom_mine(atoms, position, nth_closest=0):
    """

    args:
      atoms:
      position:
      nth_closest: pass 0 for nearest atom, 1 for 2nd neareset and so on...
    """
    #| - nearest_atom_mine
    struct = AseAtomsAdaptor.get_structure(atoms)
    Lattice = struct.lattice

    # #########################################################
    dummy_site_j = PeriodicSite(
        "N", position, Lattice,
        to_unit_cell=False, coords_are_cartesian=True,
        properties=None, skip_checks=False)

    data_dict_list = []
    for index_i, site_i in enumerate(struct):
        data_dict_i = dict()
        # site_i = struct[32]

        distance_i, image_i = site_i.distance_and_image(dummy_site_j)
        # print("distance_i:", distance_i)

        # #####################################################
        data_dict_i["atoms_index"] = int(index_i)
        data_dict_i["distance"] = distance_i
        data_dict_i["image"] = image_i
        # #####################################################
        data_dict_list.append(data_dict_i)

    # #########################################################
    df_dist = pd.DataFrame(data_dict_list)
    df_dist = df_dist.sort_values("distance")


    # #########################################################
    closest_site = df_dist.iloc[nth_closest]

    closest_index = int(closest_site.atoms_index)
    closest_distance = closest_site.distance
    image = closest_site.image

    closest_atom = atoms[closest_index]


    out_dict = dict(
        closest_atom=closest_atom,
        closest_distance=closest_distance,
        image=image,
        )
    return(out_dict)
    # return(closest_atom)
    #__|

def get_magmom_diff_data(
    ads_atoms=None,
    slab_atoms=None,
    ads_magmoms=None,
    slab_magmoms=None,
    ):
    """
    """
    #| - get_magmom_diff_data

    # #########################################################
    out_dict__no_flipped = _get_magmom_diff_data(
        ads_atoms, slab_atoms,
        flip_spin_sign=False,
        ads_magmoms=ads_magmoms,
        slab_magmoms=slab_magmoms,
        )
    tot_abs_magmom_diff__no_flip = out_dict__no_flipped["tot_abs_magmom_diff"]
    # #########################################################
    out_dict__yes_flipped = _get_magmom_diff_data(
        ads_atoms, slab_atoms,
        flip_spin_sign=True,
        ads_magmoms=ads_magmoms,
        slab_magmoms=slab_magmoms,
        )
    tot_abs_magmom_diff__yes_flip = out_dict__yes_flipped["tot_abs_magmom_diff"]

    # #########################################################
    if tot_abs_magmom_diff__yes_flip < tot_abs_magmom_diff__no_flip:
        # print("Need to use the flipped spin solution")
        out_dict = out_dict__yes_flipped
    else:
        out_dict = out_dict__no_flipped

    # #########################################################
    # delta_magmoms = out_dict["delta_magmoms"]
    # tot_abs_magmom_diff = out_dict__yes_flipped["tot_abs_magmom_diff"]
    # ads_indices_not_used = out_dict["ads_indices_not_used"]
    # atoms_ave = out_dict["atoms_ave"]
    # delta_magmoms_unsorted = out_dict["delta_magmoms_unsorted"]

    return(out_dict)
    #__|

def _get_magmom_diff_data(
    ads_atoms,
    slab_atoms,
    ads_magmoms=None,
    slab_magmoms=None,
    flip_spin_sign=False,
    ):
    """
    Args:
      flip_spin_sign: Flip the spin of the ads_atoms (doesn't matter which one is flipped), it seems that the spin state is agnostic to spin flips, so this is necessary to check.
    """
    #| - get_magmom_diff_data
    import copy

    # #########################################################
    # Dealing with magmoms
    # #########################################################

    # #########################################################
    if ads_atoms.calc is None:
        if ads_magmoms is None:
            print("ads_atoms are missing calc/magmoms, need to pass ads_magmoms manually")
    else:
        ads_magmoms = ads_atoms.get_magnetic_moments()

    # #########################################################
    if slab_atoms.calc is None:
        if slab_magmoms is None:
            print("slab_atoms are missing calc/magmoms, need to pass slab_magmoms manually")
    else:
        slab_magmoms = slab_atoms.get_magnetic_moments()


    # Flip magmom of adsorbate atoms
    if flip_spin_sign:
        ads_magmoms = -1. * ads_magmoms

    #| - Pre-config
    if len(ads_atoms) >= len(slab_atoms):
        ads = ads_atoms
        slab = slab_atoms
        indexed_by = "slab"
        not_indexed_by = "ads"
    else:
        slab = ads_atoms
        ads = slab_atoms
        indexed_by = "ads"
        not_indexed_by = "slab"
    #__|

    #| - Main loop
    positions_ave = []
    symbols_list = []

    delta_magmoms = []
    ads_indices_used = []
    for slab_atom in slab:
        # ads_atom = nearest_atom(ads, slab_atom.position)
        # ads_atom = nearest_atom_mine(ads, slab_atom.position)

        # #################################################
        out_dict_i = nearest_atom_mine(ads, slab_atom.position, nth_closest=0)
        # #################################################
        closest_atom = out_dict_i["closest_atom"]
        closest_distance = out_dict_i["closest_distance"]
        image = out_dict_i["image"]
        # #################################################

        if closest_atom.symbol != slab_atom.symbol:
            # print("WARNING! MAGMOM COMPARISON FAILURE")
            # print(slab_atom)
            # print(closest_atom)
            # print("")

            # #############################################
            out_dict_i = nearest_atom_mine(ads, slab_atom.position, nth_closest=1)
            # #############################################
            closest_atom = out_dict_i["closest_atom"]
            closest_distance = out_dict_i["closest_distance"]
            image = out_dict_i["image"]
            # #############################################

            if closest_atom.symbol != slab_atom.symbol:

                # #########################################
                out_dict_i = nearest_atom_mine(ads, slab_atom.position, nth_closest=2)
                # #########################################
                closest_atom = out_dict_i["closest_atom"]
                closest_distance = out_dict_i["closest_distance"]
                image = out_dict_i["image"]
                # #########################################

                if closest_atom.symbol != slab_atom.symbol:
                    print("Couldn't find atom BAD")

            # break




        # #################################################
        position_orig = copy.deepcopy(closest_atom.position)
        position_i = position_orig
        for i_cnt, image_i in enumerate(image):
            position_i = position_i - image_i * ads.cell[i_cnt]

        ave_position_i = (position_i + slab_atom.position) / 2
        positions_ave.append(ave_position_i)

        symbols_list.append(slab_atom.symbol)
        # #################################################


        ads_indices_used.append(closest_atom.index)

        slab_magmom_i = slab_magmoms[slab_atom.index]
        ads_magmom_i = ads_magmoms[closest_atom.index]

        # d_magmom_i = slab_atom.magmom - closest_atom.magmom
        d_magmom_i = slab_magmom_i - ads_magmom_i

        delta_magmoms.append(d_magmom_i)

    delta_magmoms_unsorted = copy.deepcopy(delta_magmoms)
    delta_magmoms = list(zip(range(len(slab)), delta_magmoms))
    delta_magmoms.sort(key=lambda x: abs(x[1]),reverse=True)
    #__|

    #| - Gathering not used indices
    ads_indices_not_used = []
    for i in range(len(ads)):
        if i not in ads_indices_used:
            ads_indices_not_used.append(i)
    #__|

    #| - Calculate total absolute magmom diff
    tot_abs_magmom_diff = 0.
    for diff_i in delta_magmoms:
        tot_abs_magmom_diff += np.abs(diff_i[1])
    norm_abs_magmom_diff = tot_abs_magmom_diff / len(delta_magmoms)
    #__|



    #| - Creating average atoms object with magmom diff set to magmoms
    from ase import Atoms

    # delta_magmoms_i = [i[1] for i in delta_magmoms_unsorted]
    delta_magmoms_i = delta_magmoms_unsorted

    atoms_ave = Atoms(
        symbols=symbols_list,
        positions=positions_ave,
        # numbers=None,
        # tags=None,
        # momenta=None,
        # masses=None,
        # magmoms=None,
        # charges=None,
        # scaled_positions=None,
        cell=slab.cell,
        # pbc=None,
        # celldisp=None,
        # constraint=None,
        # calculator=None,
        # info=None,
        # velocities=None,
        )

    atoms_ave.set_initial_magnetic_moments(magmoms=delta_magmoms_i)
    # atoms_ave.write("out_data/atoms_ave.traj")
    #__|




    out_dict = dict(
        delta_magmoms=delta_magmoms,
        tot_abs_magmom_diff=tot_abs_magmom_diff,
        norm_abs_magmom_diff=norm_abs_magmom_diff,
        ads_indices_not_used=ads_indices_not_used,
        atoms_ave=atoms_ave,
        delta_magmoms_unsorted=delta_magmoms_unsorted,
        )
    return(out_dict)
    #__|

class Get_G:
    """
    """

    # | - Get_G
    def __init__(self,
        slab,
        ads,
        index=-1,
        quiet=False,
        ):
        """
        Get_G(ads,slab) where ads/slab are either paths to traj files or paths to directory containing traj files that have energy and forces.
        The directories that contain these traj files must also contain a calculation directory with pw.inp.
        """
        # | - __init__
        self.quiet = quiet

        self.slab_atoms = slab
        self.ads_atoms = ads
        # __|

    def compare_magmoms(self):
        """
        """
        # | - compare_magmoms
        def nearest_atom(atoms,position):
            "Returns atom nearest to position"
            position = np.array(position)
            dist_list = []
            for atom in atoms:
                dist = np.linalg.norm(position - atom.position)
                dist_list.append(dist)

            return atoms[np.argmin(dist_list)]

        if len(self.ads_atoms) >= len(self.slab_atoms):
            ads = self.ads_atoms
            slab = self.slab_atoms
            indexed_by = "slab"
            not_indexed_by = "ads"
        else:
            slab = self.ads_atoms
            ads = self.slab_atoms
            indexed_by = "ads"
            not_indexed_by = "slab"

        delta_magmoms = []
        ads_indices_used = []
        for atom in slab:
            ads_atom = nearest_atom(ads,atom.position)
            if not self.quiet:
               if ads_atom.symbol != atom.symbol: print("WARNING! MAGMOM COMPARISON FAILURE")
            ads_indices_used.append(ads_atom.index)
            delta_magmoms.append(atom.magmom - ads_atom.magmom)

        ads_indices_not_used = []
        for i in range(len(ads)):
            if i not in ads_indices_used:
                ads_indices_not_used.append(i)

        # RF | 181106
        # self.delta_magmoms = zip(range(len(slab)), delta_magmoms)
        self.delta_magmoms = list(zip(range(len(slab)), delta_magmoms))
        self.delta_magmoms.sort(key=lambda x: abs(x[1]),reverse=True)

        common = ""
        uncommon = ""
        for i in range(8):
            atom = slab[self.delta_magmoms[i][0]]
            common += "%s%d: %.2f\t"%(atom.symbol,atom.index,self.delta_magmoms[i][1])
        for i in ads_indices_not_used:
            uncommon += "%s%d: %.2f\t"%(ads[i].symbol,ads[i].index,ads[i].magmom)

        if self.quiet:
            return
        else:
            print("~"*6 + "MAGNETIC MOMENT COMPARISON" + "~"*6)
            print("Largest magnetic moment discrepancies (indexed by %s)"%indexed_by)
            print(common)
            print("Magnetic moments only present in %s"%not_indexed_by)
            print(uncommon + "\n")

            # print("~"*6 + "MAGNETIC MOMENT COMPARISON" + "~"*6)
            # print("Largest magnetic moment discrepancies (indexed by %s)"%indexed_by)
            # print(common)
            # print("Magnetic moments only present in %s"%not_indexed_by)
            # print(uncommon + "\n")
        # __|

    def compare_magmoms_2(self):
        """
        """
        #| - compare_magmoms_2

        # #################################################
        ads_atoms = self.ads_atoms
        slab_atoms = self.slab_atoms
        quiet = self.quiet
        # #################################################


        if len(ads_atoms) >= len(slab_atoms):
            ads = ads_atoms
            slab = slab_atoms
            indexed_by = "slab"
            not_indexed_by = "ads"
        else:
            slab = ads_atoms
            ads = slab_atoms
            indexed_by = "ads"
            not_indexed_by = "slab"

        delta_magmoms = []
        ads_indices_used = []
        for atom in slab:
            ads_atom = nearest_atom(ads, atom.position)
            if not quiet:
               if ads_atom.symbol != atom.symbol: print("WARNING! MAGMOM COMPARISON FAILURE")
            ads_indices_used.append(ads_atom.index)
            delta_magmoms.append(atom.magmom - ads_atom.magmom)

        ads_indices_not_used = []
        for i in range(len(ads)):
            if i not in ads_indices_used:
                ads_indices_not_used.append(i)


        delta_magmoms = list(zip(range(len(slab)), delta_magmoms))
        delta_magmoms.sort(key=lambda x: abs(x[1]),reverse=True)


        common = ""
        uncommon = ""
        for i in range(8):
            atom = slab[delta_magmoms[i][0]]
            common += "%s%d: %.2f\t"%(atom.symbol, atom.index, delta_magmoms[i][1])
        for i in ads_indices_not_used:
            uncommon += "%s%d: %.2f\t"%(ads[i].symbol,ads[i].index,ads[i].magmom)

        if quiet:
            return
        else:
            print("~"*6 + "MAGNETIC MOMENT COMPARISON" + "~"*6)
            print("Largest magnetic moment discrepancies (indexed by %s)"%indexed_by)
            print(common)
            print("Magnetic moments only present in %s"%not_indexed_by)
            print(uncommon + "\n")

            # print("~"*6 + "MAGNETIC MOMENT COMPARISON" + "~"*6)
            # print("Largest magnetic moment discrepancies (indexed by %s)"%indexed_by)
            # print(common)
            # print("Magnetic moments only present in %s"%not_indexed_by)
            # print(uncommon + "\n")
        #__|

    # __|



#| - __old__
# # ads_atoms =
# # slab_atoms =
# flip_spin_sign = True
#__|
