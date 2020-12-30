"""
"""

#| - Import Modules
import os
import sys

import numpy as  np
import pandas as pd

from scipy.spatial.distance import pdist

from ase import io

# #########################################################
from vasp.vasp_methods import read_ase_sort_dat

# #########################################################
from methods import are_dicts_the_same
#__|


def get_unique_job_ids_ase_sort(job_ids, df_atoms_ind):
    """Find the job_ids that have unique ase-sort.dat files based on elimating redundant duplicates.
    """
    #| - get_unique_job_ids_ase_sort
    duplicate_job_ids = []

    job_ids_tmp = job_ids
    i_ind = 0
    while True:
        i_ind += 1
        if i_ind > 30:
            break

        duplicates_found_this_gen = []
        for job_id_i in job_ids:
            for job_id_j in job_ids:

                if job_id_i in duplicate_job_ids or job_id_j in duplicate_job_ids:
                    do_comparison = False
                else:
                    do_comparison = True

                if job_id_i == job_id_j:
                    do_comparison = False

                if do_comparison:
                    # print(job_id_i, job_id_j)
                    # #########################################
                    row_i = df_atoms_ind.loc[job_id_i]
                    row_j = df_atoms_ind.loc[job_id_j]
                    # #########################################
                    atom_index_mapping_i = row_i.atom_index_mapping
                    atom_index_mapping_j = row_j.atom_index_mapping
                    # #########################################
                    dicts_the_same = are_dicts_the_same(
                        atom_index_mapping_i, atom_index_mapping_j)

                    if dicts_the_same:
                        duplicate_job_ids.append(job_id_i)
                        duplicates_found_this_gen.append(job_id_i)

        # print(len(duplicates_found_this_gen))
        if len(duplicates_found_this_gen) == 0:
            break

    unique_ids = list(set(job_ids) - set(duplicate_job_ids))
    return(unique_ids)
    #__|


def all_keys_equal_to_vals(atom_index_mapping):
    """Checks if all the values are equal to their correspondant keys.

    Used to check ase-sort.dat file, if this is the case, the the job likely was started from a init.traj created from a previous vasp run as is therefore already reindexed.
    """
    #| - all_keys_equal_to_vals
    atom_index_mapping_i = atom_index_mapping

    key_val_equal_list = []
    for key_i, val_i in atom_index_mapping_i.items():
        key_val_equal_i = key_i == val_i
        key_val_equal_list.append(key_val_equal_i)

    all_keys_equal_to_vals = all(key_val_equal_list)
    return(all_keys_equal_to_vals)
    #__|


# df_jobs_i = df_jobs_i
# df_jobs_paths = df_jobs_paths

def get_df_atoms_ind(
    df_jobs_i=None,
    df_jobs_paths=None,
    ):
    """
    """
    #| - get_df_atoms_ind
    data_dict_list = []
    for job_id_i, row_i in df_jobs_i.iterrows():
        # #####################################################
        data_dict_i = dict()
        # #####################################################
        job_id_i = row_i.job_id
        compenv_i = row_i.compenv
        rev_num_i = row_i.rev_num
        # #####################################################

        # #####################################################
        row_paths_i = df_jobs_paths.loc[job_id_i]
        # #####################################################
        gdrive_path_i = row_paths_i.gdrive_path
        # #####################################################


        # #########################################################
        gdrive_path_i = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            gdrive_path_i)

        path_i = os.path.join(
            gdrive_path_i,
            "ase-sort.dat")

        from pathlib import Path
        my_file = Path(path_i)
        if my_file.is_file():
            atom_index_mapping, sort_list, resort_list = \
                read_ase_sort_dat(path_i=path_i)
        else:
            atom_index_mapping = None
            sort_list = None
            resort_list = None

        # #####################################################
        data_dict_i["job_id"] = job_id_i
        data_dict_i["compenv"] = compenv_i
        data_dict_i["rev_num"] = rev_num_i
        data_dict_i["atom_index_mapping"] = atom_index_mapping
        data_dict_i["sort_list"] = sort_list
        data_dict_i["resort_list"] = resort_list

        # #####################################################
        data_dict_list.append(data_dict_i)

    # #########################################################
    df_atoms_ind = pd.DataFrame(data_dict_list)
    df_atoms_ind = df_atoms_ind.set_index("job_id", drop=False)

    return(df_atoms_ind)
    #__|


def unique_ids_with_no_equal(
    unique_job_ids=None,
    df_atoms_ind_i=None,
    ):
    """
    """
    #| - unique_ids_with_no_equal
    unique_ids_no_equal = []
    for unique_job_id_i in unique_job_ids:
        row_i = df_atoms_ind_i.loc[unique_job_id_i]
        atom_index_mapping_i = row_i.atom_index_mapping
        all_keys_equal_to_vals_i = all_keys_equal_to_vals(atom_index_mapping_i)

        if not all_keys_equal_to_vals_i:
            unique_ids_no_equal.append(unique_job_id_i)

    # print("unique_ids_no_equal:", unique_ids_no_equal)
    return(unique_ids_no_equal)
    #__|


def atoms_distance_comparison(atoms_0, atoms_1):
    """
    """
    #| - atoms_distance_comparison
    # #####################################################
    num_atoms_0 = atoms_0.get_global_number_of_atoms()
    num_atoms_1 = atoms_1.get_global_number_of_atoms()

    same_num_atoms = False
    if num_atoms_0 == num_atoms_1:
        same_num_atoms = True
        num_atoms = num_atoms_0

    # #####################################################
    distance_sum = 0.
    for atom_i, atom_j in zip(atoms_0, atoms_1):

        x = np.array([
            atom_i.position,
            atom_j.position, ])
        distance_i = pdist(x)[0]

        distance_sum += distance_i

    # #####################################################
    distance_sum_per_atom = distance_sum / num_atoms

    # print("distance_sum_per_atom:", distance_sum_per_atom)
    # print("distance_sum:", distance_sum, "\n")

    return(distance_sum_per_atom)
    #__|
