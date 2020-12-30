"""PROJ_irox_oer global methods.

Author: Raul A. Flores
"""


#| - Import Modules
import os
import sys

import copy
# from pathlib import Path
from pathlib import Path
from contextlib import contextmanager

import pickle
import  json

import math
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis import local_env

# #########################################################
from misc_modules.pandas_methods import drop_columns
#__|


from methods_magmom_comp import (
    get_magmom_diff_data,
    _get_magmom_diff_data,
    )

# from methods_ccf_struct import (
#     get_ccf,
#     get_D_ij,
#     get_identical_slabs,
#     )

# #########################################################
#| - Get data objects methods

def get_df_dft():
    """
    """
    #| - get_df_dft
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/process_bulk_dft",
        "out_data/df_dft.pickle")
    with open(path_i, "rb") as fle:
        df_dft = pickle.load(fle)

    return(df_dft)
    #__|

def get_df_slab(mode="final"):
    """Get dataframe of all IrOx OER slabs.

    Args:
      mode: either "final" or "almost-final"
    """
    #| - get_df_dft

    if mode == "final":
        path_i = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/creating_slabs",
            "out_data",
            "df_slab_final.pickle")
    elif mode == "almost-final":
        path_i = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/creating_slabs",
            "out_data/df_slab.pickle")
    else:
        print("Need to pick a mode, either 'final' or 'almost-final'")


    my_file = Path(path_i)
    if my_file.is_file():
        with open(path_i, "rb") as fle:
            df_slab = pickle.load(fle)

        if "slab_id" in df_slab.columns:
            df_slab = df_slab.set_index("slab_id", drop=False)
    else:
        print("File doesn't exist or not found")
        df_slab = None

    return(df_slab)
    #__|

def get_df_slab_simil():
    """
    """
    #| - get_df_slab_simil

    # #####################################################
    import pickle; import os
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs/slab_similarity",
        "out_data/df_slab_simil.pickle")

    # with open(path_i, "rb") as fle:
    #     df_slab_simil = pickle.load(fle)


    my_file = Path(path_i)
    if my_file.is_file():
        with open(path_i, "rb") as fle:
            df_slab_simil = pickle.load(fle)
    else:
        print("Couldn't read df_slab_simil")
        print(path_i)
        print("")
        df_slab_simil = pd.DataFrame()


    return(df_slab_simil)
    #__|

def get_structure_coord_df(
    atoms,
    porous_adjustment=True,
    ):
    """
    """
    #| - get_structure_coord_df
    atoms_i = atoms
    structure = AseAtomsAdaptor.get_structure(atoms_i)

    #| - __old__
    # CrysNN = local_env.VoronoiNN(
    #     tol=0,
    #     targets=None,
    #     cutoff=13.0,
    #     allow_pathological=False,
    #     weight='solid_angle',
    #     extra_nn_info=True,
    #     compute_adj_neighbors=True,
    #     )
    #__|

    CrysNN = local_env.CrystalNN(
        weighted_cn=False,
        cation_anion=False,
        distance_cutoffs=(0.01, 0.4),
        x_diff_weight=3.0,
        # porous_adjustment=True,
        porous_adjustment=porous_adjustment,
        search_cutoff=7,
        fingerprint_length=None)


    coord_data_dict = dict()
    data_master = []
    for i_cnt, site_i in enumerate(structure.sites):
        site_elem_i = site_i.species_string

        data_dict_i = dict()
        data_dict_i["element"] = site_elem_i
        data_dict_i["structure_index"] = i_cnt

        nn_info_i = CrysNN.get_nn_info(structure, i_cnt)
        data_dict_i["nn_info"] = nn_info_i

        neighbor_list = []
        for neighbor_j in nn_info_i:
            neigh_elem_j = neighbor_j["site"].species_string
            neighbor_list.append(neigh_elem_j)

        neighbor_count_dict = dict()
        for i in neighbor_list:
            neighbor_count_dict[i] = neighbor_count_dict.get(i, 0) + 1

        data_dict_i["neighbor_count"] = neighbor_count_dict
        data_master.append(data_dict_i)

    df_struct_coord_i = pd.DataFrame(data_master)

    # #####################################################
    # #####################################################
    def method(row_i):
        neighbor_count = row_i.neighbor_count
        num_neighbors = 0
        for key, val in neighbor_count.items():
            num_neighbors += val
        return(num_neighbors)

    df_struct_coord_i["num_neighbors"] = df_struct_coord_i.apply(
        method,
        axis=1)

    return(df_struct_coord_i)
    #__|

def get_df_jobs(exclude_wsl_paths=True):
    """
    The data object is created by the following notebook:

    $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.ipynb
    """
    #| - get_df_jobs

    # #####################################################
    import pickle; import os
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_processing",
        "out_data/df_jobs_combined.pickle")
    with open(path_i, "rb") as fle:
        df_jobs = pickle.load(fle)
    # #####################################################

    if exclude_wsl_paths:
        df_jobs = df_jobs[df_jobs.compenv != "wsl"]

    df_jobs = df_jobs.fillna("NaN")

    # print("TEMP | Removing NERSC manually")
    # df_jobs = df_jobs[df_jobs.compenv != "nersc"]

    #| - __old__
    # # Sorting dataframe
    # sort_list = ["compenv", "bulk_id", "slab_id", "att_num", "rev_num"]
    # df_jobs = df_jobs.sort_values(sort_list)
    #
    #
    # #| - Adding short path column
    # def method(row_i):
    #     """
    #     """
    #     path_job_root_w_att_rev = row_i.path_job_root_w_att_rev
    #
    #     new_path_list = []
    #     start_adding = False; start_adding_ind = None
    #     for i_cnt, i in enumerate(path_job_root_w_att_rev.split("/")):
    #         if i == "dft_jobs":
    #             start_adding = True
    #             start_adding_ind = i_cnt
    #         if start_adding and i_cnt > start_adding_ind:
    #             new_path_list.append(i)
    #
    #     path_short_i = "/".join(new_path_list)
    #
    #     return(path_short_i)
    #
    # df_i = df_jobs
    # df_i["path_short"] = df_i.apply(method, axis=1)
    # df_jobs = df_i
    # #__|
    #
    # #| - Dropping Columns
    # cols_to_drop = [
    #     "is_rev_dir",
    #     "is_attempt_dir",
    #     "path_job_root_w_att",
    #     "gdrive_path",
    #     "path_rel_to_proj",
    #     "path_full",
    #     "path_job_root_w_att_rev",
    #     # "path_job_root",
    #     ]
    #
    # df_jobs = df_jobs.drop(
    #     # labels=["", ],
    #     # axis=1,
    #     # index=None,
    #     columns=cols_to_drop,
    #     # level=None,
    #     # inplace=False,
    #     # errors="raise",
    #     )
    # #__|
    #__|

    return(df_jobs)
    #__|

def get_df_jobs_paths():
    """
    The data object is created by the following notebook:

    $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.ipynb
    """
    #| - get_df_jobs

    # #####################################################
    import pickle; import os

    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_processing",
        "out_data/df_jobs_paths.pickle")
    with open(path_i, "rb") as fle:
        df_jobs_paths = pickle.load(fle)
    # #####################################################

    #  if exclude_wsl_paths:
    #      df_jobs = df_jobs[df_jobs.compenv != "wsl"]

    # Sorting dataframe
    #  sort_list = ["compenv", "bulk_id", "slab_id", "att_num", "rev_num"]
    #  df_jobs = df_jobs.sort_values(sort_list)

    return(df_jobs_paths)
    #__|

def get_df_jobs_data(
    exclude_wsl_paths=True,
    drop_cols=False,
    ):
    """
    from methods import get_df_jobs_data

    df_jobs_data = get_df_jobs_data()
    df_jobs_data

    """
    #| - get_df_jobs_data

    # #########################################################
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_processing",
        "out_data/df_jobs_data.pickle")
    with open(path_i, "rb") as fle:
        df_jobs_data = pickle.load(fle)
    # #########################################################

    if exclude_wsl_paths:
        df_jobs_data = df_jobs_data[df_jobs_data.compenv != "wsl"]


    #| - Drop columns
    if drop_cols:
        drop_cols_list = [
            "incar_params",
            "submitted",
            "isif",
            "ediff_conv_reached_dict",
            "true_false_ratio",
            "num_nonconv_scf",
            "num_conv_scf",
            ]
        df_jobs_data = df_jobs_data.drop(columns=drop_cols_list)
    #__|


    return(df_jobs_data)
    #__|

def get_df_jobs_data_clusters():
    """
    """
    #| - get_df_jobs_data_clusters
    compenv = os.environ["COMPENV"]

    compenvs = ["nersc", "sherlock", "slac", ]
    # compenvs = ["nersc", "sherlock", "slac", "wsl", ]

    df_jobs_data_clusters_empty = pd.DataFrame()

    df_list = []
    #  if compenv == "wsl":
    root_dir = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_processing")
    for compenv_i in compenvs:
        path_i = os.path.join(
            root_dir, "out_data",
            "df_jobs_data_" + compenv_i + ".pickle")
        my_file = Path(path_i)
        if my_file.is_file():

            # with open(path_i, "rb") as fle:
            #     df_jobs_data_i = pickle.load(fle)

            # _pickle.UnpicklingError
            try:
                with open(path_i, "rb") as fle:
                    df_jobs_data_i = pickle.load(fle)
            # except _pickle.UnpicklingError:
            except pickle.UnpicklingError:
                print("Yikes | _pickle.UnpicklingError")
                print(path_i)

            df_list.append(df_jobs_data_i)

    if len(df_list) > 0:
        df_jobs_data_clusters = pd.concat(df_list)
    else:
        df_jobs_data_clusters = df_jobs_data_clusters_empty
    #  else:
    #      df_jobs_data_clusters = df_jobs_data_clusters_empty


    # print("TEMP | Removing NERSC manually")
    # df_jobs_data_clusters = df_jobs_data_clusters[df_jobs_data_clusters.compenv != "nersc"]

    return(df_jobs_data_clusters)
    #__|

def get_df_jobs_anal():
    """
    """
    #| - get_df_jobs_anal
    # #####################################################
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_processing",
        "out_data")
    file_name_i = "df_jobs_anal.pickle"
    path_i = os.path.join(directory, file_name_i)
    with open(path_i, "rb") as fle:
        df_jobs_anal = pickle.load(fle)
    # #####################################################


    return(df_jobs_anal)
    #__|

def get_df_atoms_sorted_ind():
    """
    """
    #| - get_df_atoms_sorted_ind
    import pickle; import os
    path_i = os.path.join(
       os.environ["PROJ_irox_oer"],
       "dft_workflow/job_analysis/atoms_indices_order",
       "out_data/df_atoms_sorted_ind.pickle")
       # "out_data/df_atoms_index.pickle")
    with open(path_i, "rb") as fle:
        df_atoms_sorted_ind = pickle.load(fle)

    return(df_atoms_sorted_ind)
    #__|

def get_df_ads():
    """
    """
    #| - get_df_ads

    # #####################################################
    # import pickle; import os
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/collect_collate_dft_data",
        "out_data/df_ads.pickle")

    # with open(path_i, "rb") as fle:
    #     df_ads = pickle.load(fle)
    # # #####################################################


    my_file = Path(path_i)
    if my_file.is_file():
        with open(path_i, "rb") as fle:
            df_ads = pickle.load(fle)
    else:
        print("Couldn't read df_ads")
        print(path_i)
        print("")
        df_ads = pd.DataFrame()


    return(df_ads)
    #__|

# #########################################################

def get_df_slab_ids():
    """
    """
    #| - get_df_slab_ids
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs",
        "in_data/slab_id_mapping.csv")

    df_slab_ids = pd.read_csv(path_i, dtype=str)

    df_slab_ids = df_slab_ids.set_index(["bulk_id", "facet", ], drop=False)


    return(df_slab_ids)
    #__|

def update_df_slab_ids():
    """
    """
    #| - update_df_slab_ids
    # #####################################################
    # Read Data
    from methods import get_df_slab_ids
    df_slab_ids = get_df_slab_ids()

    from methods import get_df_slab
    df_slab = get_df_slab()


    # #####################################################
    # Checking that df_slab_ids has only unique bulk_id+facet pairs
    num_entries = len(df_slab_ids.index.tolist())
    num_unique_entries = len(list(set(df_slab_ids.index.tolist())))

    if num_entries != num_unique_entries:
        print("Woh what's going on here")

    mess_i = "Woops not good"
    assert num_entries == num_unique_entries, mess_i


    # #########################################################
    # Looping through df_slab rows and checking if the slab_id is present in df_slab_ids

    # #########################################################
    data_dict_list = []
    # #########################################################
    for slab_id_i, row_i in df_slab.iterrows():
        # #####################################################
        data_dict_i = dict()
        # #####################################################
        bulk_id_i = row_i.bulk_id
        facet_i = row_i.facet
        # #####################################################

        # df_slab_ids.loc[bulk_id_i, facet_i]
        slab_ids_match_i = "NaN"
        index_in_df_slab_ids_i = (bulk_id_i, facet_i, ) in df_slab_ids.index
        if index_in_df_slab_ids_i:
            # #################################################
            row_id_i = df_slab_ids.loc[(bulk_id_i, facet_i)]
            # #################################################
            slab_id__from_fle = row_id_i.slab_id
            # #################################################

            slab_ids_match_i = slab_id_i == slab_id__from_fle

        # #####################################################
        data_dict_i["bulk_id"] = bulk_id_i
        data_dict_i["facet"] = facet_i
        data_dict_i["slab_id"] = slab_id_i
        data_dict_i["index_in_df_slab_ids"] = index_in_df_slab_ids_i
        data_dict_i["slab_ids_match"] = slab_ids_match_i
        # #####################################################
        data_dict_list.append(data_dict_i)
        # #####################################################

    # #########################################################
    df = pd.DataFrame(data_dict_list)
    # #########################################################




    # #####################################################
    # Checking that data objects are consistent with each other
    df_i = df[df.index_in_df_slab_ids == True]

    df_slab__df_slab_ids__consistent = False
    unique_vals = list(set(df_i.slab_ids_match.tolist()))
    if (len(unique_vals) == 1) and unique_vals[0] == True:
        df_slab__df_slab_ids__consistent = True

    mess_i = "df_slab and df_slab_ids are not consistent"
    assert df_slab__df_slab_ids__consistent, mess_i

    # #####################################################
    # Getting all the entries in df_slab that aren't present in df_slab_ids
    df_1 = df[df.index_in_df_slab_ids == False]
    df_1 = df_1.set_index(["bulk_id", "facet", ], drop=False)
    df_1 = df_1.sort_index()

    # #####################################################
    # Combining the old and new df_slab_ids and saving
    df_slab_ids_new = pd.concat(
        [
            df_1[["bulk_id", "facet", "slab_id", ]],
            df_slab_ids,
            ],
        axis=0,
        )

    # #####################################################
    # Writing data to file
    pre_dir = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs")

    df_slab_ids_new.to_csv(
        os.path.join(pre_dir, "out_data/slab_id_mapping.csv"),
        index=False)

    df_slab_ids_new.to_csv(
        os.path.join(pre_dir, "in_data/slab_id_mapping.csv"),
        index=False)

        # "in_data/slab_id_mapping.csv",
    #__|

def get_df_job_ids():
    """
    """
    #| - get_df_job_ids
    df_job_ids_empty = pd.DataFrame(
        columns=[
            "job_id",

            "compenv",
            "bulk_id",
            "slab_id",
            "facet",
            "att_num",
            "rev_num",
            "ads",
            "active_site",
            ],
        )


    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_processing",
        "out_data/job_id_mapping.csv")
    my_file = Path(path_i)
    if my_file.is_file():
        #| -  Attempt to read csv file
        try:
            df_job_ids = pd.read_csv(path_i, dtype=str)

            df_job_ids.att_num = df_job_ids.att_num.astype(int)
            df_job_ids.rev_num = df_job_ids.rev_num.astype(int)
            # df_job_ids.active_site = df_job_ids.active_site.astype(int)
            df_job_ids.active_site = pd.to_numeric(
                df_job_ids["active_site"], errors='coerce')

        except pd.io.common.EmptyDataError as err:
            df_job_ids = df_job_ids_empty
            print(err)
        #__|

    else:
        #| - Return empty dataframe
        # df_job_ids = None
        df_job_ids = df_job_ids_empty
        #__|

    return(df_job_ids)
    #__|

def get_df_slabs_to_run():
    """Returns dataframe of DFT relaxed slabs that are ok to continue calcs with."""
    #| - get_df_slabs_to_run
    file_path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/manually_analyze_slabs",
        "slabs_to_run.csv")

    df_slabs_to_run = pd.read_csv(
        file_path_i,
        header=None,
        names=["compenv", "slab_id", "att_num", "status", ],
        )

    return(df_slabs_to_run)
    #__|


def get_df_init_slabs():
    """Returns a dataframe which contains the initial atoms object for all systems.
    """
    #| - get_df_init_slabs
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/get_init_slabs_bare_oh",
        "out_data/df_init_slabs.pickle")

    my_file = Path(path_i)
    if my_file.is_file():
        with open(path_i, "rb") as fle:
            df_init_slabs = pickle.load(fle)
    else:
        print("Couldn't find df_init_slabs file")
        print(path_i)
        print("")

        df_init_slabs = pd.DataFrame()

    return(df_init_slabs)
    #__|

def get_df_slabs_oh():
    """
    """
    #| - get_df_slabs_oh
    # #########################################################
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/create_oh_slabs",
        "out_data/df_slabs_oh.pickle")
    with open(path_i, "rb") as fle:
        df_slabs_oh = pickle.load(fle)
    # #########################################################

    return(df_slabs_oh)
    #__|

def get_df_xrd():
    """
    """
    #| - get_df_xrd

    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/xrd_bulks",
        "out_data/df_xrd.pickle")

    # from pathlib import Path
    my_file = Path(path_i)
    if my_file.is_file():
        with open(path_i, "rb") as fle:
            df_xrd = pickle.load(fle)
    else:
        df_xrd = pd.DataFrame()

    return(df_xrd)
    #__|

def get_df_bulk_manual_class():
    """
    """
    #| - get_df_bulk_manual_class
    # #################################################
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        # "workflow/process_bulk_dft/manually_classify_bulks",
        "workflow/process_bulk_dft/manually_classify_bulks",
        "out_data",
        "bulk_manual_classification.csv")
    df_bulk_class = pd.read_csv(path_i)
    # df_bulk_class = pd.read_csv("./bulk_manual_classification.csv")
    # #################################################

    # Filling empty spots of layerd column with False (if not True)
    df_bulk_class[["layered"]] = df_bulk_class[["layered"]].fillna(value=False)

    # Setting index
    df_bulk_class = df_bulk_class.set_index("bulk_id", drop=False)

    return(df_bulk_class)
    #__|

def get_bulk_selection_data():
    """
    """
    #| - get_bulk_selection_data

    # ########################################################
    data_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs/selecting_bulks",
        "out_data/data.json")
    with open(data_path, "r") as fle:
        data = json.load(fle)
    # ########################################################

    # bulk_ids__octa_unique = data["bulk_ids__octa_unique"]

    return(data)
    #__|


# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################


#| - Feature Engineering

def get_df_features_targets():
    """
    """
    #| - get_df_features_targets
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering",
        "out_data/df_features_targets.pickle")
    with open(path_i, "rb") as fle:
        df_features_targets = pickle.load(fle)

    return(df_features_targets)
    #__|

def get_df_features():
    """
    """
    #| - get_df_features

    # #########################################################
    import pickle; import os
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering",
        "out_data/df_features.pickle")
    with open(path_i, "rb") as fle:
        df_features = pickle.load(fle)
    # #########################################################


    return(df_features)
    #__|



def get_df_eff_ox():
    """
    """
    #| - get_df_eff_ox
    # #####################################################
    import pickle; import os
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        # "workflow/feature_engineering",
        "workflow/feature_engineering/oxid_state",
        "out_data/df_eff_ox.pickle")
    with open(path_i, "rb") as fle:
        df_eff_ox = pickle.load(fle)
    # #####################################################

    return(df_eff_ox)
    #__|

def get_df_octa_vol():
    """
    """
    #| - get_df_eff_ox

    # #########################################################
    import pickle; import os
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering/octahedra_volume",
        "out_data/df_octa_vol.pickle")
    with open(path_i, "rb") as fle:
            df_octa_vol = pickle.load(fle)
    # #########################################################

    return(df_octa_vol)
    #__|

def get_metal_index_of_active_site(
    df_coord=None,
    active_site=None,
    verbose=False,
    ):
    """
    """
    #| - get_metal_index_of_active_site
    # #####################################################
    df_coord_i = df_coord
    active_site_i = active_site
    # #####################################################
    metal_index_i = None
    all_good = True
    # #####################################################

    row_coord_i = df_coord_i.loc[active_site_i]
    nn_info_i = row_coord_i.nn_info


    num_non_H_neigh = 0
    nn_info_non_H = []
    for nn_j in nn_info_i:
        site_j = nn_j["site"]
        if site_j.specie.name != "H":
            num_non_H_neigh += 1
            nn_info_non_H.append(nn_j)

    # Check if the active *O has exactly 1 non-hydrogen neigh
    if num_non_H_neigh != 1:
        all_good = False
        if verbose:
            print("The active oxygen has more than 1 NN, this is ambigious")

    # If good to go, then check that active Ir has 6 *O neigh
    if all_good:
        nn_info_i = nn_info_non_H[0]
        metal_index_i = nn_info_i["site_index"]

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["all_good"] = all_good
    out_dict["metal_index"] = metal_index_i
    # #####################################################
    return(out_dict)
    # #####################################################

    # return(metal_index_i)
    #__|

#__|


# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################

def get_df_coord(
    slab_id=None,
    bulk_id=None,
    mode="bulk",  # 'bulk', 'slab', 'post-dft', 'init-slab'
    slab=None,
    post_dft_name_tuple=None,
    porous_adjustment=True,

    init_slab_name_tuple=None,

    verbose=False,
    ):
    """
    mode:
        'bulk', 'slab', 'post-dft'
    post_dft_name_tuple:
        ex.) ('nersc', 'fosurufu_23',    'o',    'NaN',       1)
              compenv      slab_id       ads   active_site att_num
    init_slab_name_tuple:
        ex.) (compenv, slab_id, ads, active_site, att_num)

    """
    #| - get_df_coord

    if mode == "slab":
        #| - Get slab df_coord
        file_name_i = slab_id + ".pickle"
        root_path = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/creating_slabs",
            "out_data/df_coord_files")
        path_i = os.path.join(root_path, file_name_i)
        with open(path_i, "rb") as fle:
            df_coord_slab_i = pickle.load(fle)
            df_coord_i = df_coord_slab_i

        #| - Check if size of df_coord_i matches num_atoms in slab (if given)
        if slab is not None:
            num_atoms_in_slab = slab.get_global_number_of_atoms()
            num_atoms_in_df = len(df_coord_i.structure_index.tolist())
            if num_atoms_in_slab != num_atoms_in_df:
                file_name_i = slab_id + "_after_rep" + ".pickle"
                path_i = os.path.join(root_path, file_name_i)

                if verbose:
                    print("Reading the _after_rep file")
                    print(path_i)

                from pathlib import Path
                my_file = Path(path_i)
                if my_file.is_file():
                    with open(path_i, "rb") as fle:
                        df_coord_slab_i = pickle.load(fle)
                        df_coord_i = df_coord_slab_i
        #__|

        #__|
    elif mode == "bulk":
        #| - Get slab df_coord
        path_i = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/process_bulk_dft",
            "out_data/df_coord_files",
            bulk_id + ".pickle")
        with open(path_i, "rb") as fle:
            df_coord_bulk_i = pickle.load(fle)
            df_coord_i = df_coord_bulk_i
        #__|
    elif mode == "post-dft":
        #| - post-dft
        name_i = post_dft_name_tuple

        def is_number(s):
            #| - is_number
            try:
                float(s)
                return True
            except ValueError:
                return False
            #__|

        new_name_i = []
        for i in name_i:
            str_i = str(i)
            if is_number(str_i) and "." in str_i:
                new_name_i.append(int(i))
            else:
                new_name_i.append(i)

        new_name_i = tuple(new_name_i)

        # #################################################
        # file_name_i = "_".join([str(i) for i in list(name_i)])
        file_name_i = "_".join([str(i) for i in list(new_name_i)])
        if not porous_adjustment:
            file_name_i += "_porous_adj_False"
        file_name_i += ".pickle"
        path_i = os.path.join(
            os.environ["PROJ_irox_oer"],
            "dft_workflow/job_analysis/df_coord_for_post_dft",
            "out_data/df_coord_files", file_name_i)

        from pathlib import Path
        my_file = Path(path_i)
        if my_file.is_file():
            with open(path_i, "rb") as fle:
                df_coord_i = pickle.load(fle)
        else:
            df_coord_i = None

        # #################################################
        #__|
    elif mode == "init-slab":
        #| - init-slab

        mess_i = "IJSFDIS9ijdjfsij"
        assert init_slab_name_tuple is not None, mess_i
        # ex.) (compenv, slab_id, ads, active_site, att_num)

        compenv, slab_id, ads, active_site, att_num = init_slab_name_tuple

        # #####################################################
        # mode = "init-slab"
        # compenv = compenv_i
        # slab_id = slab_id_i
        # ads = ads_i
        # active_site = active_site_i
        # att_num = att_num_i
        # #####################################################


        if active_site == "NaN":
            active_site_str = active_site
        else:
            active_site_str = str(int(active_site))

        # #####################################################
        file_name_i = "" + \
            compenv + "__" + \
            slab_id + "__" + \
            ads + "__" +  \
            active_site_str + "__" +  \
            str(att_num) + \
            ""
        file_name_i += ".pickle"

        directory = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/creating_slabs/maintain_df_coord",
            "out_data/df_coord__init_slabs")

        path_i = os.path.join(directory, file_name_i)

        # print(path_i)

        # #################################################
        with open(path_i, "rb") as fle:
            df_coord_i = pickle.load(fle)
        # #################################################

        #__|

    else:
        print("Come back to this eventually")

    return(df_coord_i)
    #__|

def get_df_coord_wrap(
    name=None,
    active_site=None,
    ):
    """
    """
    #| - get_df_coord_wrap
    # #####################################################
    name_i = name
    active_site_j = active_site
    # #####################################################
    compenv_i = name[0]
    slab_id_i = name[1]
    ads_i = name[2]
    active_site_i = name[3]
    att_num_i = name[4]
    # #####################################################


    # # #############################################
    # if read_orig_O_df_coord:
    #     name_i = (
    #         compenv_i, slab_id_i, ads_i,
    #         "NaN", att_num_i)
    # else:
    #     name_i = (
    #         compenv_i, slab_id_i, ads_i,
    #         active_site_i, att_num_i)

    if ads_i == "o":
        porous_adjustment_i = True
    elif ads_i == "oh":
        # porous_adjustment_i = True
        porous_adjustment_i = False


    df_coord_i = get_df_coord(
        mode="post-dft",
        post_dft_name_tuple=name_i,
        porous_adjustment=porous_adjustment_i,
        )

    df_coord_i = df_coord_i.set_index("structure_index", drop=False)
    # row_coord_i = df_coord_i.loc[active_site_i]
    row_coord_i = df_coord_i.loc[active_site_j]

    nn_info_i = row_coord_i.nn_info
    neighbor_count_i = row_coord_i.neighbor_count

    num_Ir_neigh = neighbor_count_i.get("Ir", 0)

    if num_Ir_neigh != 1 and ads_i == "oh":
        porous_adjustment_i = True
        df_coord_i = get_df_coord(
            mode="post-dft",
            post_dft_name_tuple=name_i,
            porous_adjustment=porous_adjustment_i,
            )

    return(df_coord_i)
    #__|



# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################



def get_df_active_sites():
    """
    """
    #| - get_df_active_sites
    # /home/raulf2012/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_OER
    # workflow/enumerate_adsorption/out_data/df_active_sites.pickle

    # #####################################################
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/enumerate_adsorption",
        "out_data/df_active_sites.pickle")
    my_file = Path(path_i)
    if my_file.is_file():
        with open(path_i, "rb") as fle:
            df_active_sites = pickle.load(fle)
    else:
        df_active_sites = None
    # #####################################################


    return(df_active_sites)
    #__|

def get_df_magmoms():
    """
    """
    #| - get_df_magmoms
    # #########################################################
    import pickle; import os
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/compare_magmoms",
        "out_data")
    path_i = os.path.join(directory, "df_magmoms.pickle")


    my_file = Path(path_i)
    if my_file.is_file():
        with open(path_i, "rb") as fle:
            df_magmoms = pickle.load(fle)
    else:
        print("Couldn't read df_magmoms")
        print(path_i)
        print("")
        df_magmoms = pd.DataFrame()


    return(df_magmoms)
    #__|



def read_magmom_comp_data(
    name=None,
    ):
    """
    If `name` var is given then simply return the data dict for that system
    Example of `name`:
        name = ('slac', 'kuwurupu_88', 26.0)
    """
    #| - read_magmom_comp_data
    import os
    import pickle

    if name is None:
        #| - Read all data objects and collate
        root_dir = os.path.join(
            os.environ["PROJ_irox_oer"],
            "dft_workflow/job_analysis/compare_magmoms",
            "out_data/magmom_comparison_data")

        magmom_comparison_data = dict()
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:

                if "conflicted copy" in file:
                    continue
                if "conflicted copy" in file:
                    print("This shouldn't be printed")

                if "pickle" in file:
                    path_i = os.path.join(subdir, file)
                    with open(path_i, "rb") as fle:
                        data_dict_i = pickle.load(fle)

                    # #####################################
                    file_prt = file.split(".")[0]

                    name_list = []
                    name_list.extend(file_prt.split("__")[0:2])
                    name_list.append(float(file_prt.split("__")[2]))

                    name_i = tuple(name_list)

                    # #####################################
                    magmom_comparison_data[name_i] = data_dict_i
        #__|

        out_obj = magmom_comparison_data
    else:
        #| - Read indiv data dict
        name_i = name
        name_i_new = []
        for i in name_i:
            if type(i) == float:
                i_new = str(int(i))
                name_i_new.append(i_new)
            else:
                name_i_new.append(i)
        name_str = "__".join(name_i_new)
        file_name_i = name_str + ".pickle"

        file_name_i

        root_dir = os.path.join(
            os.environ["PROJ_irox_oer"],
            "dft_workflow/job_analysis/compare_magmoms",
            "out_data/magmom_comparison_data")

        path_i = os.path.join(root_dir, file_name_i)

        from pathlib import Path
        my_file = Path(path_i)
        if my_file.is_file():
            with open(path_i, "rb") as fle:
                data_dict_i = pickle.load(fle)
        else:
            data_dict_i = None
        #__|

        out_obj = data_dict_i

    return(out_obj)
    # return(magmom_comparison_data)
    #__|


#| - __old__
# def read_magmom_comp_data():
#     """
#     """
#     #| - read_magmom_comp_data
#     # #########################################################
#     # import pickle; import os
#
#     directory = os.path.join(
#         os.environ["PROJ_irox_oer"],
#         "dft_workflow/job_analysis/compare_magmoms",
#         "out_data")
#
#     path_i = os.path.join(directory, "magmom_comparison_data.pickle")
#
#     file_size = os.path.getsize(path_i)
#
#     if Path(path_i).is_file() and file_size > 0.:
#         print(path_i)
#         with open(path_i, "rb") as fle:
#             data_dict = pickle.load(fle)
#     else:
#         data_dict = dict()
#     # #########################################################
#
#     return(data_dict)
#     #__|
#
#__|


def get_df_jobs_oh_anal():
    """
    """
    #| - get_df_jobs_oh_anal
    # #########################################################
    import pickle; import os
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/analyze_oh_jobs",
        "out_data")
    path_i = os.path.join(directory, "df_jobs_oh_anal.pickle")
    with open(path_i, "rb") as fle:
        df_jobs_oh_anal = pickle.load(fle)
    # #########################################################

    return(df_jobs_oh_anal)
    #__|

def get_df_rerun_from_oh():
    """
    """
    #| - get_df_rerun_from_oh
    # #####################################################
    import pickle; import os
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/compare_magmoms",
        "out_data")
    path_i = os.path.join(directory, "df_rerun_from_oh.pickle")

    my_file = Path(path_i)
    if my_file.is_file():
        with open(path_i, "rb") as fle:
            df_rerun_from_oh = pickle.load(fle)
    else:
        print("Couldn't read df_rerun_from_oh")
        print(path_i)
        print("")
        df_rerun_from_oh = pd.DataFrame()


    return(df_rerun_from_oh)
    #__|

def get_df_struct_drift():
    """
    """
    #| - get_df_struct_drift

    # import pickle; import os
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/slab_struct_drift",
        "out_data/df_struct_drift.pickle")
    with open(path_i, "rb") as fle:
        df_struct_drift = pickle.load(fle)

    return(df_struct_drift)
    #__|

def read_data_json():
    """
    """
    #| - read_data_json

    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs",
        "out_data/data.json")
        # "out_data", "data.json")

    my_file = Path(path_i)
    if my_file.is_file():
        # data_path = os.path.join(
        #     "out_data/data.json")
        with open(path_i, "r") as fle:
            data = json.load(fle)
    else:
        data = dict()

    return(data)
    #__|




# #########################################################
#| - More specialized data related methods

def get_slab_id(bulk_id_i, facet_i, df_slab_ids):
    """Will check file of bulk_id+facet --> slab_id mapings and returns if in the data file, otherwise will return <None>.
    """
    #| - get_slab_id
    df_i = df_slab_ids[
        (df_slab_ids.bulk_id == bulk_id_i) & \
        (df_slab_ids.facet == facet_i)
        ]

    assert df_i.shape[0] <= 1, "There should be only 1 slab_id for a bulk_id+facet paring"

    if df_i.shape[0] == 1:
        row_i = df_i.iloc[0]
        slab_id_i = row_i.slab_id
    elif df_i.shape[0] == 0:
        slab_id_i = None

    return(slab_id_i)
    #__|

def get_job_id(
    compenv,
    bulk_id,
    slab_id,
    facet,
    att_num,
    rev_num,
    ads,
    active_site,
    df_job_ids=None,
    ):
    """
    """
    #| - get_job_id
    df_job_ids = df_job_ids.fillna("NaN")

    df_job_ids = df_job_ids.drop_duplicates()

    df = df_job_ids


    # if np.isnan(active_site):
    if pd.isnull(active_site):
        active_site = "NaN"
    else:
        pass


    df_i = df[
        (df.compenv == compenv) & \
        (df.bulk_id == bulk_id) & \
        (df.slab_id == slab_id) & \
        (df.facet == facet) & \
        (df.att_num == att_num) & \
        (df.rev_num == rev_num) & \
        (df.ads == ads) & \
        (df.active_site == active_site) & \
        [True for i in range(len(df))]
        ]

    assert df_i.shape[0] <= 1, "Must only have 1 hit or none"

    job_id_i = "Not defined"
    if df_i.shape[0] == 1:
        row_i = df_i.iloc[0]
        job_id_i = row_i.job_id
    elif df_i.shape[0] == 0:
        job_id_i = None
    elif df_i.shape[0] > 1:
        job_ids = df_i.job_id

        job_ids_unique = list(set(job_ids.tolist()))
        if len(job_ids_unique) == 1:
            job_id_i = job_ids_unique[0]
        else:
            print("There wasn't a unique job_id for this system")

    else:
        print("Couldn't get job id here")
        print("df_i:", df_i)
        print("df_i.shape:", df_i.shape)


    if job_id_i == "Not defined" or job_id_i == None:
        # print("")
        job_id_i = None
        # print("failed to get job_id")

        # print("df_i:", df_i)
        # print("df_i.shape:", df_i.shape)


    return(job_id_i)
    #__|

def get_df_oer_groups():
    """
    """
    #| - get_df_oer_groups
    # #########################################################
    import pickle; import os
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/prepare_oer_sets",
        "out_data")
    path_i = os.path.join(directory, "df_oer_groups.pickle")
    with open(path_i, "rb") as fle:
        df_oer_groups = pickle.load(fle)
    # #########################################################

    return(df_oer_groups)
    #__|

def get_other_job_ids_in_set(
    job_id,
    df_jobs=None,
    oer_set=False,
    ):
    """
    """
    #| - get_other_job_ids_in_set
    row_jobs = df_jobs.loc[job_id]

    # #####################################################
    compenv_i = row_jobs.compenv
    bulk_id_i = row_jobs.bulk_id
    slab_id_i = row_jobs.slab_id
    ads_i = row_jobs.ads
    att_num_i = row_jobs.att_num
    active_site_i = row_jobs.active_site
    # #####################################################


    # #####################################################
    if oer_set:
        #| - Create OER set, all jobs in OER  set
        # #####################################################
        df_jobs_i = df_jobs[
            (df_jobs.compenv == compenv_i) & \
            (df_jobs.bulk_id == bulk_id_i) & \
            (df_jobs.slab_id == slab_id_i) & \
            # (df_jobs.ads == ads_i) & \
            # (df_jobs.att_num == att_num_i) & \
            (df_jobs.active_site == active_site_i) & \
            [True for i in range(len(df_jobs))]
            ]

        # df_jobs_i =
        df_jobs_o_i = df_jobs[
            (df_jobs.compenv == compenv_i) & \
            (df_jobs.bulk_id == bulk_id_i) & \
            (df_jobs.slab_id == slab_id_i) & \
            (df_jobs.ads == "o") & \
            # (df_jobs.att_num == att_num_i) & \
            (df_jobs.active_site == "NaN") & \
            [True for i in range(len(df_jobs))]
            ]

        df_jobs_out_i = pd.concat([
            df_jobs_o_i,
            df_jobs_i,
            ])
        #__|
    else:
        #| - Create job set, all jobs in one system
        # #####################################################
        df_jobs_i = df_jobs[
            (df_jobs.compenv == compenv_i) & \
            (df_jobs.bulk_id == bulk_id_i) & \
            (df_jobs.slab_id == slab_id_i) & \
            (df_jobs.ads == ads_i) & \
            (df_jobs.att_num == att_num_i) & \
            (df_jobs.active_site == active_site_i) & \
            [True for i in range(len(df_jobs))]
            ]
        df_jobs_out_i = df_jobs_i
        #__|
    # #####################################################
    df_jobs_out_i = df_jobs_out_i.sort_values(["ads", "att_num", "rev_num", ])
    # #####################################################

    return(df_jobs_out_i)
    #__|


#| - __old__
# def get_df_jobs_max_rev(
#     df_jobs=None,
#     ):
#     """
#     """
#     #| - get_df_jobs_max_rev
#     max_rev_job_ids = []
#
#     rows_list = []
#     group_list = ["compenv", "bulk_id", "slab_id", "facet", "att_num", ]
#     # group_list = ["slab_id", "att_num", ]
#     grouped = df_jobs.groupby(group_list)
#     for name, group in grouped:
#         # print("---")
#         slab_id = name[0]
#         att_num = name[1]
#
#         # #####################################################
#         facet = group.facet.unique().tolist()[0]
#
#         # #####################################################
#         max_rev_num = group.rev_num.max()
#         df_max_i = group[group.rev_num == max_rev_num]
#         assert df_max_i.shape[0] == 1, "This should be one I think"
#         job_id_max_i = df_max_i.iloc[0].name
#         row_i = df_max_i.iloc[0]
#         rows_list.append(row_i)
#         # #####################################################
#         max_rev_job_ids.append(job_id_max_i)
#
#         # # #############################################################################
#         # # #############################################################################
#         # # #############################################################################
#         # indices = [
#         #     "weritidu_20",
#         #     "mohegosa_07",
#         #     ]
#
#         # for index_i in indices:
#         #     if index_i in group.index:
#         #         print("IJSDIFISDI")
#         #         group_tmp = group
#         #         from IPython.display import display
#         #         display(group)
#         # # #############################################################################
#         # # #############################################################################
#         # # #############################################################################
#
#
#     # df_jobs_max = df_jobs.loc[max_rev_job_ids]
#     # rows_list
#
#     df_jobs_max = pd.DataFrame(rows_list)
#
#     mess_i = "All rows's rev number must be the same as the max rev num"
#     check_i = all(df_jobs_max.num_revs == df_jobs_max.rev_num)
#     assert check_i, mess_i
#
#     df_jobs_max = df_jobs_max.sort_values(["compenv", "path_job_root", ])
#
#     return(df_jobs_max)
#     #__|
#__|

#__|

#__|

# #########################################################
#| - ASE Atoms Methods
def symmetrize_atoms(
    write_atoms=False,
    ):
    """
    """
    #| - symmetrize_atoms
    name = argv[1]
    atoms = read(name)

    images = [atoms.copy()]

    spglibdata = get_symmetry_dataset(
        (
            atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers(),
            ),
        symprec=1e-3,
        )

    spacegroup = spglibdata['number']

    wyckoffs = spglibdata['wyckoffs']
    print(spglibdata['number'])
    print(wyckoffs)


    s_name = spglibdata['international']
    #str(spacegroup) + '_' + '_'.join(sorted(wyckoffs))

    std_cell = spglibdata['std_lattice']
    positions = spglibdata['std_positions']
    numbers = spglibdata['std_types']

    atoms = Atoms(
        numbers=numbers,
        cell=std_cell,
        pbc=True,
        )

    atoms.set_scaled_positions(positions)
    atoms.wrap()
    images += [atoms]


    if write_atoms:
        # view(images)
        new_name = name.rstrip('.cif') + '_conventional_' + str(spacegroup) + '.cif'
        #print(new_name)

        atoms.write(new_name)

    return(atoms)
    #__|

def remove_atoms(atoms=None, atoms_to_remove=None):
    """Remove atoms that match indices of `atoms_to_remove` and return new ase atoms object
    """
    #| - remove_atoms
    atoms_new = copy.deepcopy(atoms)

    bool_mask = []
    for atom in atoms:

        if atom.index in atoms_to_remove:
            bool_mask.append(False)
        else:
            bool_mask.append(True)

    atoms_new = atoms_new[bool_mask]

    return(atoms_new)
    #__|

def get_slab_thickness(atoms=None):
    """
    """
    #| - get_slab_thickness
    z_positions = atoms.positions[:,2]

    z_max = np.max(z_positions)
    z_min = np.min(z_positions)

    slab_thickness = z_max - z_min

    return(slab_thickness)
    # print("slab_thickness:", slab_thickness)
    #__|

def add_adsorbate_and_pass_magmoms(
    atoms=None,
    ads_atoms=None,
    ):
    """
    """
    #| - add_adsorbate_and_pass_magmoms
    # directory = "out_data"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    cell = atoms.cell
    magmoms_before_append = atoms.get_magnetic_moments()

    # #########################################################
    ads_atoms.set_cell(cell)

    # ads_atoms = Atoms(
    #     symbols=["H", ],
    #     positions=positions,
    #     tags=None,
    #     magmoms=[0.5, ],
    #     cell=cell,
    #     pbc=[True, True, True],
    #     velocities=None)


    # #########################################################
    # Write atoms before modifcation to file
    # atoms.write("pre.traj")
    # atoms.write("pre.json")

    # #########################################################
    # Append adsorbate atoms
    for atom in ads_atoms:
        atoms.append(atom)

    # #########################################################
    # Set new magmoms to atoms object
    magmoms_new = np.concatenate([
        magmoms_before_append,
        ads_atoms.get_initial_magnetic_moments(),
        ])

    atoms.set_initial_magnetic_moments(magmoms_new)

    return(atoms)
    #__|

def remove_atom_keep_magmoms(
    atoms=None,
    index=None,
    ):
    """
    """
    #| - remove_atom_keep_magmoms
    # directory = "out_data"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    index_i = index
    atoms = copy.deepcopy(atoms)

    magmoms_init = atoms.get_magnetic_moments()
    magmoms_new = np.delete(magmoms_init, [index_i])

    atoms.pop(i=index_i)
    atoms.set_initial_magnetic_moments(magmoms_new)


    # #########################################################
    #atoms.write("new_temp.traj")
    # atoms.write("new_temp.json")

    return(atoms)
    #__|

def plot_atoms(
    atoms=None,
    color_array=None,
    out_plot_name="atoms_vis",
    ):
    """Plot 3d plotly scatter plot of atoms object.

    `color_array` will be used to color scatter markers
    """
    #| - plot_atoms

    assert atoms is not None, "Must provide atoms object"

    # Read Atoms Object
    # atoms = io.read("atoms.traj")

    # #########################################################
    # magmoms = atoms.get_magnetic_moments()


    # Marker sizes based on atom identity

    marker_sizes = []
    for atom in atoms:
        if atom.symbol == "Ir":
            # marker_sizes.append(48)
            marker_sizes.append(30)
        elif atom.symbol == "O":
            marker_sizes.append(20)

    cell = atoms.cell


    #| - Plotting

    #| - Creating unit cell traces
    data = []
    for i in range(3):

        x = [0, cell[i][0]]
        y = [0, cell[i][1]]
        z = [0, cell[i][2]]

        trace_i = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            )
        data.append(trace_i)
    #__|

    #| - Creating main scatter point traces
    # Plotting
    x = atoms.positions[:,0]
    y = atoms.positions[:,1]
    z = atoms.positions[:,2]


    trace_i = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=marker_sizes,
            # color=magmoms,
            color=color_array,

            # colorscale="Viridis",
            # colorscale="Jet",
            # colorscale="Hot",
            # colorscale="Picnic",
            # colorscale="Blackbody",
            colorscale="Cividis",

            showscale=True,
            opacity=0.95,
            ),
        )
    data.append(trace_i)
    #__|

    # Scene
    scene = dict(
        # the default values are 1.25, 1.25, 1.25
        camera=dict(
            eye=dict(x=1.15, y=1.15, z=0.8),
            ),
        xaxis=dict(),
        yaxis=dict(),
        zaxis=dict(),
        #  This string can be 'data', 'cube', 'auto', 'manual'
        # aspectmode=string,
        aspectmode="data",

        # A custom aspectratio is defined as follows:
        # aspectratio=dict(x=1, y=1, z=0.95)
        )

    layout = go.Layout(scene=scene)

    fig = go.Figure(
        data=data,
        layout=layout,
        )
    # fig.show()

    #| - Saving plot to file
    from plotting.my_plotly import my_plotly_plot

    my_plotly_plot(
        figure=fig,
        # plot_name="atoms_vis",
        plot_name=out_plot_name,
        write_html=True,
        # write_png=False,
        # png_scale=6.0,
        # write_pdf=False,
        # write_svg=False,
        # try_orca_write=False,
        )
    #__|

    #__|

    #__|





def remove_protruding_bottom_Os(
    atoms=None,
    df_coord=None,

    dz=0.75,
    angle_thresh=30,
    ):
    """
    """
    #| - remove_protruding_bottom_Os
    df_coord_i = df_coord

    # df_coord_i = get_df_coord(
    #     slab_id=slab_id,
    #     slab=atoms,
    #     mode="slab")





    # Getting min/max z position
    z_pos = atoms.positions[:, 2]

    z_min = z_pos.min()
    z_max = z_pos.max()

    # Creating partial slab of slab bottom
    # atoms_bottom_sliver = atoms[atoms.positions[:, 2] < (z_min + dz)]

    # print("TEMP")
    # atoms.write("out_data/remove_bottom_oxy/atoms_orig.cif")
    # atoms_bottom_sliver.write("out_data/remove_bottom_oxy/temp_0.cif")








    data_dict_list = []
    for atom in atoms:
        # #####################################################
        data_dict_i = dict()
        # #####################################################
        index_i = atom.index
        symbol_i = atom.symbol
        pos_i = atom.position
        # #####################################################

        if symbol_i == "O":
            tmp = 42

            # #################################################
            data_dict_i["index"] = index_i
            data_dict_i["pos"] = pos_i
            data_dict_i["pos_z"] = pos_i[2]
            # #################################################
            data_dict_list.append(data_dict_i)
            # #################################################

    df_o_pos = pd.DataFrame(data_dict_list)

    row_min = df_o_pos.sort_values("pos_z").iloc[0]
    z_min_o = row_min.pos_z

    # #########################################################






    o_indices_to_remove = []
    for atom in atoms:
        # #####################################################
        index_i = atom.index
        symbol_i = atom.symbol
        pos_i = atom.position
        # #####################################################

        if pos_i[2] < (z_min_o + dz):
            if symbol_i == "O":
                # #############################################
                row_coord_j = df_coord_i[df_coord_i.structure_index == index_i]
                row_coord_j = row_coord_j.iloc[0]
                # #############################################
                nn_info_j = row_coord_j.nn_info
                # #############################################

    #             print(20 * "-")
    #             print("len(nn_info_j):", len(nn_info_j))
    #             print("index_i:", index_i)

                any_ir_o_angles_less_than_thresh = False
                for nn_info_k in nn_info_j:
    #                 print(15 * "+")
                    # #########################################
                    site_k = nn_info_k["site"]
                    # #########################################
                    coords_k = site_k.coords
                    site_index_k = nn_info_k["site_index"]
                    # #########################################

                    metal_to_o_vect = coords_k - pos_i
                    angle_rad = angle_between(
                        metal_to_o_vect,
                        [0, 0, 1],
                        )
                    angle_deg = math.degrees(angle_rad)

                    if angle_deg < angle_thresh:
                        o_indices_to_remove.append(index_i)
                        any_ir_o_angles_less_than_thresh = True

    #                 print("site_index_k:", site_index_k)
    #                 print("angle_deg:", angle_deg)
    #                 print("")

    #             print("")
    #             print("")
    #             print("")

    atoms_new = copy.deepcopy(atoms)

    indices_to_remove = o_indices_to_remove

    mask = []
    for atom in atoms_new:
        if atom.index in indices_to_remove:
            mask.append(True)
        else:
            mask.append(False)

    del atoms_new[mask]

    return(atoms_new)
    #__|

#__|

# #########################################################
#| - __misc__

def CountFrequency(my_list):
    """
    """
    #| - CountFrequency
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return(freq)
    #__|

@contextmanager
def cwd(path):
    """
    """
    #| - cwd
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)
    #__|

# Custom exception for the timeout
class TimeoutException(Exception):
    pass

# Handler function to be called when SIGALRM is received
def sigalrm_handler(signum, frame):
    # We get signal!
    raise TimeoutException()

def convert_str_facet_to_list(facet):
    """
    """
    #| - convert_str_facet_to_list

    if type(facet) == str:
        assert len(facet) == 3, "Facet string must be only 3 in lenght, otherwise I'm not sure how to process"

        # [facet[0], facet[1], facet[2], ]

        facet_list = []
        for str_i in facet:
            facet_list.append(int(str_i))
        facet_out = facet_list

        # print(facet_out)
    else:
        facet_out = facet

    return(facet_out)
    #__|


def are_dicts_the_same(dict_i, dict_j):
    """
    """
    #| - are_dicts_the_same
    atom_index_mapping_i = dict_i
    atom_index_mapping_j = dict_j

    # #########################################################
    indices_keys_i = list(np.sort(list(atom_index_mapping_i.keys())))
    num_mappings_i = len(indices_keys_i)

    indices_keys_j = list(np.sort(list(atom_index_mapping_j.keys())))
    num_mappings_j = len(indices_keys_j)


    # mess_i = "Mapping dicts need to be the same size to be equal"
    # assert num_mappings_i == num_mappings_j, mess_i

    num_mappings_same = False
    if num_mappings_i == num_mappings_j:
        num_mappings_same = True

    # mess_i = "The keys for mapping must be identical, this is a prerequisite for being equal"
    # assert indices_keys_i == indices_keys_j



    indices_keys_same = False
    if indices_keys_i == indices_keys_j:
        indices_keys_same = True
        indices_keys = indices_keys_i

    mapping_equal_list = []
    for ind_k in indices_keys:
        map_i = atom_index_mapping_i[ind_k]
        map_j = atom_index_mapping_j[ind_k]

        mapping_equal_k = map_i == map_j
        mapping_equal_list.append(mapping_equal_k)

    mappings_all_same = all(mapping_equal_list)

    dicts_are_the_same = False
    if mappings_all_same and num_mappings_same and indices_keys_same:
        dicts_are_the_same = True

    return(dicts_are_the_same)
    #__|

import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    #| - angle_between
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    #__|

def isnotebook():
    """Figure out whether you're in a notebook or  not.

    Useful for progress bars, which need different imports depending on context
    """
    #| - isnotebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    #__|

def create_name_str_from_tup(name_tup):
    """
    """
    #| - create_name_str_from_tup
    name_list = []
    for i in name_tup:
        if type(i) == int or type(i) == float:
            name_list.append(str(int(i)))
        elif type(i) == str:
            name_list.append(i)
        else:
            name_list.append(str(i))
    name_i = "__".join(name_list)
    # name_i += "___" + init_or_final + "___" + str(intact) + ".pickle"
    # name_i += ".pickle"

    return(name_i)
    #__|

#__|


def get_systems_to_stop_run_indices(
    df_jobs_anal=None,
    ):
    """
    """
    #| - get_systems_to_stop_run_indices
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/in_data",
        "systems_to_stop_running.json")
    with open(path_i, "r") as fle:
        systems_to_stop_running = json.load(fle)








    indices_to_stop_running = []
    for sys_i in systems_to_stop_running:

        # #########################################################
        compenv_i = sys_i[0]
        slab_id_i = sys_i[1]
        active_site_i = sys_i[2]
        att_num_i = sys_i[3]
        # #########################################################

        adsorbates = ["bare", "o", "oh"]
        for ads_j in adsorbates:
            index_i = (
                compenv_i,
                slab_id_i,
                ads_j,
                active_site_i,
                att_num_i,
                )
            indices_to_stop_running.append(index_i)

            if ads_j == "o":
                if active_site_i != "NaN":
                    active_site_tmp = "NaN"
                    index_i = (
                        compenv_i,
                        slab_id_i,
                        ads_j,
                        active_site_tmp,
                        att_num_i,
                        )
                    indices_to_stop_running.append(index_i)

    indices_to_stop_running_2 = []
    for index_i in indices_to_stop_running:
        # index_in_df = index_i in df_resubmit.index
        index_in_df = index_i in df_jobs_anal.index
        if index_in_df:
            indices_to_stop_running_2.append(index_i)

    # df_resubmit = df_resubmit.drop(index=indices_to_stop_running_2)
    df_jobs_anal = df_jobs_anal.drop(index=indices_to_stop_running_2)

    return(indices_to_stop_running_2)
    # return(df_jobs_anal)
    #__|

def compare_facets_for_being_the_same(
    facet_0,
    facet_1,
    ):
    """
    Checks whether facet_0 and facet_1 differ only by an integer multiplicative.
    """
    #| - compare_facets_for_being_the_same
    # #########################################################
    facet_j = facet_0
    facet_k = facet_1


    if len(facet_j) != len(facet_k):
        duplicate_found = False
        return(duplicate_found)


    # #########################################################
    facet_j_abs = [np.abs(i) for i in facet_j]
    facet_j_sum = np.sum(facet_j_abs)

    # #########################################################
    facet_k_abs = [np.abs(i) for i in facet_k]
    facet_k_sum = np.sum(facet_k_abs)

    # #########################################################
    if facet_j_sum > facet_k_sum:
        # facet_j_abs / facet_k_abs

        facet_larger = facet_j_abs
        facet_small = facet_k_abs
    else:
        facet_larger = facet_k_abs
        facet_small = facet_j_abs

    # #########################################################
    facet_frac = np.array(facet_larger) / np.array(facet_small)

    # #####################################################
    something_wrong = False
    all_terms_are_whole_nums = True
    # #####################################################
    div_ints =  []
    # #####################################################
    for i_cnt, i in enumerate(facet_frac):
        # print(i.is_integer())
        if np.isnan(i):
            if facet_j_abs[i_cnt] != 0 or facet_k_abs[i_cnt] != 0:
                something_wrong = True
                print("Not good, these should both be zero")

        elif not i.is_integer() or i == 0:
            all_terms_are_whole_nums = False
            # print("Not a whole number here")

        elif i.is_integer():
            div_ints.append(int(i))

    all_int_factors_are_same = False
    if len(list(set(div_ints))) == 1:
        all_int_factors_are_same = True

    duplicate_found = False
    if all_terms_are_whole_nums and not something_wrong and all_int_factors_are_same:
        duplicate_found = True
        # print("Found a duplicate facet here")

    return(duplicate_found)
    #__|




def temp_job_test():
    """
    """
    #| - temp_job_test
    print("This is just a test")
    return(42)
    #__|
