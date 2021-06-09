# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Collect DFT data into OER sets 
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
import random
import itertools
from pathlib import Path

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import plotly.graph_objs as go

# #########################################################
from IPython.display import display

from plotting.my_plotly import my_plotly_plot

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_jobs_data,
    get_df_slabs_to_run,
    get_df_jobs_oh_anal,
    get_df_atoms_sorted_ind,
    get_df_features_targets,
    get_df_magmom_drift,
    get_df_jobs,
    get_df_struct_drift,
    )

# #########################################################
from local_methods import (
    get_oer_triplet__low_e,
    get_oer_triplet__from_oh,
    get_oer_triplet__magmom,

    get_group_w_all_ads,
    are_any_ads_done,
    calc_ads_e,
    get_oer_triplet,
    )

# from local_methods import 
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/collect_collate_dft_data")

# ### Read Data

# +
# #########################################################
df_jobs = get_df_jobs()

# #########################################################
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

# #########################################################
df_jobs_data = get_df_jobs_data()

# #########################################################
df_slabs_to_run = get_df_slabs_to_run()
df_slabs_to_run = df_slabs_to_run.set_index(
    ["compenv", "slab_id", "att_num"], drop=False)

# #########################################################
df_jobs_oh_anal = get_df_jobs_oh_anal()
df_jobs_oh_anal = df_jobs_oh_anal.set_index(["compenv", "slab_id", "active_site"])

# #########################################################
df_atoms_sorted = get_df_atoms_sorted_ind()

# #########################################################
df_features_targets = get_df_features_targets()

# #########################################################
df_struct_drift = get_df_struct_drift()

# #########################################################
df_magmom_drift = get_df_magmom_drift()
# -

# ### Filtering `df_jobs_anal` to only `oer_adsorbate` job types

# +
df_ind = df_jobs_anal_i.index.to_frame()

df_jobs_anal_i = df_jobs_anal_i.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_jobs_anal_i = df_jobs_anal_i.droplevel(level=0)
# -

# ### Filter columns in `df_jobs_anal_i`

# +
from misc_modules.pandas_methods import drop_columns

cols_to_keep = [
    'job_id_max',
    # 'timed_out',
    # 'completed',
    # 'brmix_issue',
    # 'job_understandable',
    # 'decision',
    # 'dft_params_new',
    'job_completely_done',
    ]

df_jobs_anal_i = drop_columns(
    df=df_jobs_anal_i,
    columns=cols_to_keep,
    keep_or_drop="keep",
    )


# + active=""
#
#
# -

# ### Grafting `pot_e` and `as_is_nan` to dataframe

# +
def method(row_i):
    # #####################################################
    new_column_values_dict = {
        "pot_e": None,
        "as_is_nan": None,
        }
    # #####################################################
    compenv_i = row_i.name[0]
    slab_id_i = row_i.name[1]
    ads_i = row_i.name[2]
    active_site_i = row_i.name[3]
    att_num_i = row_i.name[4]
    # #####################################################
    job_id_max_i = row_i.job_id_max
    job_completely_done_i = row_i.job_completely_done
    # #####################################################

    as_is_nan = False
    if active_site_i == "NaN":
        as_is_nan = True

    # #####################################################
    row_data_i = df_jobs_data.loc[job_id_max_i]
    # #####################################################
    pot_e_i = row_data_i.pot_e
    rerun_from_oh_i = row_data_i.rerun_from_oh
    # #####################################################


    # #####################################################
    new_column_values_dict["pot_e"] = pot_e_i
    new_column_values_dict["as_is_nan"] = as_is_nan
    new_column_values_dict["rerun_from_oh"] = rerun_from_oh_i
    # #####################################################
    for key, value in new_column_values_dict.items():
        row_i[key] = value
    # #####################################################
    return(row_i)

df_jobs_anal_i = df_jobs_anal_i.apply(
    method,
    axis=1,
    )
# -

# ### Removing O slabs from dataframe

# +
# #########################################################
# Remove the *O slabs for now
# The fact that they have NaN active sites will mess up the groupby
ads_list = df_jobs_anal_i.index.get_level_values("ads").tolist()
ads_list_no_o = [i for i in list(set(ads_list)) if i != "o"]

idx = pd.IndexSlice
df_jobs_anal_no_o = df_jobs_anal_i.loc[idx[:, :, ads_list_no_o, :, :], :]
# -

# ### Removing rows whose atoms failed to sort

# +
df_atoms_sorted_i = df_atoms_sorted[df_atoms_sorted.index.to_frame().job_type == "oer_adsorbate"] 
df_atoms_sorted_i = df_atoms_sorted_i.droplevel(level=0)

df_atoms_sorted_i = df_atoms_sorted_i[df_atoms_sorted_i.failed_to_sort == True]


# Dropping rows that have failed to sort atoms objects
df_jobs_anal_no_o = df_jobs_anal_no_o.drop(df_atoms_sorted_i.index)
# -

# # `get_oer_triplet__low_e`

# +
path_i = os.path.join(
    root_dir, "out_data",
    "df_ads__low_e.pickle",
    )

my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_ads__low_e = pickle.load(fle)
else:
# if True:

    # #########################################################
    data_dict_list = []
    # #########################################################
    groupby_cols = ["compenv", "slab_id", "active_site", ]
    grouped = df_jobs_anal_no_o.groupby(groupby_cols)
    for name_i, group in grouped:

    # # #########################################################
    # if True:
    #     name_i = ('sherlock', 'vipikema_98', 47.0)
    #     # #####################################################
    #     group = grouped.get_group(name_i)

        # print(name_i)

        # #####################################################
        ads_e_o_i = None
        ads_e_oh_i = None
        job_id_o_i = None
        job_id_oh_i  = None
        job_id_bare_i = None
        all_jobs_in_group_done = None
        any_bare_done = None
        any_oh_done = None
        any_o_done = None
        any_o_done_with_active_sites = None
        # #####################################################


        # #####################################################
        data_dict_i = dict()
        # #####################################################
        name_dict_i = dict(zip(groupby_cols, name_i))
        # #####################################################
        compenv_i = name_i[0]
        slab_id_i = name_i[1]
        active_site_i = name_i[2]
        # #####################################################


        out_dict = get_group_w_all_ads(
            name=name_i,
            group=group,
            df_jobs_anal_i=df_jobs_anal_i,
            )
        group_i = out_dict["group_i"]
        any_o_done_with_active_sites = out_dict["any_o_done_with_active_sites"]


        all_jobs_in_group_done = group_i.job_completely_done.all()

        # Check that potential energy is numerical
        for i in group_i.pot_e.tolist():
            if type(i) != float:
                print("A non-numerical potential energy entered WF: ", name_i)


        # Only consider done jobs from here
        group_done_i = group_i[group_i.job_completely_done == True]

        group_ind_i = group_done_i.index.to_frame()


        # #####################################################
        necessary_ads_present = False
        # #####################################################
        o_avail = "o" in group_ind_i.ads.tolist()
        oh_avail = "oh" in group_ind_i.ads.tolist()
        bare_avail = "bare" in group_ind_i.ads.tolist()
        # #####################################################
        if o_avail and oh_avail and bare_avail:
            necessary_ads_present = True
        # #####################################################

        if necessary_ads_present:

            oer_trip_i = get_oer_triplet__low_e(
                name=name_i,
                group=group_done_i,
                )

            # oer_trip_i = get_oer_triplet(
            #     name=name_i,
            #     # group=group_i,
            #     group=group_done_i,
            #     df_jobs_oh_anal=df_jobs_oh_anal,
            #     # heuristic__if_lower_e=False,
            #     heuristic__if_lower_e=True,
            #     )



            # #################################################
            idx = pd.IndexSlice
            row_o_i = oer_trip_i.loc[idx[:, :, "o", :, :], :].iloc[0]
            job_id_o_i = row_o_i.job_id_max

            # #################################################
            idx = pd.IndexSlice
            row_oh_i = oer_trip_i.loc[idx[:, :, "oh", :, :], :].iloc[0]
            job_id_oh_i = row_oh_i.job_id_max

            # #################################################
            idx = pd.IndexSlice
            row_bare_i = oer_trip_i.loc[idx[:, :, "bare", :, :], :].iloc[0]
            job_id_bare_i = row_bare_i.job_id_max

            # #################################################
            # Computing adsorption energy
            df_ads_i = calc_ads_e(oer_trip_i.reset_index())
            df_ads_i = df_ads_i.set_index("ads", drop=False)

            ads_g_o_i = df_ads_i.loc["o"]["ads_e"]
            ads_e_o_i = df_ads_i.loc["o"]["ads_e_elec"]

            ads_g_oh_i = df_ads_i.loc["oh"]["ads_e"]
            ads_e_oh_i = df_ads_i.loc["oh"]["ads_e_elec"]
            job_id_oh_i = df_ads_i.loc["oh"]["job_id_max"]


            # #################################################
            data_dict_i.update(name_dict_i)
            # #################################################
            data_dict_i["g_o"] = ads_g_o_i
            data_dict_i["g_oh"] = ads_g_oh_i
            data_dict_i["e_o"] = ads_e_o_i
            data_dict_i["e_oh"] = ads_e_oh_i
            data_dict_i["job_id_o"] = job_id_o_i
            data_dict_i["job_id_oh"] = job_id_oh_i 
            data_dict_i["job_id_bare"] = job_id_bare_i
            data_dict_i["all_done"] = all_jobs_in_group_done
            data_dict_i["any_bare_done"] = any_bare_done
            data_dict_i["any_oh_done"] = any_oh_done
            data_dict_i["any_o_done"] = any_o_done
            data_dict_i["any_o_w_as_done"] = any_o_done_with_active_sites
            # #################################################
            data_dict_list.append(data_dict_i)
            # #################################################


    # #########################################################
    df_ads__low_e = pd.DataFrame(data_dict_list)
    # #########################################################

    # #########################################################
    # Pickling data ###########################################
    directory = os.path.join(
        root_dir, "out_data")
    if not os.path.exists(directory): os.makedirs(directory)
    with open(os.path.join(directory, "df_ads__low_e.pickle"), "wb") as fle:
        pickle.dump(df_ads__low_e, fle)
    # #########################################################
# -

# # `get_oer_triplet__from_oh`

# +
path_i = os.path.join(
    root_dir, "out_data",
    "df_ads__from_oh.pickle",
    )

my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_ads__from_oh = pickle.load(fle)
else:
# if True:

    # #########################################################
    data_dict_list = []
    # #########################################################
    groupby_cols = ["compenv", "slab_id", "active_site", ]
    grouped = df_jobs_anal_no_o.groupby(groupby_cols)
    for name_i, group in grouped:

    # if True:
    #     # name_i = ('sherlock', 'vipikema_98', 47.0)
    #     # name_i = ('nersc', 'kalisule_45', 62.0)
    #     # name_i = ('sherlock', 'momaposi_60', 50.0)
    #     name_i = ('nersc', 'fosurufu_23', 43.0)
    #     group = grouped.get_group(name_i)


        # print(name_i)

        # #####################################################
        ads_e_o_i = None
        ads_e_oh_i = None
        job_id_o_i = None
        job_id_oh_i  = None
        job_id_bare_i = None
        all_jobs_in_group_done = None
        any_bare_done = None
        any_oh_done = None
        any_o_done = None
        any_o_done_with_active_sites = None
        # #####################################################


        # #####################################################
        data_dict_i = dict()
        # #####################################################
        name_dict_i = dict(zip(groupby_cols, name_i))
        # #####################################################
        compenv_i = name_i[0]
        slab_id_i = name_i[1]
        active_site_i = name_i[2]
        # #####################################################


        out_dict = get_group_w_all_ads(
            name=name_i,
            group=group,
            df_jobs_anal_i=df_jobs_anal_i,
            )
        group_i = out_dict["group_i"]
        any_o_done_with_active_sites = out_dict["any_o_done_with_active_sites"]


        all_jobs_in_group_done = group_i.job_completely_done.all()


        # Check that potential energy is numerical
        for i in group_i.pot_e.tolist():
            if type(i) != float:
                print("A non-numerical potential energy entered WF: ", name_i)


        # Only consider done jobs from here
        group_done_i = group_i[group_i.job_completely_done == True]

        group_ind_i = group_done_i.index.to_frame()


        # #####################################################
        necessary_ads_present = False
        # #####################################################
        o_avail = "o" in group_ind_i.ads.tolist()
        oh_avail = "oh" in group_ind_i.ads.tolist()
        bare_avail = "bare" in group_ind_i.ads.tolist()
        # #####################################################
        if o_avail and oh_avail and bare_avail:
            necessary_ads_present = True
        # #####################################################

        if necessary_ads_present:

            oer_trip_dict_i = get_oer_triplet__from_oh(
                name=name_i,
                group=group_done_i,
                df_jobs_oh_anal=df_jobs_oh_anal,
                )
            oer_trip_i = oer_trip_dict_i["df_oer_triplet"]
            error = oer_trip_dict_i["error"]
            note = oer_trip_dict_i["note"]

            # # TEMP
            # break


            ads_g_o_i = None
            ads_g_oh_i = None
            ads_e_o_i = None
            ads_e_oh_i = None
            if not error:
                # #################################################
                idx = pd.IndexSlice
                row_o_i = oer_trip_i.loc[idx[:, :, "o", :, :], :].iloc[0]
                job_id_o_i = row_o_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_oh_i = oer_trip_i.loc[idx[:, :, "oh", :, :], :].iloc[0]
                job_id_oh_i = row_oh_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_bare_i = oer_trip_i.loc[idx[:, :, "bare", :, :], :].iloc[0]
                job_id_bare_i = row_bare_i.job_id_max

                # #################################################
                # COmputing adsorption energy
                df_ads_i = calc_ads_e(oer_trip_i.reset_index())
                df_ads_i = df_ads_i.set_index("ads", drop=False)

                ads_g_o_i = df_ads_i.loc["o"]["ads_e"]
                ads_e_o_i = df_ads_i.loc["o"]["ads_e_elec"]

                ads_g_oh_i = df_ads_i.loc["oh"]["ads_e"]
                ads_e_oh_i = df_ads_i.loc["oh"]["ads_e_elec"]
                job_id_oh_i = df_ads_i.loc["oh"]["job_id_max"]


            # #################################################
            data_dict_i.update(name_dict_i)
            # #################################################
            data_dict_i["g_o"] = ads_g_o_i
            data_dict_i["g_oh"] = ads_g_oh_i
            data_dict_i["e_o"] = ads_e_o_i
            data_dict_i["e_oh"] = ads_e_oh_i
            data_dict_i["job_id_o"] = job_id_o_i
            data_dict_i["job_id_oh"] = job_id_oh_i 
            data_dict_i["job_id_bare"] = job_id_bare_i
            data_dict_i["all_done"] = all_jobs_in_group_done
            data_dict_i["any_bare_done"] = any_bare_done
            data_dict_i["any_oh_done"] = any_oh_done
            data_dict_i["any_o_done"] = any_o_done
            data_dict_i["any_o_w_as_done"] = any_o_done_with_active_sites
            data_dict_i["error"] = error
            data_dict_i["note"] = note
            # #################################################
            data_dict_list.append(data_dict_i)
            # #################################################


    # #########################################################
    df_ads__from_oh = pd.DataFrame(data_dict_list)
    # #########################################################



    # #########################################################
    # Pickling data ###########################################
    directory = os.path.join(
        root_dir, "out_data")
    if not os.path.exists(directory): os.makedirs(directory)
    with open(os.path.join(directory, "df_ads__from_oh.pickle"), "wb") as fle:
        pickle.dump(df_ads__from_oh, fle)
    # #########################################################

# +
# oer_trip_dict_i = get_oer_triplet__from_oh(
#     name=name_i,
#     group=group_done_i,
#     df_jobs_oh_anal=df_jobs_oh_anal,
#     )
# oer_trip_i = oer_trip_dict_i["df_oer_triplet"]
# error = oer_trip_dict_i["error"]
# note = oer_trip_dict_i["note"]

# +
# oer_trip_dict_i

# oer_trip_i

# +
# assert False
# -

# # `get_oer_triplet__magmom`

# +
path_i = os.path.join(
    root_dir, "out_data",
    "df_ads__magmom.pickle",
    )

my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_ads__magmom = pickle.load(fle)
else:
# if True:

    # #########################################################
    data_dict_list = []
    # #########################################################
    groupby_cols = ["compenv", "slab_id", "active_site", ]
    grouped = df_jobs_anal_no_o.groupby(groupby_cols)
    for name_i, group in grouped:

    # if True:
    #     # name_i = ('sherlock', 'vipikema_98', 47.0)
    #     # name_i = ('nersc', 'kalisule_45', 62.0)
    #     # name_i = ('sherlock', 'momaposi_60', 50.0)
    #     name_i = ('nersc', 'fosurufu_23', 43.0)
    #     group = grouped.get_group(name_i)


        # print(name_i)

        # #####################################################
        ads_e_o_i = None
        ads_e_oh_i = None
        job_id_o_i = None
        job_id_oh_i  = None
        job_id_bare_i = None
        all_jobs_in_group_done = None
        any_bare_done = None
        any_oh_done = None
        any_o_done = None
        any_o_done_with_active_sites = None
        # #####################################################


        # #####################################################
        data_dict_i = dict()
        # #####################################################
        name_dict_i = dict(zip(groupby_cols, name_i))
        # #####################################################
        compenv_i = name_i[0]
        slab_id_i = name_i[1]
        active_site_i = name_i[2]
        # #####################################################


        out_dict = get_group_w_all_ads(
            name=name_i,
            group=group,
            df_jobs_anal_i=df_jobs_anal_i,
            )
        group_i = out_dict["group_i"]
        any_o_done_with_active_sites = out_dict["any_o_done_with_active_sites"]


        all_jobs_in_group_done = group_i.job_completely_done.all()


        # Check that potential energy is numerical
        for i in group_i.pot_e.tolist():
            if type(i) != float:
                print("A non-numerical potential energy entered WF: ", name_i)


        # Only consider done jobs from here
        group_done_i = group_i[group_i.job_completely_done == True]

        group_ind_i = group_done_i.index.to_frame()


        # #####################################################
        necessary_ads_present = False
        # #####################################################
        o_avail = "o" in group_ind_i.ads.tolist()
        oh_avail = "oh" in group_ind_i.ads.tolist()
        bare_avail = "bare" in group_ind_i.ads.tolist()
        # #####################################################
        if o_avail and oh_avail and bare_avail:
            necessary_ads_present = True
        # #####################################################

        if necessary_ads_present:
            tmp = 42

            oer_trip_dict_i = get_oer_triplet__magmom(
                name=name_i,
                group=group_done_i,
                df_jobs=df_jobs,
                df_jobs_oh_anal=df_jobs_oh_anal,
                df_magmom_drift=df_magmom_drift,
                )
            oer_trip_i = oer_trip_dict_i["df_oer_triplet"]
            error = oer_trip_dict_i["error"]
            note = oer_trip_dict_i["note"]

            # oer_trip_dict_i = get_oer_triplet__magmom(
            #     name=name_i,
            #     group=group_done_i,
            #     df_jobs_oh_anal=df_jobs_oh_anal,
            #     )
            # oer_trip_i = oer_trip_dict_i["df_oer_triplet"]
            # error = oer_trip_dict_i["error"]
            # note = oer_trip_dict_i["note"]


            ads_g_o_i = None
            ads_g_oh_i = None
            ads_e_o_i = None
            ads_e_oh_i = None
            if not error:
                # #################################################
                idx = pd.IndexSlice
                row_o_i = oer_trip_i.loc[idx[:, :, "o", :, :], :].iloc[0]
                job_id_o_i = row_o_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_oh_i = oer_trip_i.loc[idx[:, :, "oh", :, :], :].iloc[0]
                job_id_oh_i = row_oh_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_bare_i = oer_trip_i.loc[idx[:, :, "bare", :, :], :].iloc[0]
                job_id_bare_i = row_bare_i.job_id_max

                # #################################################
                # COmputing adsorption energy
                df_ads_i = calc_ads_e(oer_trip_i.reset_index())
                df_ads_i = df_ads_i.set_index("ads", drop=False)

                ads_g_o_i = df_ads_i.loc["o"]["ads_e"]
                ads_e_o_i = df_ads_i.loc["o"]["ads_e_elec"]

                ads_g_oh_i = df_ads_i.loc["oh"]["ads_e"]
                ads_e_oh_i = df_ads_i.loc["oh"]["ads_e_elec"]
                job_id_oh_i = df_ads_i.loc["oh"]["job_id_max"]


            # #################################################
            data_dict_i.update(name_dict_i)
            # #################################################
            data_dict_i["g_o"] = ads_g_o_i
            data_dict_i["g_oh"] = ads_g_oh_i
            data_dict_i["e_o"] = ads_e_o_i
            data_dict_i["e_oh"] = ads_e_oh_i
            data_dict_i["job_id_o"] = job_id_o_i
            data_dict_i["job_id_oh"] = job_id_oh_i 
            data_dict_i["job_id_bare"] = job_id_bare_i
            data_dict_i["all_done"] = all_jobs_in_group_done
            data_dict_i["any_bare_done"] = any_bare_done
            data_dict_i["any_oh_done"] = any_oh_done
            data_dict_i["any_o_done"] = any_o_done
            data_dict_i["any_o_w_as_done"] = any_o_done_with_active_sites
            data_dict_i["error"] = error
            data_dict_i["note"] = note
            # #################################################
            data_dict_list.append(data_dict_i)
            # #################################################


    # #########################################################
    df_ads__magmom = pd.DataFrame(data_dict_list)
    # #########################################################



    # #########################################################
    # Pickling data ###########################################
    directory = os.path.join(
        root_dir, "out_data")
    if not os.path.exists(directory): os.makedirs(directory)
    with open(os.path.join(directory, "df_ads__magmom.pickle"), "wb") as fle:
        pickle.dump(df_ads__magmom, fle)
    # #########################################################
# -

# # `my_oer_picker`

# +
path_i = os.path.join(
    root_dir, "out_data",
    "df_ads__mine.pickle",
    )

my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_ads__mine = pickle.load(fle)
else:
# if True:

    # #########################################################
    data_dict_list = []
    # #########################################################
    groupby_cols = ["compenv", "slab_id", "active_site", ]
    grouped = df_jobs_anal_no_o.groupby(groupby_cols)
    for name_i, group in grouped:

    # if True:
    #     # name_i = ('sherlock', 'vipikema_98', 47.0)
    #     # name_i = ('nersc', 'kalisule_45', 62.0)
    #     # name_i = ('sherlock', 'momaposi_60', 50.0)
    #     name_i = ('nersc', 'fosurufu_23', 43.0)
    #     group = grouped.get_group(name_i)


        # print(name_i)

        # #####################################################
        ads_e_o_i = None
        ads_e_oh_i = None
        job_id_o_i = None
        job_id_oh_i  = None
        job_id_bare_i = None
        all_jobs_in_group_done = None
        any_bare_done = None
        any_oh_done = None
        any_o_done = None
        any_o_done_with_active_sites = None
        # #####################################################


        # #####################################################
        data_dict_i = dict()
        # #####################################################
        name_dict_i = dict(zip(groupby_cols, name_i))
        # #####################################################
        compenv_i = name_i[0]
        slab_id_i = name_i[1]
        active_site_i = name_i[2]
        # #####################################################


        out_dict = get_group_w_all_ads(
            name=name_i,
            group=group,
            df_jobs_anal_i=df_jobs_anal_i,
            )
        group_i = out_dict["group_i"]
        any_o_done_with_active_sites = out_dict["any_o_done_with_active_sites"]


        all_jobs_in_group_done = group_i.job_completely_done.all()


        # Check that potential energy is numerical
        for i in group_i.pot_e.tolist():
            if type(i) != float:
                print("A non-numerical potential energy entered WF: ", name_i)


        # Only consider done jobs from here
        group_done_i = group_i[group_i.job_completely_done == True]

        group_ind_i = group_done_i.index.to_frame()


        # #####################################################
        necessary_ads_present = False
        # #####################################################
        o_avail = "o" in group_ind_i.ads.tolist()
        oh_avail = "oh" in group_ind_i.ads.tolist()
        bare_avail = "bare" in group_ind_i.ads.tolist()
        # #####################################################
        if o_avail and oh_avail and bare_avail:
            necessary_ads_present = True
        # #####################################################

        if necessary_ads_present:
            # tmp = 42
            # get_oer_triplet
            # oer_trip_dict_i = get_oer_triplet__magmom(


            oer_trip_dict_i = get_oer_triplet(
                name=name_i,
                group=group_done_i,
                df_jobs_oh_anal=df_jobs_oh_anal,
                heuristic__if_lower_e=True,
                )
            oer_trip_i = oer_trip_dict_i

            # oer_trip_i = oer_trip_dict_i["df_oer_triplet"]
            # error = oer_trip_dict_i["error"]
            # note = oer_trip_dict_i["note"]

            error = False


            ads_g_o_i = None
            ads_g_oh_i = None
            ads_e_o_i = None
            ads_e_oh_i = None
            # if not error:
            if "oh" in oer_trip_i.index.to_frame()["ads"].unique().tolist():
                # #################################################
                idx = pd.IndexSlice
                row_o_i = oer_trip_i.loc[idx[:, :, "o", :, :], :].iloc[0]
                job_id_o_i = row_o_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_oh_i = oer_trip_i.loc[idx[:, :, "oh", :, :], :].iloc[0]
                job_id_oh_i = row_oh_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_bare_i = oer_trip_i.loc[idx[:, :, "bare", :, :], :].iloc[0]
                job_id_bare_i = row_bare_i.job_id_max

                # #################################################
                # COmputing adsorption energy
                df_ads_i = calc_ads_e(oer_trip_i.reset_index())
                df_ads_i = df_ads_i.set_index("ads", drop=False)

                ads_g_o_i = df_ads_i.loc["o"]["ads_e"]
                ads_e_o_i = df_ads_i.loc["o"]["ads_e_elec"]

                ads_g_oh_i = df_ads_i.loc["oh"]["ads_e"]
                ads_e_oh_i = df_ads_i.loc["oh"]["ads_e_elec"]
                job_id_oh_i = df_ads_i.loc["oh"]["job_id_max"]


            # #################################################
            data_dict_i.update(name_dict_i)
            # #################################################
            data_dict_i["g_o"] = ads_g_o_i
            data_dict_i["g_oh"] = ads_g_oh_i
            data_dict_i["e_o"] = ads_e_o_i
            data_dict_i["e_oh"] = ads_e_oh_i
            data_dict_i["job_id_o"] = job_id_o_i
            data_dict_i["job_id_oh"] = job_id_oh_i 
            data_dict_i["job_id_bare"] = job_id_bare_i
            data_dict_i["all_done"] = all_jobs_in_group_done
            data_dict_i["any_bare_done"] = any_bare_done
            data_dict_i["any_oh_done"] = any_oh_done
            data_dict_i["any_o_done"] = any_o_done
            data_dict_i["any_o_w_as_done"] = any_o_done_with_active_sites
            data_dict_i["error"] = error
            # data_dict_i["note"] = note
            # #################################################
            data_dict_list.append(data_dict_i)
            # #################################################


    # #########################################################
    df_ads__mine = pd.DataFrame(data_dict_list)
    # #########################################################



    # #########################################################
    # Pickling data ###########################################
    directory = os.path.join(
        root_dir, "out_data")
    if not os.path.exists(directory): os.makedirs(directory)
    with open(os.path.join(directory, "df_ads__mine.pickle"), "wb") as fle:
        pickle.dump(df_ads__mine, fle)
    # #########################################################
# -

# # `my_oer_picker_2`

# +
path_i = os.path.join(
    root_dir, "out_data",
    "df_ads__mine_2.pickle",
    )

my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_ads__mine_2 = pickle.load(fle)
else:
# if True:

    # #########################################################
    data_dict_list = []
    # #########################################################
    groupby_cols = ["compenv", "slab_id", "active_site", ]
    grouped = df_jobs_anal_no_o.groupby(groupby_cols)
    for name_i, group in grouped:

    # if True:
    #     # name_i = ('sherlock', 'vipikema_98', 47.0)
    #     # name_i = ('nersc', 'kalisule_45', 62.0)
    #     # name_i = ('sherlock', 'momaposi_60', 50.0)
    #     name_i = ('nersc', 'fosurufu_23', 43.0)
    #     group = grouped.get_group(name_i)


        # print(name_i)

        # #####################################################
        ads_e_o_i = None
        ads_e_oh_i = None
        job_id_o_i = None
        job_id_oh_i  = None
        job_id_bare_i = None
        all_jobs_in_group_done = None
        any_bare_done = None
        any_oh_done = None
        any_o_done = None
        any_o_done_with_active_sites = None
        # #####################################################


        # #####################################################
        data_dict_i = dict()
        # #####################################################
        name_dict_i = dict(zip(groupby_cols, name_i))
        # #####################################################
        compenv_i = name_i[0]
        slab_id_i = name_i[1]
        active_site_i = name_i[2]
        # #####################################################


        out_dict = get_group_w_all_ads(
            name=name_i,
            group=group,
            df_jobs_anal_i=df_jobs_anal_i,
            )
        group_i = out_dict["group_i"]
        any_o_done_with_active_sites = out_dict["any_o_done_with_active_sites"]


        all_jobs_in_group_done = group_i.job_completely_done.all()


        # Check that potential energy is numerical
        for i in group_i.pot_e.tolist():
            if type(i) != float:
                print("A non-numerical potential energy entered WF: ", name_i)


        # Only consider done jobs from here
        group_done_i = group_i[group_i.job_completely_done == True]

        group_ind_i = group_done_i.index.to_frame()


        # #####################################################
        necessary_ads_present = False
        # #####################################################
        o_avail = "o" in group_ind_i.ads.tolist()
        oh_avail = "oh" in group_ind_i.ads.tolist()
        bare_avail = "bare" in group_ind_i.ads.tolist()
        # #####################################################
        if o_avail and oh_avail and bare_avail:
            necessary_ads_present = True
        # #####################################################

        if necessary_ads_present:
            # tmp = 42
            # get_oer_triplet
            # oer_trip_dict_i = get_oer_triplet__magmom(


            oer_trip_dict_i = get_oer_triplet(
                name=name_i,
                group=group_done_i,
                df_jobs_oh_anal=df_jobs_oh_anal,
                heuristic__if_lower_e=False,
                )
            oer_trip_i = oer_trip_dict_i

            # oer_trip_i = oer_trip_dict_i["df_oer_triplet"]
            # error = oer_trip_dict_i["error"]
            # note = oer_trip_dict_i["note"]

            error = False


            ads_g_o_i = None
            ads_g_oh_i = None
            ads_e_o_i = None
            ads_e_oh_i = None
            # if not error:
            if "oh" in oer_trip_i.index.to_frame()["ads"].unique().tolist():
                # #################################################
                idx = pd.IndexSlice
                row_o_i = oer_trip_i.loc[idx[:, :, "o", :, :], :].iloc[0]
                job_id_o_i = row_o_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_oh_i = oer_trip_i.loc[idx[:, :, "oh", :, :], :].iloc[0]
                job_id_oh_i = row_oh_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_bare_i = oer_trip_i.loc[idx[:, :, "bare", :, :], :].iloc[0]
                job_id_bare_i = row_bare_i.job_id_max

                # #################################################
                # COmputing adsorption energy
                df_ads_i = calc_ads_e(oer_trip_i.reset_index())
                df_ads_i = df_ads_i.set_index("ads", drop=False)

                ads_g_o_i = df_ads_i.loc["o"]["ads_e"]
                ads_e_o_i = df_ads_i.loc["o"]["ads_e_elec"]

                ads_g_oh_i = df_ads_i.loc["oh"]["ads_e"]
                ads_e_oh_i = df_ads_i.loc["oh"]["ads_e_elec"]
                job_id_oh_i = df_ads_i.loc["oh"]["job_id_max"]


            # #################################################
            data_dict_i.update(name_dict_i)
            # #################################################
            data_dict_i["g_o"] = ads_g_o_i
            data_dict_i["g_oh"] = ads_g_oh_i
            data_dict_i["e_o"] = ads_e_o_i
            data_dict_i["e_oh"] = ads_e_oh_i
            data_dict_i["job_id_o"] = job_id_o_i
            data_dict_i["job_id_oh"] = job_id_oh_i 
            data_dict_i["job_id_bare"] = job_id_bare_i
            data_dict_i["all_done"] = all_jobs_in_group_done
            data_dict_i["any_bare_done"] = any_bare_done
            data_dict_i["any_oh_done"] = any_oh_done
            data_dict_i["any_o_done"] = any_o_done
            data_dict_i["any_o_w_as_done"] = any_o_done_with_active_sites
            data_dict_i["error"] = error
            # data_dict_i["note"] = note
            # #################################################
            data_dict_list.append(data_dict_i)
            # #################################################


    # #########################################################
    df_ads__mine_2 = pd.DataFrame(data_dict_list)
    # #########################################################



    # #########################################################
    # Pickling data ###########################################
    directory = os.path.join(
        root_dir, "out_data")
    if not os.path.exists(directory): os.makedirs(directory)
    with open(os.path.join(directory, "df_ads__mine_2.pickle"), "wb") as fle:
        pickle.dump(df_ads__mine_2, fle)
    # #########################################################

# +
# df_ads__mine
# df_ads__mine

# +
# oer_trip_dict_i

# +
# assert False

# + active=""
#
#
#
#
#

# +
# df_ads__from_oh.loc[('nersc', 'kalisule_45', 62.0)]
# -

for name_i, row_mine_i in df_ads__mine_2.iterrows():
    job_id_o_1_i = row_mine_i.job_id_o

    row_from_oh_i = df_ads__from_oh.loc[name_i]
    job_id_o_2_i =row_from_oh_i.job_id_o

    if not job_id_o_1_i == job_id_o_2_i:
        print("")
        print(name_i)
        print(job_id_o_1_i, job_id_o_2_i)

# # Comparing different methods

df_ads__magmom = df_ads__magmom.set_index(["compenv", "slab_id", "active_site", ])
df_ads__from_oh = df_ads__from_oh.set_index(["compenv", "slab_id", "active_site", ])
df_ads__low_e = df_ads__low_e.set_index(["compenv", "slab_id", "active_site", ])
df_ads__mine = df_ads__mine.set_index(["compenv", "slab_id", "active_site", ])
df_ads__mine_2 = df_ads__mine_2.set_index(["compenv", "slab_id", "active_site", ])

# +
all_indices = df_ads__low_e.index.tolist() + \
    df_ads__from_oh.index.tolist() + \
    df_ads__magmom.index.tolist()

idx = pd.MultiIndex.from_tuples(all_indices)
idx = idx.drop_duplicates()

unique_indices = idx.tolist()

# +
# #########################################################
data_dict_list = []
# #########################################################
for name_i in unique_indices:
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################

    rows_dict = dict()

    row_magmom__exists = False
    if name_i in df_ads__magmom.index:
        row_magmom_i = df_ads__magmom.loc[name_i]
        rows_dict["magmom"] = row_magmom_i
        row_magmom__exists = True

    row_from_oh__exists = False
    if name_i in df_ads__from_oh.index:
        row_from_oh_i = df_ads__from_oh.loc[name_i]
        rows_dict["from_oh"] = row_from_oh_i
        row_from_oh__exists = True

    row_low_e__exists = False
    if name_i in df_ads__low_e.index:
        row_low_e_i = df_ads__low_e.loc[name_i]
        rows_dict["low_e"] = row_low_e_i
        row_low_e__exists = True

    row_mine__exists = False
    if name_i in df_ads__mine.index:
        row_mine_i = df_ads__mine.loc[name_i]
        rows_dict["mine"] = row_mine_i
        row_mine__exists = True

    row_mine_2__exists = False
    if name_i in df_ads__mine_2.index:
        row_mine_2_i = df_ads__mine_2.loc[name_i]
        rows_dict["mine_2"] = row_mine_2_i
        row_mine_2__exists = True

    # print(

    #     # "\n",
    #     "row_magmom__exists:  ", row_magmom__exists,

    #     "\n",
    #     "row_from_oh__exists: ", row_from_oh__exists,

    #     "\n",
    #     "row_low_e__exists:   ", row_low_e__exists,

    #     sep="")
    # print(20 * "-")
    












    comparisons = dict()
    for key_i, row_i in rows_dict.items():

        for key_j, row_j in rows_dict.items():
            sorted_keys_ij = tuple(np.sort([key_i, key_j]))

            if key_i == key_j:
                continue

            if sorted_keys_ij not in comparisons:

                job_id_o__same = False
                if rows_dict[key_i].job_id_o == rows_dict[key_j].job_id_o:
                    job_id_o__same = True

                job_id_oh__same = False
                if rows_dict[key_i].job_id_oh == rows_dict[key_j].job_id_oh:
                    job_id_oh__same = True

                job_id_bare__same = False
                if rows_dict[key_i].job_id_bare == rows_dict[key_j].job_id_bare:
                    job_id_bare__same = True

                data_ij = dict(
                    job_id_o__same=job_id_o__same,
                    job_id_oh__same=job_id_oh__same,
                    job_id_bare__same=job_id_bare__same,
                    )

                comparisons[sorted_keys_ij] = data_ij



























    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["active_site"] = active_site_i
    # #####################################################
    for key_i, data_ij in comparisons.items():
        name_pre = "__".join(list(key_i))
        for key_j, val_j in data_ij.items():
            new_name = name_pre + "__" + key_j
            data_dict_i[new_name] = val_j
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_oer_trip_comp = pd.DataFrame(data_dict_list)
df_oer_trip_comp = df_oer_trip_comp.set_index(["compenv", "slab_id", "active_site", ])
# #########################################################

# +
new_cols = []
for col_i in df_oer_trip_comp.columns:
    modes = col_i.split("__")[0:2]

    new_col_i = (
        "__".join(modes),
        col_i.split("__")[2],
        )
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_oer_trip_comp.columns = idx
# -

main_0_levels = list(df_oer_trip_comp.columns.levels[0])

# +
for level_0_i in list(df_oer_trip_comp.columns.levels[0]):
    df_oer_trip_comp_i = df_oer_trip_comp[level_0_i]
    df_oer_trip_comp[(level_0_i, "all_True")] = df_oer_trip_comp_i.all(axis=1)

cols_tmp = []
for i in list(df_oer_trip_comp.columns.levels[0]):
    cols_tmp.append((i, "all_True", ))

all_all_True_col = df_oer_trip_comp[
    cols_tmp
    ].all(axis=1)

df_oer_trip_comp[("all_all_True", "", )] = all_all_True_col

# +
df_tmp = df_oer_trip_comp[df_oer_trip_comp.all_all_True == True]

print(
    df_tmp.shape[0],
    " systems have identical OER sets regardless of what method is used",
    sep="")

# +
print(
    df_oer_trip_comp.shape[0],
    " TOTAL SYSTEMS",
    "\n",
    20 * "-",
    sep="")

for main_0_lev_i in main_0_levels:

    df_tmp = df_oer_trip_comp[
        (main_0_lev_i, "all_True", )
        ]

    print(
        df_tmp[df_tmp == True].shape[0],
        " systems have identical OER sets for ",
        main_0_lev_i,
        sep="")
# -

assert False

# # Plotting

# +
data = []

shared_scatter_props = go.Scatter(
    mode="markers",
    marker=go.scatter.Marker(
        opacity=0.7,
        ),
    )


# #########################################################
y_array = df_ads__low_e.g_o
x_array = df_ads__low_e.g_oh
trace_i = go.Scatter(
    x=x_array,
    y=y_array,
    name="low_e",
    )
trace_i.update(
    dict1=shared_scatter_props.to_plotly_json(),
    )
data.append(trace_i)

# #########################################################
y_array = df_ads__from_oh.g_o
x_array = df_ads__from_oh.g_oh
trace_i = go.Scatter(
    x=x_array,
    y=y_array,
    name="from_oh",
    )
trace_i.update(
    dict1=shared_scatter_props.to_plotly_json(),
    )
data.append(trace_i)

# #########################################################
y_array = df_ads__magmom.g_o
x_array = df_ads__magmom.g_oh
trace_i = go.Scatter(
    x=x_array,
    y=y_array,
    name="magmom",
    )
trace_i.update(
    dict1=shared_scatter_props.to_plotly_json(),
    )
data.append(trace_i)


fig = go.Figure(data=data)
fig.show()
# -

assert False

# + active=""
#
#
#
#
# -

# # Doing all triplet combinations

# +
path_i = os.path.join(
    root_dir, "out_data",
    "df_dict.pickle",
    )

my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_dict = pickle.load(fle)
else:


    # #########################################################
    data_dict_list = []
    df_dict = dict()
    # #########################################################
    groupby_cols = ["compenv", "slab_id", "active_site", ]
    grouped = df_jobs_anal_no_o.groupby(groupby_cols)
    for name_i, group in grouped:

    # if True:
    #     name_i = ('sherlock', 'vipikema_98', 47.0)
    #     group = grouped.get_group(name_i)

        # print(20 * "-")
        # print(name_i)

        # #####################################################
        ads_e_o_i = None
        ads_e_oh_i = None
        job_id_o_i = None
        job_id_oh_i  = None
        job_id_bare_i = None
        all_jobs_in_group_done = None
        any_bare_done = None
        any_oh_done = None
        any_o_done = None
        any_o_done_with_active_sites = None
        # #####################################################


        # #####################################################
        # #####################################################
        name_dict_i = dict(zip(groupby_cols, name_i))
        # #####################################################
        compenv_i = name_i[0]
        slab_id_i = name_i[1]
        active_site_i = name_i[2]
        # #####################################################


        out_dict = get_group_w_all_ads(
            name=name_i,
            group=group,
            df_jobs_anal_i=df_jobs_anal_i,
            )
        group_i = out_dict["group_i"]
        any_o_done_with_active_sites = out_dict["any_o_done_with_active_sites"]


        all_jobs_in_group_done = group_i.job_completely_done.all()

        # Check that potential energy is numerical
        for i in group_i.pot_e.tolist():
            if type(i) != float:
                print("A non-numerical potential energy entered WF: ", name_i)


        # Only consider done jobs from here
        group_done_i = group_i[group_i.job_completely_done == True]

        group_ind_i = group_done_i.index.to_frame()


        # #####################################################
        necessary_ads_present = False
        # #####################################################
        o_avail = "o" in group_ind_i.ads.tolist()
        oh_avail = "oh" in group_ind_i.ads.tolist()
        bare_avail = "bare" in group_ind_i.ads.tolist()
        # #####################################################
        if o_avail and oh_avail and bare_avail:
            necessary_ads_present = True
        # #####################################################

        if necessary_ads_present:

            all_triplets = list(itertools.combinations(group_done_i.job_id_max.tolist(), 3))

            data_dict_list = []
            good_triplets = []
            for trip_i in all_triplets:
                df_i = pd.concat([
                    group_done_i.index.to_frame(),
                    group_done_i],
                    axis=1)

                df_i = df_i.set_index("job_id_max")

                df_trip_i = df_i.loc[list(trip_i)]

                num_uniq_ads = len(list(df_trip_i.ads.unique()))

                if num_uniq_ads == 3:
                    good_triplets.append(trip_i)


            # good_triplets = [
            #     ('nowowesi_15', 'nihihagu_67', 'fufohoru_09'),
            #     ('nowowesi_15', 'kofakibu_00', 'kogabeku_65'),
            #     # ('nowowesi_15', 'kofakibu_00', 'kenewina_92'),
            #     # ('pekukele_64', 'nihihagu_67', 'kenewina_92'),
            #     # ('pekukele_64', 'kofakibu_00', 'kenewina_92'),
            #     ]
            for trip_i in good_triplets:
                # print(trip_i)

                df = group_done_i
                df = df[
                    (df["job_id_max"].isin(list(trip_i))) &
                    [True for i in range(len(df))]
                    ]
                oer_trip_i = df

                # from IPython.display import display
                # display(oer_trip_i)


                # #################################################
                idx = pd.IndexSlice
                row_o_i = oer_trip_i.loc[idx[:, :, "o", :, :], :].iloc[0]
                job_id_o_i = row_o_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_oh_i = oer_trip_i.loc[idx[:, :, "oh", :, :], :].iloc[0]
                job_id_oh_i = row_oh_i.job_id_max

                # #################################################
                idx = pd.IndexSlice
                row_bare_i = oer_trip_i.loc[idx[:, :, "bare", :, :], :].iloc[0]
                job_id_bare_i = row_bare_i.job_id_max

                # #################################################
                # Computing adsorption energy
                df_ads_i = calc_ads_e(oer_trip_i.reset_index())
                df_ads_i = df_ads_i.set_index("ads", drop=False)

                ads_g_o_i = df_ads_i.loc["o"]["ads_e"]
                ads_e_o_i = df_ads_i.loc["o"]["ads_e_elec"]

                ads_g_oh_i = df_ads_i.loc["oh"]["ads_e"]
                ads_e_oh_i = df_ads_i.loc["oh"]["ads_e_elec"]
                job_id_oh_i = df_ads_i.loc["oh"]["job_id_max"]


                # #############################################
                data_dict_i = dict()
                # #############################################
                data_dict_i.update(name_dict_i)
                # #############################################
                data_dict_i["g_o"] = ads_g_o_i
                data_dict_i["g_oh"] = ads_g_oh_i
                data_dict_i["e_o"] = ads_e_o_i
                data_dict_i["e_oh"] = ads_e_oh_i
                data_dict_i["job_id_o"] = job_id_o_i
                data_dict_i["job_id_oh"] = job_id_oh_i 
                data_dict_i["job_id_bare"] = job_id_bare_i
                data_dict_i["all_done"] = all_jobs_in_group_done
                data_dict_i["any_bare_done"] = any_bare_done
                data_dict_i["any_oh_done"] = any_oh_done
                data_dict_i["any_o_done"] = any_o_done
                data_dict_i["any_o_w_as_done"] = any_o_done_with_active_sites
                # #############################################
                data_dict_list.append(data_dict_i)
                # #############################################


        # #########################################################
        df_ads_i = pd.DataFrame(data_dict_list)
        # #########################################################

        name_str_i = [str(i) for i in list(name_i)]
        name_str_i = "__".join(name_str_i)

        df_dict[name_str_i] = df_ads_i



        # #########################################################
        # Pickling data ###########################################
        directory = os.path.join(
            root_dir, "out_data")
        if not os.path.exists(directory): os.makedirs(directory)
        with open(os.path.join(directory, "df_dict.pickle"), "wb") as fle:
            pickle.dump(df_dict, fle)
        # #########################################################
# -

df_ads__from_oh = df_ads__from_oh.set_index(["compenv", "slab_id", "active_site", ])

# #########################################################
import pickle; import os
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/oer_analysis/oer_scaling", 
    "out_data/trace_poly_1.pickle")
with open(path_i, "rb") as fle:
    trace_poly_1 = pickle.load(fle)
# #########################################################

most_deviated_systems = [

    ('sherlock', 'vipikema_98', 47.0),
    ('sherlock', 'wafitemi_24', 29.0),
    ('sherlock', 'kapapohe_58', 29.0),
    ('sherlock', 'sifebelo_94', 63.0),
    ('sherlock', 'momaposi_60', 54.0),
    ('sherlock', 'kamevuse_75', 53.0),
    ('sherlock', 'vegarebo_06', 50.0),
    ('slac', 'dotivela_46', 26.0),
    ('nersc', 'kererape_22', 88.0),
    ('slac', 'damidiwi_47', 29.0),
    ('sherlock', 'vegarebo_06', 48.0),
    ('nersc', 'legofufi_61', 91.0),
    ('sherlock', 'filetumi_93', 67.0),
    ('slac', 'paritile_76', 40.0),
    ('nersc', 'dakoputu_58', 76.0),
    ('slac', 'damidiwi_47', 28.0),
    ('sherlock', 'hahesegu_39', 20.0),
    ('sherlock', 'vipikema_98', 48.0),
    ('sherlock', 'mibumime_94', 61.0),
    ('nersc', 'kererape_22', 94.0),
    ('sherlock', 'sitilowi_31', 38.0),
    ('sherlock', 'mibumime_94', 60.0),
    ('slac', 'fevahaso_90', 27.0),
    ('slac', 'sunuheka_77', 51.0),
    ('slac', 'gulipita_22', 47.0),
    ('sherlock', 'hahesegu_39', 21.0),
    ('sherlock', 'gavibawi_45', 40.0),
    ('slac', 'powodupo_20', 26.0),
    ('nersc', 'dakoputu_58', 74.0),
    ('sherlock', 'bekusuvu_00', 67.0),
    ('slac', 'relovalu_12', 24.0),
    ('sherlock', 'ripirefu_15', 67.0),
    ('slac', 'hefikala_18', 64.0),
    ('nersc', 'winomuvi_99', 83.0),
    ('sherlock', 'filetumi_93', 65.0),
    ('sherlock', 'vevoraso_36', 24.0),
    ('sherlock', 'sihisalu_64', 68.0),
    ('sherlock', 'tagediso_07', 42.0),
    ('sherlock', 'ripirefu_15', 49.0),
    ('slac', 'vomelawi_63', 66.0),

    ]

# +

most_deviated_systems = [
 ('sherlock', 'vipikema_98', 47.0),
 ('sherlock', 'wafitemi_24', 29.0),
 ('sherlock', 'kapapohe_58', 29.0),
 ('sherlock', 'sifebelo_94', 63.0),
 ('sherlock', 'momaposi_60', 54.0),
 ('sherlock', 'kamevuse_75', 53.0),
 ('sherlock', 'vegarebo_06', 50.0),
 ('slac', 'dotivela_46', 26.0),
 ('nersc', 'kererape_22', 88.0),
 ('slac', 'damidiwi_47', 29.0),
 ('sherlock', 'vegarebo_06', 48.0),
 ('nersc', 'legofufi_61', 91.0),
 ('sherlock', 'filetumi_93', 67.0),
 ('slac', 'paritile_76', 40.0),
 ('nersc', 'dakoputu_58', 76.0),
 ('slac', 'damidiwi_47', 28.0),
 ('sherlock', 'hahesegu_39', 20.0),
 ('sherlock', 'vipikema_98', 48.0),
 ('sherlock', 'mibumime_94', 61.0),
 ('nersc', 'kererape_22', 94.0),
 ('sherlock', 'sitilowi_31', 38.0),
 ('sherlock', 'mibumime_94', 60.0),
 ('slac', 'fevahaso_90', 27.0),
 ('slac', 'sunuheka_77', 51.0),
 ('slac', 'gulipita_22', 47.0),
 ('sherlock', 'hahesegu_39', 21.0),
 ('sherlock', 'gavibawi_45', 40.0),
 ('slac', 'powodupo_20', 26.0),
 ('nersc', 'dakoputu_58', 74.0),
 ('sherlock', 'bekusuvu_00', 67.0),
 ('slac', 'relovalu_12', 24.0),
 ('sherlock', 'ripirefu_15', 67.0),
 ('slac', 'hefikala_18', 64.0),
 ('nersc', 'winomuvi_99', 83.0),
 ('sherlock', 'filetumi_93', 65.0),
 ('sherlock', 'vevoraso_36', 24.0),
 ('sherlock', 'sihisalu_64', 68.0),
 ('sherlock', 'tagediso_07', 42.0),
 ('sherlock', 'ripirefu_15', 49.0),
 ('slac', 'vomelawi_63', 66.0),
 ('sherlock', 'novoloko_50', 20.0),
 ('sherlock', 'lufinanu_76', 46.0),
 ('sherlock', 'bekusuvu_00', 69.0),
 ('sherlock', 'vevarehu_32', 63.0),
 ('sherlock', 'newopedu_17', 33.0),
 ('sherlock', 'fugorumi_32', 42.0),
 ('slac', 'dipamife_45', 22.0),
 ('slac', 'diwarise_06', 32.0),
 ('sherlock', 'gihiseru_17', 28.0),
 ('sherlock', 'tanewani_59', 50.0),
 ('sherlock', 'gavibawi_45', 42.0),
 ('sherlock', 'vevarehu_32', 65.0),
 ('slac', 'lagubapi_05', 39.0),
 ('sherlock', 'filetumi_93', 60.0),
 ('nersc', 'letapivu_80', 85.0),
 ('slac', 'gigisanu_24', 32.0),
 ('slac', 'nuriramu_38', 32.0),
 ('nersc', 'giworuge_14', 85.0),
 ('nersc', 'giworuge_14', 81.0),
 ('sherlock', 'mabivuso_96', 50.0),
 ('slac', 'vepufiga_56', 24.0),
 ('sherlock', 'posifuvi_45', 21.0),
 ('slac', 'seravuha_97', 41.0),
 ('nersc', 'legofufi_61', 88.0),
 ('sherlock', 'tanewani_59', 53.0),
 ('sherlock', 'kobehubu_94', 52.0),
 ('slac', 'sesiguva_21', 16.0),
 ('slac', 'vovumota_03', 32.0),
 ('sherlock', 'mabivuso_96', 48.0),
 ('sherlock', 'mokapipu_61', 61.0),
 ('sherlock', 'logusole_78', 41.0),
 ('sherlock', 'tesameli_14', 50.0),
 ('sherlock', 'pegapesa_22', 16.0),
 ('slac', 'wihuwone_95', 26.0),
 ('sherlock', 'dimafowe_05', 20.0),
 ('sherlock', 'lenabefe_62', 49.0),
 ('slac', 'votafefa_68', 35.0),
 ('sherlock', 'lenabefe_62', 48.0),
 ('sherlock', 'sifebelo_94', 65.0),
 ('sherlock', 'pidanule_44', 41.0)]

# +
# ("slac", "dotivela_46", 26., )


df_dict[
    "slac__dotivela_46__26.0"
    ].loc[[4, 12]]
    # ].loc[[0, 8]]
    # ]

# +
shared_scatter_props = go.Scatter(
    mode="markers+lines",
    marker=go.scatter.Marker(
        opacity=0.7,
        ),
    )

traces_to_add_at_end = []

data = []
# iterator = enumerate(random.sample(list(df_dict.keys()), 10))
iterator = enumerate(most_deviated_systems)
for i_cnt, name_str_i in iterator:
    if type(name_str_i) is tuple:
        name_str_i = "__".join(
            [str(i) for i in list(name_str_i)]
            )

    # #####################################################
    name_i = name_str_i.split("__")[0:2] + [float(name_str_i.split("__")[2])]
    name_i = tuple(name_i)
    # #####################################################

    df_ads_i = df_dict[
        name_str_i
        ]

    trace_i = go.Scatter(
        x=df_ads_i.sort_values("g_oh").g_oh,
        y=df_ads_i.sort_values("g_oh").g_o,
        name=name_str_i + "_X",
        legendgroup=name_str_i,
        )
    trace_i.update(
        dict1=shared_scatter_props.to_plotly_json(),
        )
    data.append(trace_i)


    if name_i in df_features_targets.index:
        trace_i = go.Scatter(
            x=[df_features_targets.loc[name_i][("targets", "g_oh", "", )]],
            y=[df_features_targets.loc[name_i][("targets", "g_o", "", )]],
            # y=[df_ads__from_oh.loc[name_i].g_o, ],
            mode="markers",
            marker=dict(size=12, color="black", ),
            name=name_str_i + "_XX",
            legendgroup=name_str_i,
            )
        traces_to_add_at_end.append(trace_i)

    # if name_i in df_ads__from_oh.index:
    #     trace_i = go.Scatter(
    #         x=[df_ads__from_oh.loc[name_i].g_oh, ],
    #         y=[df_ads__from_oh.loc[name_i].g_o, ],
    #         mode="markers",
    #         marker=dict(size=12, color="black", ),
    #         name=name_str_i + "_XX",
    #         )
    #     traces_to_add_at_end.append(trace_i)



traces = data + traces_to_add_at_end + [trace_poly_1]

fig = go.Figure(
    data=traces
    )
fig.show()
# -

# ### Saving figure

# +
my_plotly_plot(
    figure=fig,
    # save_dir=None,
    # place_in_out_plot=True,
    plot_name="scaling_plot__all_oer_triplets",
    write_html=True,
    try_orca_write=True,
    )

fig.write_json(
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/collect_collate_dft_data",
        "out_plot/scaling_plot__all_oer_triplets.json"))

# + active=""
#
#
#
# -

assert False

# +
# from methods import get_df_struct_drift

# df_struct_drift = get_df_struct_drift()

# +
# df_ads_out[df_ads_out.ads == "bare"]

var = "o"
row_o = df_ads_out.query('ads == @var')
job_id_o = row_o.iloc[0].job_id_max

var = "oh"
row_oh = df_ads_out.query('ads == @var')
job_id_oh = row_oh.iloc[0].job_id_max

var = "bare"
row_bare = df_ads_out.query('ads == @var')
job_id_bare = row_bare.iloc[0].job_id_max

job_id_bare
job_id_o
job_id_oh

# +
job_ids = [job_id_bare, job_id_o]

job_ids_str = "__".join(list(np.sort(job_ids)))
# -

df_struct_drift = df_struct_drift.set_index("pair_str")

# +
row_drift = df_struct_drift.loc[job_ids_str]

row_drift.mean_displacement

# + active=""
#
#
#
# -

assert False

# ### Writing data to pickle

# + jupyter={"source_hidden": true}
# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/collect_collate_dft_data",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_ads.pickle"), "wb") as fle:
    pickle.dump(df_ads, fle)
# #########################################################

# + jupyter={"source_hidden": true}
from methods import get_df_ads

df_ads_tmp = get_df_ads()
df_ads_tmp.head()

# + jupyter={"source_hidden": true}
# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("collect_collate_dft.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# trace_i.update(
#     dict1=shared_scatter_props.to_plotly_json(),
#     )

# + jupyter={"source_hidden": true}
# group_i

# + jupyter={"source_hidden": true}
# group_done_i = group_i

# + jupyter={"source_hidden": true}
# oer_trip_i

# + jupyter={"source_hidden": true}
# all_triplets = list(itertools.combinations(group_done_i.job_id_max.tolist(), 3))

# data_dict_list = []
# good_triplets = []
# for trip_i in all_triplets:
#     df_i = pd.concat([
#         group_done_i.index.to_frame(),
#         group_done_i],
#         axis=1)

#     df_i = df_i.set_index("job_id_max")

#     df_trip_i = df_i.loc[list(trip_i)]

#     num_uniq_ads = len(list(df_trip_i.ads.unique()))

#     if num_uniq_ads == 3:
#         good_triplets.append(trip_i)


# data_dict_list = []
# for trip_i in good_triplets:
#     # print(20 * "-")
#     # print(trip_i)


#     job_id_o = None
#     job_id_oh = None
#     job_id_bare = None
#     for job_id_i in trip_i:
#         if df_jobs.loc[job_id_i].ads == "o":
#             job_id_o = job_id_i
#         elif df_jobs.loc[job_id_i].ads == "oh":
#             job_id_oh = job_id_i
#         elif df_jobs.loc[job_id_i].ads == "bare":
#             job_id_bare = job_id_i
#         else:
#             print("This isn't good sidjfisdj89")

#     assert job_id_bare is not None, "TEMP"
#     assert job_id_o is not None, "TEMP"
#     assert job_id_oh is not None, "TEMP"


#     # #####################################
#     pair_oh_bare = np.sort(
#         [job_id_oh, job_id_bare, ]
#         )
#     pair_oh_bare_sort_i = tuple(pair_oh_bare)

#     pair_oh_bare_sort_str_i = "__".join(pair_oh_bare_sort_i)


#     # #####################################
#     pair_o_bare = np.sort(
#         [job_id_o, job_id_bare, ]
#         )
#     pair_o_bare_sort_i = tuple(pair_o_bare)

#     pair_o_bare_sort_str_i = "__".join(pair_o_bare_sort_i)

#     # #####################################
#     df_magmom_drift_i = df_magmom_drift.loc[[
#         pair_oh_bare_sort_str_i,
#         pair_o_bare_sort_str_i,
#         ]]

#     magmom_diff_metric = df_magmom_drift_i["sum_abs_d_magmoms__nonocta_pa"].sum()

#     # print(
#     #     magmom_diff_metric
#     #     )

#     # #####################################################
#     data_dict_i = dict()
#     # #####################################################
#     # data_dict_i["triplet"] = 
#     data_dict_i["job_id_o"] = job_id_o
#     data_dict_i["job_id_oh"] = job_id_oh
#     data_dict_i["job_id_bare"] = job_id_bare
#     data_dict_i["magmom_diff_metric"] = magmom_diff_metric
#     # #####################################################
#     data_dict_list.append(data_dict_i)
#     # #####################################################











# df_trip_magmom_i = pd.DataFrame(data_dict_list)

# row_best_i = df_trip_magmom_i.sort_values("magmom_diff_metric").iloc[0]

# # row_best_i.job_id_o
# # row_best_i.job_id_oh
# # row_best_i.job_id_bare

# group_done_tmp = group_done_i.set_index("job_id_max", drop=False)

# df_ads_out = pd.concat([
#     group_done_i[group_done_i.job_id_max == row_best_i.job_id_o],
#     group_done_i[group_done_i.job_id_max == row_best_i.job_id_oh],
#     group_done_i[group_done_i.job_id_max == row_best_i.job_id_bare],

#     # group_done_tmp.loc[[row_best_i.job_id_o]],
#     # group_done_tmp.loc[[row_best_i.job_id_oh]],
#     # group_done_tmp.loc[[row_best_i.job_id_bare]],

#     ], axis=0)

# + jupyter={"source_hidden": true}
# fifasula_02

# + jupyter={"source_hidden": true}
# detumalu_52

# + jupyter={"source_hidden": true}
# df_features_targets.loc[name_i][("targets", "g_o", "", )]

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# sorted_keys_ij in comparisons.items()

# + jupyter={"source_hidden": true}
# comparisons

# + jupyter={"source_hidden": true}
# comparisons

# + jupyter={"source_hidden": true}
# # sorted_keys_ij = 

# # tuple(np.sort(key_i, key_j))

# np.sort([key_i, key_j])

# key_i

# key_j

# + jupyter={"source_hidden": true}
# # key

# if row_i.job_id_o == row_j.job_id_o:
#     tmp = 42

#     print(key_i, key_j)

# + jupyter={"source_hidden": true}
# if row_magmom__exists:
#     tmp = 42


# if row_from_oh__exists:
#     tmp = 42

# + jupyter={"source_hidden": true}
# row_magmom_i.job_id_o

# + jupyter={"source_hidden": true}
# row_from_oh_i.job_id_o
# -



# + jupyter={"source_hidden": true}
# df_oer_trip_comp

# + jupyter={"source_hidden": true}
# new_col_i
# new_cols

# + jupyter={"source_hidden": true}
# main_0_lev_i

# + jupyter={"source_hidden": true}
# df_tmp =

# df_oer_trip_comp[df_oer_trip_comp.all_all_True == True]

# df_oer_trip_comp.columns.levels[0]
