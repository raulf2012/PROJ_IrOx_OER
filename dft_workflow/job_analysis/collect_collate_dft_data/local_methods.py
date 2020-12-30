"""
"""

#| - Import Modules
import os
import sys

sys.path.insert(0, os.path.join(
    os.environ["PROJ_irox"],
    "data"))

import pandas as pd

from oxr_reaction.oxr_methods import df_calc_adsorption_e

from energetics.dft_energy import Element_Refs

from proj_data_irox import (
    h2_ref,
    h2o_ref,
    )

from proj_data_irox import (
    corrections_dict,
    )

#__|


h2o_ref = h2o_ref
h2_ref = h2_ref

# h2o_ref = h2o_ref + h2o_corr
# h2_ref = h2_ref + h2_corr

Elem_Refs = Element_Refs(
    H2O_dict={
        "gibbs_e": h2o_ref,
        "electronic_e": h2o_ref,
        },

    H2_dict={
        "gibbs_e": h2_ref,
        "electronic_e": h2_ref,
        },
    )

oxy_ref, hyd_ref = Elem_Refs.calc_ref_energies()

oxy_ref = oxy_ref.gibbs_e
hyd_ref = hyd_ref.gibbs_e


def calc_ads_e(group):
    """Calculate species adsorption energy.

    Args:
        group
    """
    # | - calc_ads_e
    df_calc_adsorption_e(
        group,
        oxy_ref,
        hyd_ref,
        group[
            group["ads"] == "bare"
            ]["pot_e"].iloc[0],
        corrections_mode="corr_dict",
        corrections_dict=corrections_dict,
        adsorbate_key_var="ads",
        dft_energy_key_var="pot_e",
        )

    return(group)
    #__|


def get_group_w_all_ads(
    name=None,
    group=None,
    df_jobs_anal_i=None,
    ):
    """
    """
    #| - get_group_w_all_ads
    # #####################################################
    compenv_i = name[0]
    slab_id_i = name[1]
    active_site_i = name[2]
    # #####################################################

    idx = pd.IndexSlice
    df_o_slabs_NaN = df_jobs_anal_i.loc[
        idx[compenv_i, slab_id_i, "o", "NaN", :], :]
    df_o_slabs_as = df_jobs_anal_i.loc[
        idx[compenv_i, slab_id_i, "o", active_site_i, :], :]

    any_o_done_with_active_sites = False
    if df_o_slabs_as.shape[0] > 0:
        any_o_done_with_active_sites = df_o_slabs_as.job_completely_done.any()


    df_o_slabs = pd.concat([
        df_o_slabs_NaN,
        df_o_slabs_as,
        ])


    group_i = pd.concat([
        df_o_slabs,
        group
        ])

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["group_i"] = group_i
    out_dict["any_o_done_with_active_sites"] = any_o_done_with_active_sites
    # #####################################################
    return(out_dict)
    # #####################################################
    # return(group_i)
    #__|

def are_any_ads_done(
    group=None,
    ):
    """
    """
    #| - are_any_ads_done
    group_i = group

    # Figuring out if there are any completed  jobs for oh, bare, and o
    df_ind = group_i.index.to_frame()

    df_tmp_oh = df_ind[df_ind.ads == "oh"]
    if df_tmp_oh.shape[0] > 0:
        idx = pd.IndexSlice
        group_oh_i = group_i.loc[idx[:, :, "oh", ], :]
        any_oh_done = group_oh_i.job_completely_done.any()
    else:
        any_oh_done = False

    df_tmp_bare = df_ind[df_ind.ads == "bare"]
    if df_tmp_bare.shape[0] > 0:
        idx = pd.IndexSlice
        group_bare_i = group_i.loc[idx[:, :, "bare", ], :]
        any_bare_done = group_bare_i.job_completely_done.any()
    else:
        any_bare_done = False


    df_tmp_o = df_ind[df_ind.ads == "o"]
    if df_tmp_o.shape[0] > 0:
        idx = pd.IndexSlice
        group_o_i = group_i.loc[idx[:, :, "o", ], :]
        any_o_done = group_o_i.job_completely_done.any()
    else:
        any_o_done = False

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["any_o_done"] = any_o_done
    out_dict["any_oh_done"] = any_oh_done
    out_dict["any_bare_done"] = any_bare_done
    # #####################################################
    return(out_dict)
    # #####################################################
    #__|

# name = name_i
# group = group_i
# df_jobs_oh_anal = df_jobs_oh_anal

def get_oer_triplet(
    name=None,
    group=None,
    df_jobs_oh_anal=None,
    ):
    """
    """
    #| - get_oer_triplet
    name_i = name
    group_i = group


    group_ind_i = group_i.index.to_frame()

    o_avail = "o" in group_ind_i.ads.tolist()
    oh_avail = "oh" in group_ind_i.ads.tolist()
    bare_avail = "bare" in group_ind_i.ads.tolist()


    #| - Selecting adsorbate rows
    rows_to_concat = []

    # #####################################################
    #| - Selecting *O slab
    idx = pd.IndexSlice
    df_ads_o = group_i.loc[idx[:, :, "o", :], :]

    # If there is a *O ran from *OH, then just pick the min energy sys
    # o_row_from_oh = False in df_ads_o.as_is_nan.tolist()
    o_row_from_oh = True in df_ads_o.rerun_from_oh.tolist()
    if o_row_from_oh:
        row_o = df_ads_o[df_ads_o.as_is_nan == False]
        row_o = row_o.sort_values("pot_e").iloc[[0]]

    # Else just pick the lowest energy O
    else:
        row_o = df_ads_o.sort_values("pot_e").iloc[[0]]

    rows_to_concat.append(row_o)
    #__|

    # #####################################################
    #| - Selecting *OH slab
    if oh_avail:
        idx = pd.IndexSlice
        df_ads_oh = group_i.loc[idx[:, :, "oh", :], :]

        if name_i in df_jobs_oh_anal.index:
            #| - If *OH analysis has been done
            # #############################################
            row_oh_i = df_jobs_oh_anal.loc[name_i]
            # #############################################
            job_id_most_stable_i = row_oh_i.job_id_most_stable
            job_ids_sorted_energy_i = row_oh_i.job_ids_sorted_energy
            # #############################################

            if len(job_ids_sorted_energy_i) != 0:
                row_oh = df_ads_oh[df_ads_oh.job_id_max == job_id_most_stable_i]
                row_oh = row_oh.iloc[[0]]

                rows_to_concat.append(row_oh)
            #__|
        else:
            row_oh = df_ads_oh.sort_values("pot_e").iloc[[0]]
            rows_to_concat.append(row_oh)
    #__|

    # #####################################################
    #| - Selecting *bare slab
    if bare_avail:
        idx = pd.IndexSlice
        df_ads_bare = group_i.loc[idx[:, :, "bare", :], :]

        # If there is a *O ran from *OH, then just pick the min energy sys
        bare_row_from_oh = True in df_ads_bare.rerun_from_oh.tolist()
        if bare_row_from_oh:
            row_bare = df_ads_bare[df_ads_bare.rerun_from_oh == True]
            # row_bare = row_bare.sort_values("pot_e").iloc[0]
            row_bare = row_bare.sort_values("pot_e").iloc[[0]]
        else:
            # row_bare = df_ads_bare.sort_values("pot_e").iloc[0]
            row_bare = df_ads_bare.sort_values("pot_e").iloc[[0]]

        rows_to_concat.append(row_bare)
    #__|

    #__|



    # df_ads_out = pd.concat(rows_to_concat, axis=1).T
    # df_ads_out = pd.concat(rows_to_concat, axis=1)
    df_ads_out = pd.concat(rows_to_concat, axis=0)

    # df_ads_out = pd.concat([
    #     row_bare,
    #     row_o,
    #     row_oh,
    #     ], axis=1).T

    return(df_ads_out)
    #__|



#| - __old__
# def get_oer_triplet():
#     """
#     """
#     #| - get_oer_triplet
#     # #####################################################
#
#     idx = pd.IndexSlice
#     df_ads_o = group_i.loc[idx[:, :, "o", :], :]
#     idx = pd.IndexSlice
#     df_ads_oh = group_i.loc[idx[:, :, "oh", :], :]
#     idx = pd.IndexSlice
#     df_ads_bare = group_i.loc[idx[:, :, "bare", :], :]
#
#     if (df_ads_o.shape[0] > 1) or (df_ads_oh.shape[0] > 1) or (df_ads_bare.shape[0] > 1):
#         if verbose_local:
#             print("There is more than 1 row per state here, need a better way to select")
#     # #####################################################
#
#     # TEMP
#     row_bare_i = df_ads_bare.iloc[0]
#     pot_e_i = row_bare_i.pot_e
#     job_id_bare_max_i = row_bare_i.job_id_max
#
#
#
#     df_ads_o_i_tmp = df_ads_o[
#         ~df_ads_o.pot_e.isna()
#         ]
#
#     if False in df_ads_o_i_tmp.as_is_nan.tolist():
#         df_ads_o_i_tmp = df_ads_o_i_tmp[df_ads_o_i_tmp.as_is_nan == False]
#
#         mess_i = "Just putting this here to check, should only every be 1 row (or none) here"
#         assert df_ads_o_i_tmp.shape[0] == 1, mess_i
#
#         df_ads_o_i = df_ads_o_i_tmp
#
#     else:
#         # print("Put code here")
#         df_ads_o_i = df_ads_o[df_ads_o.pot_e == df_ads_o.pot_e.min()]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#     df_ads_oh_i = df_ads_oh[df_ads_oh.pot_e == df_ads_oh.pot_e.min()]
#     df_ads_bare_i = df_ads_bare[df_ads_bare.pot_e == df_ads_bare.pot_e.min()]
#
#     if df_ads_bare.shape[0] == 0:
#         #| - If there isn't a bare * calculation then skip for now
#
#         if verbose_local:
#             print("No bare slab available")
#         continue
#         #__|
#     elif df_ads_bare.shape[0] == 1 and np.isnan(pot_e_i):
#         #| - __temp__
#         if df_ads_o_i.shape[0] > 0:
#             job_ids_o_max_i = df_ads_o_i.job_id_max.tolist()
#             if len(job_ids_o_max_i) == 1:
#                 job_id_o_max_i = job_ids_o_max_i[0]
#             else:
#                 job_id_o_max_i = job_ids_o_max_i
#
#         if df_ads_oh_i.shape[0] > 0:
#             job_id_oh_max_i = df_ads_oh_i.job_id_max.iloc[0]
#         else:
#             job_id_oh_max_i = None
#
#
#         ads_e_o_i = None
#         ads_e_oh_i = None
#         job_id_o_i = job_id_o_max_i
#         job_id_oh_i  = job_id_oh_max_i
#         job_id_bare_i = job_id_bare_max_i
#         #__|
#     else:
#         #| - __temp__
#         job_id_bare_i = df_ads_bare_i.iloc[0].job_id_max
#
#         df_ads_i = pd.concat([
#             df_ads_o_i,
#             df_ads_oh_i,
#             df_ads_bare_i])
#         df_ads_i = df_ads_i.reset_index(drop=False)
#
#         # Calculating the
#         df_ads_i = calc_ads_e(df_ads_i)
#
#
#
#         # #########################################################
#         row_oh_i = df_ads_i[df_ads_i.ads == "oh"]
#         if row_oh_i.shape[0] == 1:
#             row_oh_i = row_oh_i.iloc[0]
#             # #####################################################
#             ads_e_oh_i = row_oh_i.ads_e
#             job_id_oh_i = row_oh_i.job_id_max
#             # #####################################################
#         else:
#             ads_e_oh_i = None
#             job_id_oh_i = None
#
#         # #########################################################
#         row_o_i = df_ads_i[df_ads_i.ads == "o"]
#         if row_o_i.shape[0] == 1:
#             row_o_i = row_o_i.iloc[0]
#             # #####################################################
#             ads_e_o_i = row_o_i.ads_e
#             job_id_o_i = row_o_i.job_id_max
#             # #####################################################
#         else:
#             ads_e_o_i = None
#             job_id_o_i = None
#         #__|
#
#     #__|
#__|
