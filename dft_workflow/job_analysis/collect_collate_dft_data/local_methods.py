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

print("oxy_ref, hyd_ref")
print(oxy_ref, hyd_ref)

print("")
print("corrections_dict:", corrections_dict)






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


# name=name_i
# group=group_done_i
# df_octa_info=df_octa_info
# df_jobs_oh_anal=df_jobs_oh_anal
# heuristic__if_lower_e=False
# # heuristic__if_lower_e=True

def get_oer_triplet(
    name=None,
    group=None,
    df_octa_info=None,
    df_jobs_oh_anal=None,
    heuristic__if_lower_e=False,
    ):
    """
    """
    #| - get_oer_triplet
    name_i = name
    group_i = group

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################


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


    job_with_suff_lower_energy = False


    # If there is a *O ran from *OH, then just pick the min energy sys
    # o_row_from_oh = False in df_ads_o.as_is_nan.tolist()
    o_row_from_oh = True in df_ads_o.rerun_from_oh.tolist()
    if o_row_from_oh:















        new_index = []
        for index_i, row_i in df_ads_o.iterrows():
            i = index_i

            rerun_from_oh_i = row_i.rerun_from_oh
            if rerun_from_oh_i is False or np.isnan(rerun_from_oh_i):
                from_oh_i = False
            else:
                from_oh_i = True

            i_tmp_2 = (
                i[0], i[1], i[2],
                active_site_i,
                i[4], from_oh_i,
                )
            new_index.append(i_tmp_2)

        df_octa_info_i = df_octa_info.loc[new_index]

        df_octa_info_i_2 = df_octa_info_i[df_octa_info_i.error == False]

        if df_octa_info_i_2.shape[0] > 1:
            tmp = 42
            # print("TEMP TEMP TEMP TEMP")





        new_indices_o = []
        for name_i, row_i in df_octa_info_i_2.iterrows():
            compenv = name_i[0]
            slab_id = name_i[1]
            ads = name_i[2]
            active_site = name_i[3]
            att_num = name_i[4]
            from_oh = name_i[5]

            if from_oh is False:
                active_site = "NaN"

            new_index = (compenv, slab_id, ads, active_site, att_num)
            new_indices_o.append(new_index)

        if len(new_indices_o) > 0:
            df_ads_o_2 = df_ads_o.loc[new_indices_o]
            df_ads_o = df_ads_o_2











        row_o__from_oh = df_ads_o[df_ads_o.rerun_from_oh == True]
        # row_o__from_oh = row_o__from_oh.sort_values("pot_e").iloc[[0]]




        if row_o__from_oh.shape[0] > 0:
            row_o__from_oh = row_o__from_oh.sort_values("pot_e").iloc[[0]]
            row_o = row_o__from_oh
        elif df_ads_o.shape[0] > 0:
            row_o__tmp = df_ads_o.sort_values("pot_e").iloc[[0]]
            row_o = row_o__tmp





        # # row_o__from_oh = df_ads_o[df_ads_o.as_is_nan == False]
        # row_o__from_oh = df_ads_o[df_ads_o.rerun_from_oh == True]
        # row_o__from_oh = row_o__from_oh.sort_values("pot_e").iloc[[0]]

        # row_o = row_o__from_oh

        if heuristic__if_lower_e:
            # | - heuristic__if_lower_e

            # #####################################################
            # If the from_oh *O slab is not within 0.1 eV of being the most stable, then pick the most stable instead
            df_ads_o__wo_from_oh = df_ads_o.drop(index=row_o__from_oh.index)
            df_ads_o__wo_from_oh["pot_e_diff"] = df_ads_o__wo_from_oh["pot_e"] - row_o__from_oh.pot_e.iloc[0]
            df_i = df_ads_o__wo_from_oh[df_ads_o__wo_from_oh["pot_e_diff"] < -0.05]

            # job_with_suff_lower_energy = False
            if df_i.shape[0] > 0:
                job_with_suff_lower_energy = True

            if job_with_suff_lower_energy:
                row_o = df_i.sort_values("pot_e").iloc[[0]]

                # print("Taking more stable *O instead of from_oh *O")
                # print("pot_e_diff:", row_o.iloc[0]["pot_e_diff"])

            # row_o["low_e_not_from_oh"] = job_with_suff_lower_energy
            # __|

    # Else just pick the lowest energy O
    else:
        row_o = df_ads_o.sort_values("pot_e").iloc[[0]]


    row_o["low_e_not_from_oh"] = job_with_suff_lower_energy


    rows_to_concat.append(row_o)
    #__|

    # #####################################################
    #| - Selecting *OH slab
    if oh_avail:
        idx = pd.IndexSlice
        df_ads_oh = group_i.loc[idx[:, :, "oh", :], :]

        new_index = []
        for i in df_ads_oh.index.tolist():
            i_tmp_1 = list(i)
            i_tmp_1.append(True)
            i_tmp_2 = tuple(i_tmp_1)
            new_index.append(i_tmp_2)

        df_octa_info_i = df_octa_info.loc[new_index]

        df_octa_info_i_2 = df_octa_info_i[df_octa_info_i.error == False]

        new_index_oh = []
        for i in df_octa_info_i_2.index.tolist():
            new_index_oh.append(tuple(list(i)[0:-1]))



        if len(new_index_oh) > 0:
            df_ads_oh_new = df_ads_oh.loc[
                new_index_oh
                ]

            row_oh = df_ads_oh_new.sort_values("pot_e").iloc[[0]]
            rows_to_concat.append(row_oh)


        # if name_i in df_jobs_oh_anal.index:
        #     #| - If *OH analysis has been done
        #     # #############################################
        #     row_oh_i = df_jobs_oh_anal.loc[name_i]
        #     # #############################################
        #     job_id_most_stable_i = row_oh_i.job_id_most_stable
        #     job_ids_sorted_energy_i = row_oh_i.job_ids_sorted_energy
        #     # #############################################

        #     if len(job_ids_sorted_energy_i) != 0:
        #         row_oh = df_ads_oh[df_ads_oh.job_id_max == job_id_most_stable_i]
        #         row_oh = row_oh.iloc[[0]]

        #         rows_to_concat.append(row_oh)
        #     #__|
        # else:
        #     row_oh = df_ads_oh.sort_values("pot_e").iloc[[0]]
        #     rows_to_concat.append(row_oh)

    #__|

    # #####################################################
    #| - Selecting *bare slab

    job_with_suff_lower_energy = False

    if bare_avail:
        idx = pd.IndexSlice
        df_ads_bare = group_i.loc[idx[:, :, "bare", :], :]

        # If there is a *O ran from *OH, then just pick the min energy sys
        bare_row_from_oh = True in df_ads_bare.rerun_from_oh.tolist()
        if bare_row_from_oh:
            # row_bare = df_ads_bare[df_ads_bare.rerun_from_oh == True]
            # # row_bare = row_bare.sort_values("pot_e").iloc[0]
            # row_bare = row_bare.sort_values("pot_e").iloc[[0]]




            # row_bare__from_oh = df_ads_bare[df_ads_bare.as_is_nan == False]
            row_bare__from_oh = df_ads_bare[df_ads_bare.rerun_from_oh == True]

            row_bare__from_oh = row_bare__from_oh.sort_values("pot_e").iloc[[0]]

            row_bare = row_bare__from_oh

            if heuristic__if_lower_e:
                # | - heuristic__if_lower_e
                # #####################################################
                # If the from_oh *bare slab is not within 0.1 eV of being the most stable, then pick the most stable instead
                df_ads_bare__wo_from_oh = df_ads_bare.drop(index=row_bare__from_oh.index)
                df_ads_bare__wo_from_oh["pot_e_diff"] = df_ads_bare__wo_from_oh["pot_e"] - row_bare__from_oh.pot_e.iloc[0]


                # from IPython.display import display
                # print(40 * "-")
                # display(df_ads_bare__wo_from_oh)

                df_i = df_ads_bare__wo_from_oh[df_ads_bare__wo_from_oh["pot_e_diff"] < -0.05]


                # job_with_suff_lower_energy = False
                if df_i.shape[0] > 0:
                    job_with_suff_lower_energy = True

                if job_with_suff_lower_energy:
                    row_bare = df_i.sort_values("pot_e").iloc[[0]]

                    # print("Taking more stable * instead of from_oh *")
                    # print("pot_e_diff:", row_bare.iloc[0]["pot_e_diff"])

                # row_bare["low_e_not_from_oh"] = job_with_suff_lower_energy
                # __|

        else:
            # row_bare = df_ads_bare.sort_values("pot_e").iloc[0]
            row_bare = df_ads_bare.sort_values("pot_e").iloc[[0]]

        row_bare["low_e_not_from_oh"] = job_with_suff_lower_energy


        rows_to_concat.append(row_bare)
    #__|

    #__|


    df_ads_out = pd.concat(rows_to_concat, axis=0)

    return(df_ads_out)
    #__|



# | - EXPERIMENTING WITH MORE OER TRIPLET SELECTION RULES

# name=name_i
# group=group_done_i

def get_oer_triplet__low_e(
    name=None,
    group=None,
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

    df_ads_o_i = df_ads_o.sort_values("pot_e", ascending=True)
    row_o_low_e = df_ads_o_i.iloc[[0]]

    row_o = row_o_low_e
    rows_to_concat.append(row_o)
    #__|



    # #####################################################
    #| - Selecting *OH slab
    if oh_avail:
        idx = pd.IndexSlice
        df_ads_oh = group_i.loc[idx[:, :, "oh", :], :]

        df_ads_oh_i = df_ads_oh.sort_values("pot_e", ascending=True)
        row_oh_low_e = df_ads_oh_i.iloc[[0]]

        row_oh = row_oh_low_e
        rows_to_concat.append(row_oh)
    #__|

    # #####################################################
    #| - Selecting *bare slab
    if bare_avail:
        idx = pd.IndexSlice
        df_ads_bare = group_i.loc[idx[:, :, "bare", :], :]

        df_ads_bare_i = df_ads_bare.sort_values("pot_e", ascending=True)
        row_bare_low_e = df_ads_bare_i.iloc[[0]]

        row_bare = row_bare_low_e
        rows_to_concat.append(row_bare)
    #__|

    #__|


    df_ads_out = pd.concat(rows_to_concat, axis=0)

    return(df_ads_out)
    #__|


# name=name_i
# group=group_done_i
# df_jobs_oh_anal=df_jobs_oh_anal

def get_oer_triplet__from_oh(
    name=None,
    group=None,
    df_jobs_oh_anal=None,
    ):
    """
    """
    #| - get_oer_triplet
    name_i = name
    group_i = group

    error = False
    note = ""

    group_ind_i = group_i.index.to_frame()

    #| - Selecting adsorbate rows
    rows_to_concat = []

    # #####################################################
    #| - Selecting *O slab
    idx = pd.IndexSlice
    df_ads_o = group_i.loc[idx[:, :, "o", :], :]


    o_row_from_oh = True in df_ads_o.rerun_from_oh.tolist()
    if o_row_from_oh:
        row_o__from_oh = df_ads_o[df_ads_o.rerun_from_oh == True]
        row_o__from_oh = row_o__from_oh.sort_values("pot_e").iloc[[0]]
        row_o = row_o__from_oh
        rows_to_concat.append(row_o)
    else:
        error = True
        note += "*O from *OH not avail | "
    #__|

    # #####################################################
    #| - Selecting *OH slab
    # if oh_avail:
    idx = pd.IndexSlice
    df_ads_oh = group_i.loc[idx[:, :, "oh", :], :]

    if name_i in df_jobs_oh_anal.index:
        #| - If *OH analysis has been done
        # #################################################
        row_oh_i = df_jobs_oh_anal.loc[name_i]
        # #################################################
        job_id_most_stable_i = row_oh_i.job_id_most_stable
        job_ids_sorted_energy_i = row_oh_i.job_ids_sorted_energy
        # #################################################

        if len(job_ids_sorted_energy_i) != 0:
            row_oh = df_ads_oh[df_ads_oh.job_id_max == job_id_most_stable_i]
            row_oh = row_oh.iloc[[0]]

            rows_to_concat.append(row_oh)
        else:
            # print("Youch look into it 2")
            row_oh = df_ads_oh.sort_values("pot_e").iloc[[0]]
            rows_to_concat.append(row_oh)
        #__|
    else:
        print("Youch look into it")

        row_oh = df_ads_oh.sort_values("pot_e").iloc[[0]]
        rows_to_concat.append(row_oh)
    #__|

    # #####################################################
    #| - Selecting *bare slab
    idx = pd.IndexSlice
    df_ads_bare = group_i.loc[idx[:, :, "bare", :], :]

    bare_row_from_oh = True in df_ads_bare.rerun_from_oh.tolist()
    if bare_row_from_oh:
        row_bare__from_oh = df_ads_bare[df_ads_bare.rerun_from_oh == True]
        row_bare__from_oh = row_bare__from_oh.sort_values("pot_e").iloc[[0]]
        row_bare = row_bare__from_oh

        rows_to_concat.append(row_bare)

    else:
        error = True
        note += "* from *OH not avail | "
    #__|

    #__|



    df_ads_out = pd.concat(rows_to_concat, axis=0)

    out_dict = dict()
    out_dict["df_oer_triplet"] = df_ads_out
    out_dict["error"] = error
    out_dict["note"] = note

    return(out_dict)
    #__|


import itertools
import numpy as np

# name=name_i
# group=group_done_i
# df_jobs_oh_anal=df_jobs_oh_anal

def get_oer_triplet__magmom(
    name=None,
    group=None,
    df_jobs=None,
    df_jobs_oh_anal=None,
    df_magmom_drift=None,
    ):
    """
    """
    #| - get_oer_triplet
    name_i = name
    group_i = group

    error = False
    note = ""

    group_ind_i = group_i.index.to_frame()
















    # all_triplets = list(itertools.combinations(group_done_i.job_id_max.tolist(), 3))
    all_triplets = list(itertools.combinations(group.job_id_max.tolist(), 3))

    data_dict_list = []
    good_triplets = []
    for trip_i in all_triplets:
        df_i = pd.concat([
            group.index.to_frame(),
            group],
            # group_done_i.index.to_frame(),
            # group_done_i],
            axis=1)

        df_i = df_i.set_index("job_id_max")

        df_trip_i = df_i.loc[list(trip_i)]

        num_uniq_ads = len(list(df_trip_i.ads.unique()))

        if num_uniq_ads == 3:
            good_triplets.append(trip_i)


    data_dict_list = []
    for trip_i in good_triplets:
        # print(20 * "-")
        # print(trip_i)


        job_id_o = None
        job_id_oh = None
        job_id_bare = None
        for job_id_i in trip_i:
            if df_jobs.loc[job_id_i].ads == "o":
                job_id_o = job_id_i
            elif df_jobs.loc[job_id_i].ads == "oh":
                job_id_oh = job_id_i
            elif df_jobs.loc[job_id_i].ads == "bare":
                job_id_bare = job_id_i
            else:
                print("This isn't good sidjfisdj89")

        assert job_id_bare is not None, "TEMP"
        assert job_id_o is not None, "TEMP"
        assert job_id_oh is not None, "TEMP"


        # #####################################
        pair_oh_bare = np.sort(
            [job_id_oh, job_id_bare, ]
            )
        pair_oh_bare_sort_i = tuple(pair_oh_bare)

        pair_oh_bare_sort_str_i = "__".join(pair_oh_bare_sort_i)


        # #####################################
        pair_o_bare = np.sort(
            [job_id_o, job_id_bare, ]
            )
        pair_o_bare_sort_i = tuple(pair_o_bare)

        pair_o_bare_sort_str_i = "__".join(pair_o_bare_sort_i)

        # #####################################
        df_magmom_drift_i = df_magmom_drift.loc[[
            pair_oh_bare_sort_str_i,
            pair_o_bare_sort_str_i,
            ]]

        magmom_diff_metric = df_magmom_drift_i["sum_abs_d_magmoms__nonocta_pa"].sum()

        # print(
        #     magmom_diff_metric
        #     )

        # #####################################################
        data_dict_i = dict()
        # #####################################################
        # data_dict_i["triplet"] =
        data_dict_i["job_id_o"] = job_id_o
        data_dict_i["job_id_oh"] = job_id_oh
        data_dict_i["job_id_bare"] = job_id_bare
        data_dict_i["magmom_diff_metric"] = magmom_diff_metric
        # #####################################################
        data_dict_list.append(data_dict_i)
        # #####################################################











    df_trip_magmom_i = pd.DataFrame(data_dict_list)

    row_best_i = df_trip_magmom_i.sort_values("magmom_diff_metric").iloc[0]

    # row_best_i.job_id_o
    # row_best_i.job_id_oh
    # row_best_i.job_id_bare

    group_done_tmp = group.set_index("job_id_max", drop=False)
    # group_done_tmp = group_done_i.set_index("job_id_max", drop=False)

    df_ads_out = pd.concat([
        group[group.job_id_max == row_best_i.job_id_o],
        group[group.job_id_max == row_best_i.job_id_oh],
        group[group.job_id_max == row_best_i.job_id_bare],

        # group_done_tmp.loc[[row_best_i.job_id_o]],
        # group_done_tmp.loc[[row_best_i.job_id_oh]],
        # group_done_tmp.loc[[row_best_i.job_id_bare]],

        ], axis=0)















    # df_ads_out = pd.concat(rows_to_concat, axis=0)

    out_dict = dict()
    out_dict["df_oer_triplet"] = df_ads_out
    out_dict["error"] = error
    out_dict["note"] = note

    return(out_dict)
    #__|

# __|
