# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Compute formal oxidation state from Kirsten's CatKit code
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy
import shutil

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    create_name_str_from_tup,
    get_df_atoms_sorted_ind,
    get_df_jobs_paths,
    get_df_features,
    get_df_coord_wrap,
    get_df_features_targets,
    get_df_slabs_to_run,
    )

# from methods_features import original_slab_is_good

# #########################################################
from local_methods import set_formal_oxidation_state, get_connectivity
from local_methods import get_catkit_form_oxid_state_wrap
from local_methods import get_effective_ox_state__test
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

verbose = False

# # Read Data

# +
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_active_sites = get_df_active_sites()

df_slabs_to_run = get_df_slabs_to_run()
df_slabs_to_run = df_slabs_to_run.set_index(["compenv", "slab_id", "att_num"])

df_features_targets = get_df_features_targets()

df_features = get_df_features()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_jobs_paths = get_df_jobs_paths()

# + active=""
#
#
#

# +
df_jobs_anal_done = df_jobs_anal[df_jobs_anal.job_completely_done == True]

df_jobs_anal_i =  df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True]

# Selecting *O and *OH systems to process
df_index = df_jobs_anal_i.index.to_frame()
df_index_i = df_index[
    df_index.ads.isin(["o", "oh", ])
    ]
df_jobs_anal_i = df_jobs_anal_i.loc[
    df_index_i.index
    ]
# -

# # Further filtering df_jobs_anal

# +
# #########################################################
indices_to_run = []
# #########################################################
for name_i, row_i in df_jobs_anal_i.iterrows():
    # #####################################################
    run_row = True
    if name_i in df_atoms_sorted_ind.index:
        row_atoms_i = df_atoms_sorted_ind.loc[name_i]
        # #####################################################
        failed_to_sort_i = row_atoms_i.failed_to_sort
        # #####################################################

        if failed_to_sort_i:
            run_row = False
    else:
        run_row = False

    if run_row:
        indices_to_run.append(name_i)


# #########################################################
df_jobs_anal_i = df_jobs_anal_i.loc[
    indices_to_run    
    ]

# +
df_ind = df_jobs_anal_i.index.to_frame()

df = df_ind
df = df[
    # (df["compenv"] == compenv_i) &
    # (df["slab_id"] == slab_id_i) &
    (df["ads"] == "o") &
    # (df["active_site"] == active_site_i) &
    # (df["att_num"] == att_num_i) &
    # (df[""] == ads_i) &
    [True for i in range(len(df))]
    ]

df_jobs_anal_i = df_jobs_anal_i.loc[
    df.index
    ]
df_jobs_anal_i

# +
# assert False

# +
slab_ids = [

"tofebave_45",
"titawupu_08",
"rudosavu_57",
"filetumi_93",
"ralutiwa_59",
"lilotuta_67",
"bikoradi_95",
"kakalito_08",
"wefakuko_75",
"filetumi_93",
"rudosavu_57",
"filetumi_93",
"titawupu_08",
"wefakuko_75",
"vinamepa_43",
"filetumi_93",
"wesaburu_95",
"rudosavu_57",
"dukavula_34",
"bikoradi_95",
"lilotuta_67",
"lilotuta_67",
"bikoradi_95",
"vinamepa_43",
"ramufalu_44",
"wefakuko_75",
"putarude_21",
"dukavula_34",
"vinamepa_43",
"putarude_21",
"wefakuko_75",
"vinamepa_43",
"fogopemi_28",
"vinamepa_43",
"tofebave_45",
"kakalito_08",
"lilotuta_67",
]

# slab_ids = [
#     # "titawupu_08",
#     "ralutiwa_59",
#     ]

df_ind = df_jobs_anal_i.index.to_frame()
df_jobs_anal_i = df_jobs_anal_i.loc[
    df_ind[df_ind.slab_id.isin(slab_ids)].index
    ]

# +
# ('slac', 'ralutiwa_59', 'o', 31.0, 1, False)

# +
# #########################################################
data_dict_list = []
# #########################################################
iterator = tqdm(df_jobs_anal_i.index, desc="1st loop")
for i_cnt, name_i in enumerate(iterator):
    # print(name_i)
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    row_i = df_jobs_anal_i.loc[name_i]
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################
    job_id_max_i = row_i.job_id_max
    # #####################################################

    if verbose:
        name_concat_i = "_".join([str(i) for i in list(name_i)])
        print(40 * "=")
        print(name_concat_i)
        print(name_i)


    # #####################################################
    name_dict_i = dict(zip(
        list(df_jobs_anal_i.index.names), list(name_i)))

    # #####################################################
    row_atoms_i = df_atoms_sorted_ind.loc[name_i]
    # #####################################################
    atoms_sorted_good_i = row_atoms_i.atoms_sorted_good
    # #####################################################
    atoms = atoms_sorted_good_i

    # #####################################################
    row_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_sites_i.active_sites_unique
    # #####################################################


    data_dict_i["job_id_max"] = job_id_max_i


    if active_site_i != "NaN":
        # read_orig_O_df_coord_i = False

        active_site_j = active_site_i


        # oxid_state_i = get_catkit_form_oxid_state_wrap()
        data_out_dict_i = get_catkit_form_oxid_state_wrap(
            atoms=atoms,
            name=name_i,
            active_site=active_site_j,
            )
        oxid_state_i = data_out_dict_i["form_oxid"]
        atoms_out_i = data_out_dict_i["atoms_out"]
        neigh_dict_i = data_out_dict_i["neigh_dict"]
        
        # atoms_out_i.write("tmp.traj")

        # #################################################
        data_dict_j = dict()
        # #################################################
        data_dict_j["from_oh"] = True
        data_dict_j["form_oxid_state__catkit"] = oxid_state_i
        data_dict_j["atoms_catkit"] = atoms_out_i
        data_dict_j["neigh_dict"] = neigh_dict_i
        # #################################################
        data_dict_j.update(name_dict_i)
        # data_dict_j.update(out_dict)
        data_dict_j.update(data_dict_i)
        # data_dict_j.update(data_out_dict_i)
        # #################################################
        data_dict_list.append(data_dict_j)
        # #################################################

    
    else:
        for active_site_j in active_sites_unique_i:

            if verbose:
                print("active_site_j:", active_site_j)

            # oxid_state_i = get_catkit_form_oxid_state_wrap(
            data_out_dict_i = get_catkit_form_oxid_state_wrap(
                atoms=atoms,
                name=name_i,
                active_site=active_site_j,
                )
            oxid_state_i = data_out_dict_i["form_oxid"]
            atoms_out_i = data_out_dict_i["atoms_out"]
            neigh_dict_i = data_out_dict_i["neigh_dict"]
            # atoms_out_i.write("tmp.traj")

            # #############################################
            data_dict_j = dict()
            # #############################################
            data_dict_j["from_oh"] = False
            data_dict_j["form_oxid_state__catkit"] = oxid_state_i
            data_dict_j["active_site"] = active_site_j
            data_dict_j["atoms_catkit"] = atoms_out_i
            data_dict_j["neigh_dict"] = neigh_dict_i
            # #############################################
            name_dict_i_cpy = copy.deepcopy(name_dict_i)
            name_dict_i_cpy.pop("active_site")
            data_dict_j.update(name_dict_i_cpy)
            # data_dict_j.update(out_dict)
            data_dict_j.update(data_dict_i)
            # data_dict_j.update(data_out_dict_i)
            # #############################################
            data_dict_list.append(data_dict_j)
            # #############################################


# #########################################################
df_eff_ox = pd.DataFrame(data_dict_list)
df_eff_ox = df_eff_ox.set_index(["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh", ])
# #########################################################

# +
shared_indices = df_features.index.intersection(
    df_eff_ox.index
    ).unique()


data_dict_list = []
for index_i in shared_indices:
    data_dict_i = dict()

    # #####################################################
    row_feat_i = df_features.loc[index_i]
    # #####################################################
    eff_oxid_state__mine = row_feat_i["features"]["eff_oxid_state"]
    # #####################################################

    # #####################################################
    row_ox_i = df_eff_ox.loc[index_i]
    # #####################################################
    eff_oxid_state__catkit = row_ox_i["form_oxid_state__catkit"]
    job_id_i = row_ox_i.job_id_max
    atoms_catkit_i = row_ox_i.atoms_catkit
    neigh_dict__catkit_i = row_ox_i["neigh_dict"]
    # #####################################################

    index_slabs_to_run = (index_i[0], index_i[1], index_i[4], )
    if index_slabs_to_run in df_slabs_to_run.index:
        row_slab_i = df_slabs_to_run.loc[
            index_slabs_to_run
            ]
        status_i = row_slab_i.status
    else:
        status_i = "NaN"

    # row_slab_i = df_slabs_to_run.loc[
    #     # (name_i[0], name_i[1], name_i[4], )
    #     (index_i[0], index_i[1], index_i[4], )
    #     ]
    # status_i = row_slab_i.status


    if not np.isnan(eff_oxid_state__mine) and not np.isnan(eff_oxid_state__catkit):
        isclose_i = np.isclose(
            eff_oxid_state__mine,
            eff_oxid_state__catkit,
            atol=1e-05,
            equal_nan=True,
            )

        if not isclose_i:
            if True:
            # if status_i == "ok":
                print(
                    status_i,
                    " | ",
                    index_i,
                    ": ",
                    np.round(eff_oxid_state__mine, 3),
                    " | ",
                    np.round(eff_oxid_state__catkit, 3),
                    sep="")

            # #############################################
            data_dict_i["status"] = status_i
            data_dict_i["index"] = index_i
            data_dict_i["compenv"] = index_i[0]
            data_dict_i["slab_id"] = index_i[1]
            data_dict_i["ads"] = index_i[2]
            data_dict_i["active_site"] = index_i[3]
            data_dict_i["att_num"] = index_i[4]
            data_dict_i["from_oh"] = index_i[5]
            data_dict_i["job_id"] = job_id_i
            data_dict_i["atoms_catkit"] = atoms_catkit_i
            data_dict_i["neigh_dict__catkit"] = neigh_dict__catkit_i
            # #############################################
            data_dict_list.append(data_dict_i)
            # #############################################

# #########################################################
df_oxi_comp = pd.DataFrame(data_dict_list)
df_oxi_comp = df_oxi_comp.set_index(["compenv", "slab_id", "ads", "active_site", "att_num", ])
# #########################################################
# -

# ('sherlock', 'filetumi_93', 'o', 67.0, 1) | Kirsten's better
# ('sherlock', 'ramufalu_44', 'o', 54.0, 1) | Kirsten's better
# ('sherlock', 'filetumi_93', 'o', 65.0, 1) | Kirsten's better


# +
# 38 don't match
# 7 with ok 

# + active=""
#
#
#

# +
# assert False

# +
# index_i = ('nersc', 'titawupu_08', 'o', 78.0, 1, False)
# name_i = ('nersc', 'titawupu_08', 'o', 78.0, 1, )

names = [
    # ('sherlock', 'filetumi_93', 'o', 67.0, 1, True),
    # ('sherlock', 'ramufalu_44', 'o', 54.0, 1, True),
    # ('sherlock', 'filetumi_93', 'o', 65.0, 1, True),
    # ('sherlock', 'filetumi_93', 'o', 65.0, 1, False),
    # ('sherlock', 'filetumi_93', 'o', 60.0, 1, False),

    # ('sherlock', 'vinamepa_43', 'o', 77.0, 1, False),

    ('slac', 'ralutiwa_59', 'o', 31.0, 1, False),
    ]

for name_i in names:
    name_i = (name_i[0], name_i[1], name_i[2], name_i[3], name_i[4], )

    print(40 * "-")
    print("name_i:", name_i)
    active_site_i = name_i[3]



    # #########################################################
#     row_oxi_comp_i = df_oxi_comp.loc[name_i].iloc[0]
    row_oxi_comp_i = df_oxi_comp.loc[name_i]
    # #########################################################
    from_oh_i = row_oxi_comp_i.from_oh
    job_id_i = row_oxi_comp_i.job_id
    atoms_catkit_i = row_oxi_comp_i.atoms_catkit
    neigh_dict__catkit_i = row_oxi_comp_i["neigh_dict__catkit"]
    # #########################################################

    # #########################################################
    row_atoms_i = df_atoms_sorted_ind[df_atoms_sorted_ind.job_id == job_id_i]
    row_atoms_i = row_atoms_i.iloc[0]
    # #########################################################
    atoms_i = row_atoms_i.atoms_sorted_good
    name_orig_i = row_atoms_i.name
    # #########################################################


    atoms_i[int(active_site_i)].symbol = "N"

    # #########################################################
    dir_name_i = create_name_str_from_tup(name_i)

    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering/catkit_form_oxid",
        "out_data",
        dir_name_i,
        )
    if not os.path.exists(directory):
        os.makedirs(directory)


    atoms_catkit_i.write(os.path.join(directory, "atoms_catkit.traj"))

    # #########################################################
    # #########################################################
    # #########################################################
    # #########################################################

    df_coord_i = get_df_coord_wrap(name=name_orig_i, active_site=active_site_i)

    if from_oh_i:
        active_site_original = active_site_i
    else:
        active_site_original = "NaN"

    print("My numbers")
    metal_atom_symbol_i = "Ir"
    out_dict = get_effective_ox_state__test(
        # name=name_i,
        name=name_orig_i,
        active_site=active_site_i,
        df_coord_i=df_coord_i,
        metal_atom_symbol=metal_atom_symbol_i,
        active_site_original=name_orig_i[3],
        )
    neigh_dict_i = out_dict["neigh_dict"]
    effective_ox_state_i = out_dict["effective_ox_state"]
    # print(effective_ox_state_i)

    coord_list = []
    for atom in atoms_i:
        coord_i = neigh_dict_i.get(atom.index, 0)
        coord_list.append(coord_i)

    atoms_i.set_initial_charges(np.round(coord_list, 4))

    atoms_i.write(os.path.join(directory, "final.traj"))
    atoms_i.write(os.path.join(directory, "final.cif"))

    # #########################################################
    neigh_keys = list(neigh_dict__catkit_i.keys())

    print("")
    print("catkit numbers")
    for i in np.sort(neigh_keys):
        print(
            i,
            "|",
            neigh_dict__catkit_i[i]
            )

    coord_list = []
    for atom in atoms_i:
        coord_i = neigh_dict__catkit_i.get(atom.index, 0)
        coord_list.append(coord_i)

    atoms_i.set_initial_charges(np.round(coord_list, 4))

    atoms_i.write(os.path.join(directory, "final__catkit.traj"))
    atoms_i.write(os.path.join(directory, "final__catkit.cif"))
# -

assert False

# + active=""
#
#
#

# + jupyter={}
# row_atoms_i = df_atoms_sorted_ind[df_atoms_sorted_ind.job_id == job_id_i]
# row_atoms_i = row_atoms_i.iloc[0]

# + jupyter={}
# df_atoms_sorted_ind[df_atoms_sorted_ind.job_id == job_id_i]

# + jupyter={}
# job_id_i

# + jupyter={}
# df_atoms_sorted_ind.job_id == job_id_i

# + jupyter={}
# from methods_features import original_slab_is_good
# from methods_features import find_missing_O_neigh_with_init_df_coord

# + jupyter={}
# name = name_orig_i
# active_site = active_site_i
# df_coord_i = df_coord_i
# metal_atom_symbol = metal_atom_symbol_i
# active_site_original = name_orig_i[3]

# # def get_effective_ox_state__test(
# #     name=None,
# #     active_site=None,
# #     df_coord_i=None,
# #     metal_atom_symbol="Ir",
# #     active_site_original=None,
# #     ):
# """
# """
# #| - get_effective_ox_state
# # #########################################################
# name_i = name
# active_site_j = active_site
# # #########################################################
# compenv_i = name_i[0]
# slab_id_i = name_i[1]
# ads_i = name_i[2]
# active_site_i = name_i[3]
# att_num_i = name_i[4]
# # #########################################################


# # #########################################################
# #| - Processing central Ir atom nn_info
# df_coord_i = df_coord_i.set_index("structure_index", drop=False)


# import os
# import sys
# import pickle



# # row_coord_i = df_coord_i.loc[21]
# row_coord_i = df_coord_i.loc[active_site_j]

# nn_info_i = row_coord_i.nn_info

# neighbor_count_i = row_coord_i.neighbor_count
# num_Ir_neigh = neighbor_count_i.get("Ir", 0)

# mess_i = "For now only deal with active sites that have 1 Ir neighbor"
# # print("num_Ir_neigh:", num_Ir_neigh)
# assert num_Ir_neigh == 1, mess_i

# for j_cnt, nn_j in enumerate(nn_info_i):
#     site_j = nn_j["site"]
#     elem_j = site_j.as_dict()["species"][0]["element"]

#     if elem_j == metal_atom_symbol:
#         corr_j_cnt = j_cnt

# site_j = nn_info_i[corr_j_cnt]
# metal_index = site_j["site_index"]
# #__|

# # #########################################################
# row_coord_i = df_coord_i.loc[metal_index]

# neighbor_count_i = row_coord_i["neighbor_count"]
# nn_info_i =  row_coord_i.nn_info
# num_neighbors_i = row_coord_i.num_neighbors

# num_O_neigh = neighbor_count_i.get("O", 0)

# six_O_neigh = num_O_neigh == 6
# mess_i = "There should be exactly 6 oxygens about the Ir atom"
# # assert six_O_neigh, mess_i

# six_neigh = num_neighbors_i == 6
# mess_i = "Only 6 neighbors total is allowed, all oxygens"
# # assert six_neigh, mess_i

# skip_this_sys = False
# if not six_O_neigh or not six_neigh:
#     skip_this_sys = True









# from methods import get_df_coord

# init_slab_name_tuple_i = (
#     compenv_i, slab_id_i, ads_i,
#     active_site_original, att_num_i,
#     )
# # print("init_slab_name_tuple_i:", init_slab_name_tuple_i)
# df_coord_orig_slab = get_df_coord(
#     mode="init-slab",
#     init_slab_name_tuple=init_slab_name_tuple_i,
#     )

# orig_slab_good_i = original_slab_is_good(
#     nn_info=nn_info_i,
#     # slab_id=None,
#     metal_index=metal_index,
#     df_coord_orig_slab=df_coord_orig_slab,
#     )




# num_missing_Os = 0
# used_unrelaxed_df_coord = False
# if not six_O_neigh:
#     used_unrelaxed_df_coord = True

#     from methods import get_df_coord
#     init_slab_name_tuple_i = (
#         compenv_i, slab_id_i, ads_i,
#         # active_site_i, att_num_i,
#         active_site_original, att_num_i,
#         )
#     df_coord_orig_slab = get_df_coord(
#         mode="init-slab",
#         init_slab_name_tuple=init_slab_name_tuple_i,
#         )

#     out_dict_0 = find_missing_O_neigh_with_init_df_coord(
#         nn_info=nn_info_i,
#         slab_id=slab_id_i,
#         metal_index=metal_index,
#         df_coord_orig_slab=df_coord_orig_slab,
#         )
#     new_nn_info_i = out_dict_0["nn_info"]
#     num_missing_Os = out_dict_0["num_missing_Os"]
#     orig_slab_good_i = out_dict_0["orig_slab_good"]

#     nn_info_i = new_nn_info_i

#     if new_nn_info_i is not None:
#         skip_this_sys = False
#     else:
#         skip_this_sys = True

# # #####################################################
# effective_ox_state = None
# # if six_O_neigh and six_neigh:
# if not skip_this_sys:
#     #| - Iterating through 6 oxygens
#     second_shell_coord_list = []
#     tmp_list = []

#     print("nn_info_i:", nn_info_i)

#     neigh_dict = dict()
#     for nn_j in nn_info_i:

#         from_orig_df_coord = nn_j.get("from_orig_df_coord", False)
#         if from_orig_df_coord:
#             Ir_neigh_adjustment = 1

#             active_metal_in_nn_list = False
#             for i in df_coord_i.loc[site_index].nn_info:
#                 if i["site_index"] == metal_index:
#                     active_metal_in_nn_list = True

#             if active_metal_in_nn_list:
#                 Ir_neigh_adjustment = 0

#         else:
#             Ir_neigh_adjustment = 0


#         site_index = nn_j["site_index"]

#         row_coord_j = df_coord_i.loc[site_index]

#         neighbor_count_j = row_coord_j.neighbor_count

#         num_Ir_neigh_j = neighbor_count_j.get("Ir", 0)

#         # print(site_index, "|", num_Ir_neigh_j)

#         neigh_dict[site_index] = num_Ir_neigh_j

#         # print("num_Ir_neigh_j:", site_index, num_Ir_neigh_j)
#         num_Ir_neigh_j += Ir_neigh_adjustment

#         # print("num_Ir_neigh_j:", num_Ir_neigh_j)

#         second_shell_coord_list.append(num_Ir_neigh_j)

#         tmp_list.append(2 / num_Ir_neigh_j)

#     # second_shell_coord_list
#     effective_ox_state = np.sum(tmp_list)
#     #__|


# neigh_keys = list(neigh_dict.keys())

# for i in np.sort(neigh_keys):
#     print(
#         i,
#         "|",
#         neigh_dict[i]
#         )

# # #####################################################
# out_dict = dict()
# # #####################################################
# out_dict["effective_ox_state"] = effective_ox_state
# out_dict["used_unrelaxed_df_coord"] = used_unrelaxed_df_coord
# out_dict["num_missing_Os"] = num_missing_Os
# out_dict["orig_slab_good"] = orig_slab_good_i
# out_dict["neigh_dict"] = neigh_dict
# # #####################################################
# # return(out_dict)
# #__|

# + jupyter={}
# effective_ox_state

# + jupyter={}
# metal_index

# + jupyter={}
# df_coord_i.loc[26].nn_info

# + jupyter={}
# active_metal_in_nn_list = False
# for i in df_coord_i.loc[22].nn_info:
#     if i["site_index"] == metal_index:
#         active_metal_in_nn_list = True

# + jupyter={}
# # row_coord_i
# # row_coord_i.nn_info

# nn_info_i

# + jupyter={}
# df_oxi_comp.loc[name_i].iloc[0]
# df_oxi_comp.loc[name_i]

# + jupyter={}
# name_i
