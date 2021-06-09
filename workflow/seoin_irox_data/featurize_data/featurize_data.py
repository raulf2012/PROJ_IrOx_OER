# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# ### Featurize IrOx slabs from Seoin
# ---

# +
# #########################################################
# This haven't been done
# #########################################################
# ('features', 'o', 'O_magmom'),
# ('features', 'o', 'Ir_magmom'),
# ('features', 'o', 'Ir*O_bader'),
# ('features', 'o', 'Ir_bader'),
# ('features', 'o', 'O_bader'),
# ('features', 'o', 'p_band_center'),



# #########################################################
# These are done
# #########################################################
# ('features', 'o', 'bulk_oxid_state'),
# ('features', 'o', 'angle_O_Ir_surf_norm'),
# ('features', 'o', 'active_o_metal_dist'),
# ('features', 'o', 'effective_ox_state'),
# ('features', 'o', 'ir_o_mean'),
# ('features', 'o', 'ir_o_std'),
# ('features', 'o', 'octa_vol'),
# ('features', 'o', 'dH_bulk'),
# ('features', 'o', 'volume_pa'),
# -

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

import pickle
from pathlib import Path

import pandas as pd
import numpy as np

# #########################################################
from methods_features import get_octa_geom, get_octa_vol
from methods_features import get_angle_between_surf_normal_and_O_Ir

# #########################################################
from local_methods import get_df_coord_local
from local_methods import get_effective_ox_state
# -

pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

# +
dir_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data")

# #########################################################
path_i = os.path.join(
    dir_i, "out_data/df_ads_e.pickle")
with open(path_i, "rb") as fle:
    df_ads_e = pickle.load(fle)

# #########################################################
path_i = os.path.join(
    dir_i, "out_data/df_oer.pickle")
with open(path_i, "rb") as fle:
    df_oer = pickle.load(fle)

# #########################################################
path_i = os.path.join(
    dir_i, "process_bulk_data",
    "out_data/df_seoin_bulk.pickle")
with open(path_i, "rb") as fle:
    df_bulk = pickle.load(fle)
df_bulk = df_bulk.set_index("crystal")

# +
# # TEMP
# print(111 * "TEMP | ")

# df_ads_e = df_ads_e.dropna(axis=0, subset=["active_site__o", "active_site__oh", "active_site__ooh"])

# +
data_dict_list = []
for name_i, row_i in df_ads_e.iterrows():
    # #####################################################
    name_dict_i = dict(zip(
        df_ads_e.index.names,
        name_i))
    # #####################################################
    name_str_i = row_i["name"]
    index_o_i = row_i.index_o
    active_site_o_i = row_i.active_site__o
    # bulk_oxid_state_i = row_i.bulk_oxid_state
    # #####################################################
    crystal_i = name_dict_i["crystal"]
    # #####################################################

    # #####################################################
    row_oer_o_i = df_oer.loc[index_o_i]
    # #####################################################
    atoms_o_i = row_oer_o_i.atoms
    atoms_o_init_i = row_oer_o_i.atoms_init
    # #####################################################

    # #####################################################
    row_bulk_i = df_bulk.loc[crystal_i]
    # #####################################################
    volume_pa_i = row_bulk_i.volume_pa
    dH_i = row_bulk_i.dH
    # #####################################################



    df_coord_o_final_i = get_df_coord_local(
        name=name_str_i,
        ads="o",
        atoms=atoms_o_i,
        append_str="_final",
        )
    df_coord_o_init_i = get_df_coord_local(
        name=name_str_i,
        ads="o",
        atoms=atoms_o_init_i,
        append_str="_init",
        )

    eff_ox_out_i = get_effective_ox_state(
        active_site=active_site_o_i,
        df_coord_i=df_coord_o_final_i,
        df_coord_init_i=df_coord_o_init_i,
        metal_atom_symbol="Ir",
        )
    eff_ox_i = eff_ox_out_i["effective_ox_state"]


    # #####################################################
    # Octahedral geometry
    octa_geom_out = get_octa_geom(
        df_coord_i=df_coord_o_final_i,
        active_site_j=active_site_o_i,
        atoms=atoms_o_i,
        verbose=True,
        )
    for key_i in octa_geom_out.keys():
        octa_geom_out[key_i + "__o"] = octa_geom_out.pop(key_i)

    octa_vol_i = get_octa_vol(
        df_coord_i=df_coord_o_final_i,
        active_site_j=active_site_o_i,
        verbose=True,
        )


    # #####################################################
    # Ir-O Angle relative to surface normal
    angle_i = get_angle_between_surf_normal_and_O_Ir(
        atoms_o_i,
        df_coord=df_coord_o_final_i,
        active_site=active_site_o_i,
        )


    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["effective_ox_state__o"] = eff_ox_i
    data_dict_i["octa_vol__o"] = octa_vol_i
    data_dict_i["angle_O_Ir_surf_norm__o"] = angle_i
    data_dict_i["dH_bulk"] = dH_i
    data_dict_i["volume_pa"] = volume_pa_i
    # data_dict_i["bulk_oxid_state"] = bulk_oxid_state_i
    # #####################################################
    data_dict_i.update(octa_geom_out)
    data_dict_i.update(name_dict_i)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_feat = pd.DataFrame(data_dict_list)
df_feat = df_feat.set_index(df_ads_e.index.names)

df_features_targets = pd.concat([
    df_feat,
    df_ads_e.drop(columns=["O_Ir_frac_ave", ])
    ], axis=1)
# #########################################################
# -

# ### Processing columns

# +
df_features_targets.columns.tolist()


multicolumn_assignments = {

    # #######################
    # Features ##############
    "effective_ox_state__o":   ("features", "effective_ox_state", "", ),

    # "effective_ox_state__o":   ("features", "o", "effective_ox_state", ),
    "octa_vol__o":             ("features", "o", "octa_vol", ),
    "active_o_metal_dist__o":  ("features", "o", "active_o_metal_dist", ),
    "ir_o_mean__o":            ("features", "o", "ir_o_mean", ),
    "ir_o_std__o":             ("features", "o", "ir_o_std", ),
    "angle_O_Ir_surf_norm__o": ("features", "o", "angle_O_Ir_surf_norm", ),

    "bulk_oxid_state":         ("features", "bulk_oxid_state", "", ),
    "dH_bulk":                 ("features", "dH_bulk", "", ),
    "volume_pa":               ("features", "volume_pa", "", ),


    # #######################
    # Targets ###############
    "e_o":   ("targets", "e_o", "", ),
    "e_oh":  ("targets", "e_oh", "", ),
    "e_ooh": ("targets", "e_ooh", "", ),
    "g_o":   ("targets", "g_o", "", ),
    "g_oh":  ("targets", "g_oh", "", ),
    "g_ooh": ("targets", "g_ooh", "", ),

    # #######################
    # Data ##################
    "index_bare":          ("data", "index_bare", "", ),
    "index_o":             ("data", "index_o", "", ),
    "index_oh":            ("data", "index_oh", "", ),
    "index_ooh":           ("data", "index_ooh", "", ),
    "name":                ("data", "name", "", ),
    "active_site__o":      ("data", "active_site__o", "", ),
    "active_site__oh":     ("data", "active_site__oh", "", ),
    "active_site__ooh":    ("data", "active_site__ooh", "", ),

    "stoich":    ("data", "stoich", "", ),

    }

# +
new_cols = []
for col_i in df_features_targets.columns:
    new_col_i = multicolumn_assignments.get(col_i, col_i)
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)
df_features_targets.columns = idx
# -

df_features_targets = df_features_targets.reindex(columns=[
    "targets",
    "data",
    "format",
    "features",
    "features_pre_dft",
    "features_stan",
    ], level=0)

df_features_targets = df_features_targets.sort_index(axis=1)

# new_cols = []
other_cols = []
other_feature_cols = []
ads_feature_cols = []
for col_i in df_features_targets.columns:

    if col_i[0] == "features":
        if col_i[1] in ["o", "oh", "ooh", ]:
            # print(col_i)
            ads_feature_cols.append(col_i)
        else:
            other_feature_cols.append(col_i)

    else:
        other_cols.append(col_i)

df_features_targets = df_features_targets[
    other_cols + other_feature_cols + ads_feature_cols
    ]

# ### Write data to file

# +
# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data/featurize_data",
    "out_data")
if not os.path.exists(directory):
    os.makedirs(directory)

path_i = os.path.join(directory, "df_features_targets.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_features_targets, fle)
# #########################################################
# -

df_features_targets.head()

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_features_targets

# + jupyter={"source_hidden": true}
# # df_features_targets.sort_values([("features", ) ])
# df_features_targets.columns = df_features_targets.columns.sortlevel()[0]

# + jupyter={"source_hidden": true}
# df_features_targets

# + jupyter={"source_hidden": true}
# # df_features_targets = 
# df_features_targets.reindex(columns=[
#     # "targets",
#     # "data",
#     # "format",
#     "features",
#     # "features_pre_dft",
#     # "features_stan",
#     ], level=0)

# + jupyter={"source_hidden": true}
# df_ads_e.index.to_frame().crystal.unique().tolist()

# + jupyter={"source_hidden": true}
# row_i
# -



# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_features_targets.columns = df_features_targets.columns.sortlevel()[0]

# + jupyter={"source_hidden": true}
# df_features_targets

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_features_targets.columns

# + jupyter={"source_hidden": true}
# df_features_targets["features"]

# + jupyter={"source_hidden": true}
# assert False

# +
# df_ads_e

# +
# df_features_targets

# +
# assert False

# +
# df_features_targets["effective_ox_state__o"].tolist()

# +
# df_features_targets
