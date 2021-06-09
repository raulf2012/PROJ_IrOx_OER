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

# # Test ML models on different OER set picking heuristics
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

import numpy as np
import pandas as pd

# #########################################################
from methods import (
    get_df_features_targets,
    )

# +
from methods_models import run_gp_workflow

sys.path.insert(0, 
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/model_building"))

from methods_model_building import (
    simplify_df_features_targets,
    run_kfold_cv_wf,
    process_feature_targets_df,
    process_pca_analysis,
    pca_analysis,
    run_regression_wf,
    )
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
    show_plot = True
else:
    from tqdm import tqdm
    verbose = False
    show_plot = False

# ### Script Inputs

# +
num_pca_i = 8

gp_settings = {
    "noise": 0.02542,
    }

# Length scale parameter
sigma_l_default = 1.8  # Length scale parameter
sigma_f_default = 0.2337970892240513  # Scaling parameter.

kdict = [
    {
        'type': 'gaussian',
        'dimension': 'single',
        'width': sigma_l_default,
        'scaling': sigma_f_default,
        'scaling_bounds': ((0.0001, 10.),),
        },
    ]
# -

cols_to_keep = [

    # ('features', 'oh', 'O_magmom'),
    # ('features', 'oh', 'Ir_magmom'),
    # ('features', 'oh', 'active_o_metal_dist'),
    # ('features', 'oh', 'angle_O_Ir_surf_norm'),
    # ('features', 'oh', 'ir_o_mean'),
    # ('features', 'oh', 'ir_o_std'),
    # ('features', 'oh', 'octa_vol'),

    ('features', 'o', 'O_magmom'),
    ('features', 'o', 'Ir_magmom'),
    ('features', 'o', 'active_o_metal_dist'),
    # ('features', 'o', 'angle_O_Ir_surf_norm'),
    ('features', 'o', 'ir_o_mean'),
    ('features', 'o', 'ir_o_std'),
    ('features', 'o', 'octa_vol'),

    # ('features', 'o', 'Ir*O_bader'),
    ('features', 'o', 'Ir_bader'),
    # ('features', 'o', 'O_bader'),
    ('features', 'o', 'p_band_center'),
    # ('features', 'o', 'Ir*O_bader/ir_o_mean'),

    ('features', 'dH_bulk', ''),
    ('features', 'volume_pa', ''),
    ('features', 'bulk_oxid_state', ''),
    ('features', 'effective_ox_state', ''),

    # ('features_pre_dft', 'active_o_metal_dist__pre', ''),
    # ('features_pre_dft', 'ir_o_mean__pre', ''),
    # ('features_pre_dft', 'ir_o_std__pre', ''),
    # ('features_pre_dft', 'octa_vol__pre', ''),

    # #####################################################
    # TARGETS #############################################
    # ('targets', 'e_o', ''),
    # ('targets', 'e_oh', ''),
    # ('targets', 'g_o_m_oh', ''),
    # ('targets', 'e_o_m_oh', ''),

    # ('targets', 'g_o', ''),
    ('targets', 'g_oh', ''),

    ]

# ### Reading Data

df_features_targets = get_df_features_targets()
df_m = df_features_targets

# +
root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/collect_collate_dft_data",
    )

# #########################################################
path_i = os.path.join(root_dir,
    "out_data/df_ads__from_oh.pickle",)
with open(path_i, "rb") as fle:
    df_ads__from_oh = pickle.load(fle)

# #########################################################
path_i = os.path.join(root_dir,
    "out_data/df_ads__low_e.pickle",)
with open(path_i, "rb") as fle:
    df_ads__low_e = pickle.load(fle)

# #########################################################
path_i = os.path.join(root_dir,
    "out_data/df_ads__magmom.pickle",)
with open(path_i, "rb") as fle:
    df_ads__magmom = pickle.load(fle)

# #########################################################
path_i = os.path.join(root_dir,
    "out_data/df_ads__mine.pickle",)
with open(path_i, "rb") as fle:
    df_ads__mine = pickle.load(fle)

# #########################################################
path_i = os.path.join(root_dir,
    "out_data/df_ads__mine_2.pickle",)
with open(path_i, "rb") as fle:
    df_ads__mine_2 = pickle.load(fle)
# -

# ### Set index on OER set dataframes

# +
df_ads__from_oh = df_ads__from_oh.set_index(
    ["compenv", "slab_id", "active_site", ],
    drop=False)

df_ads__low_e = df_ads__low_e.set_index(
    ["compenv", "slab_id", "active_site", ],
    drop=False)

df_ads__magmom = df_ads__magmom.set_index(
    ["compenv", "slab_id", "active_site", ],
    drop=False)

df_ads__mine = df_ads__mine.set_index(
    ["compenv", "slab_id", "active_site", ],
    drop=False)

df_ads__mine_2 = df_ads__mine_2.set_index(
    ["compenv", "slab_id", "active_site", ],
    drop=False)

# +
df_m_wo_y = df_m.drop(
    columns=[
        ("targets", "g_o", "", ),
        ("targets", "g_oh", "", ),
        ],
    )

df_m_wo_y.iloc[0:2]
# -

# ## `from_oh`

# + jupyter={"source_hidden": true}
# #########################################################
df_ads__from_oh_y = df_ads__from_oh[["g_o", "g_oh", ]]

new_cols = []
for col_i in df_ads__from_oh_y.columns:
    new_col_i = ("targets", col_i, "", )
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_ads__from_oh_y.columns = idx

# #########################################################
df_m__from_oh = pd.concat([
    df_m_wo_y,
    df_ads__from_oh_y,
    ], axis=1)

df_m__from_oh = df_m__from_oh.reindex(
    columns=list(df_m__from_oh.columns.levels[0]),
    level=0)

# #########################################################
df_m__from_oh_2 = df_m__from_oh[
    cols_to_keep
    ]

# + jupyter={"source_hidden": true}
adsorbates = ["o", "oh", "ooh", ] 
new_cols = []
for col_i in df_m__from_oh_2.columns:
    # print(col_i)

    new_col_i = None
    if col_i[0] == "targets":
        new_col_i = ("targets", col_i[1], )
    elif col_i[0] == "features" and col_i[1] in adsorbates:
        new_col_i = ("features", col_i[2], )
    elif col_i[0] == "features" and col_i[2] == "":
        new_col_i = ("features", col_i[1], )
    else:
        print("Woops")

    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_m__from_oh_2.columns = idx

df_m__from_oh_2 = df_m__from_oh_2.dropna(how="any")
# -

df_m__from_oh_2.shape

# + jupyter={"source_hidden": true}
cols_to_use = df_m__from_oh_2["features"].columns.tolist()

out_dict = run_kfold_cv_wf(
    df_features_targets=df_m__from_oh_2,
    cols_to_use=cols_to_use,
    run_pca=True,
    num_pca_comp=num_pca_i,
    k_fold_partition_size=30,
    model_workflow=run_gp_workflow,
    model_settings=dict(
        gp_settings=gp_settings,
        kdict=kdict,
        ),
    )
# #####################################################
df_target_pred = out_dict["df_target_pred"]
MAE = out_dict["MAE"]
R2 = out_dict["R2"]
PCA = out_dict["pca"]
regression_model_list = out_dict["regression_model_list"]

df_target_pred_on_train = out_dict["df_target_pred_on_train"]
MAE_pred_on_train = out_dict["MAE_pred_on_train"]
RM_2 = out_dict["RM_2"]
# #####################################################

if verbose:
    print(
        "MAE: ",
        np.round(MAE, 5),
        " eV",
        sep="")

    print(
        "R2: ",
        np.round(R2, 5),
        sep="")

    print(
        "MAE (predicting on train set): ",
        np.round(MAE_pred_on_train, 5),
        sep="")
# -

# ## `low_e`

# + jupyter={"source_hidden": true}
# #########################################################
df_ads__low_e_y = df_ads__low_e[["g_o", "g_oh", ]]

new_cols = []
for col_i in df_ads__low_e_y.columns:
    new_col_i = ("targets", col_i, "", )
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_ads__low_e_y.columns = idx

# #########################################################
df_m__low_e = pd.concat([
    df_m_wo_y,
    df_ads__low_e_y,
    ], axis=1)

df_m__low_e = df_m__low_e.reindex(
    columns=list(df_m__low_e.columns.levels[0]),
    level=0)

# #########################################################
df_m__low_e_2 = df_m__low_e[
    cols_to_keep
    ]

# + jupyter={"source_hidden": true}
adsorbates = ["o", "oh", "ooh", ] 
new_cols = []
for col_i in df_m__low_e_2.columns:
    # print(col_i)

    new_col_i = None
    if col_i[0] == "targets":
        new_col_i = ("targets", col_i[1], )
    elif col_i[0] == "features" and col_i[1] in adsorbates:
        new_col_i = ("features", col_i[2], )
    elif col_i[0] == "features" and col_i[2] == "":
        new_col_i = ("features", col_i[1], )
    else:
        print("Woops")

    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_m__low_e_2.columns = idx

df_m__low_e_2 = df_m__low_e_2.dropna(how="any")

# + jupyter={"source_hidden": true}
cols_to_use = df_m__low_e_2["features"].columns.tolist()

out_dict = run_kfold_cv_wf(
    df_features_targets=df_m__low_e_2,
    cols_to_use=cols_to_use,
    run_pca=True,
    num_pca_comp=num_pca_i,
    k_fold_partition_size=30,
    model_workflow=run_gp_workflow,
    model_settings=dict(
        gp_settings=gp_settings,
        kdict=kdict,
        ),
    )
# #####################################################
df_target_pred = out_dict["df_target_pred"]
MAE = out_dict["MAE"]
R2 = out_dict["R2"]
PCA = out_dict["pca"]
regression_model_list = out_dict["regression_model_list"]

df_target_pred_on_train = out_dict["df_target_pred_on_train"]
MAE_pred_on_train = out_dict["MAE_pred_on_train"]
RM_2 = out_dict["RM_2"]
# #####################################################

if verbose:
    print(
        "MAE: ",
        np.round(MAE, 5),
        " eV",
        sep="")

    print(
        "R2: ",
        np.round(R2, 5),
        sep="")

    print(
        "MAE (predicting on train set): ",
        np.round(MAE_pred_on_train, 5),
        sep="")
# -

# ## `magmom`

# + jupyter={"source_hidden": true}
# #########################################################
df_ads__magmom_y = df_ads__magmom[["g_o", "g_oh", ]]

new_cols = []
for col_i in df_ads__magmom_y.columns:
    new_col_i = ("targets", col_i, "", )
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_ads__magmom_y.columns = idx

# #########################################################
df_m__magmom = pd.concat([
    df_m_wo_y,
    df_ads__magmom_y,
    ], axis=1)

df_m__magmom = df_m__magmom.reindex(
    columns=list(df_m__magmom.columns.levels[0]),
    level=0)

# #########################################################
df_m__magmom_2 = df_m__magmom[
    cols_to_keep
    ]

# + jupyter={"source_hidden": true}
adsorbates = ["o", "oh", "ooh", ] 
new_cols = []
for col_i in df_m__magmom_2.columns:
    # print(col_i)

    new_col_i = None
    if col_i[0] == "targets":
        new_col_i = ("targets", col_i[1], )
    elif col_i[0] == "features" and col_i[1] in adsorbates:
        new_col_i = ("features", col_i[2], )
    elif col_i[0] == "features" and col_i[2] == "":
        new_col_i = ("features", col_i[1], )
    else:
        print("Woops")

    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_m__magmom_2.columns = idx

df_m__magmom_2 = df_m__magmom_2.dropna(how="any")

# + jupyter={"source_hidden": true}
cols_to_use = df_m__magmom_2["features"].columns.tolist()

out_dict = run_kfold_cv_wf(
    df_features_targets=df_m__magmom_2,
    cols_to_use=cols_to_use,
    run_pca=True,
    num_pca_comp=num_pca_i,
    k_fold_partition_size=30,
    model_workflow=run_gp_workflow,
    model_settings=dict(
        gp_settings=gp_settings,
        kdict=kdict,
        ),
    )
# #####################################################
df_target_pred = out_dict["df_target_pred"]
MAE = out_dict["MAE"]
R2 = out_dict["R2"]
PCA = out_dict["pca"]
regression_model_list = out_dict["regression_model_list"]

df_target_pred_on_train = out_dict["df_target_pred_on_train"]
MAE_pred_on_train = out_dict["MAE_pred_on_train"]
RM_2 = out_dict["RM_2"]
# #####################################################

if verbose:
    print(
        "MAE: ",
        np.round(MAE, 5),
        " eV",
        sep="")

    print(
        "R2: ",
        np.round(R2, 5),
        sep="")

    print(
        "MAE (predicting on train set): ",
        np.round(MAE_pred_on_train, 5),
        sep="")
# -

# ## `mine`

# + jupyter={"source_hidden": true}
# #########################################################
df_ads__mine_y = df_ads__mine[["g_o", "g_oh", ]]

new_cols = []
for col_i in df_ads__mine_y.columns:
    new_col_i = ("targets", col_i, "", )
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_ads__mine_y.columns = idx

# #########################################################
df_m__mine = pd.concat([
    df_m_wo_y,
    df_ads__mine_y,
    ], axis=1)

df_m__mine = df_m__mine.reindex(
    columns=list(df_m__mine.columns.levels[0]),
    level=0)

# #########################################################
df_m__mine_2 = df_m__mine[
    cols_to_keep
    ]

# + jupyter={"source_hidden": true}
adsorbates = ["o", "oh", "ooh", ] 
new_cols = []
for col_i in df_m__mine_2.columns:
    # print(col_i)

    new_col_i = None
    if col_i[0] == "targets":
        new_col_i = ("targets", col_i[1], )
    elif col_i[0] == "features" and col_i[1] in adsorbates:
        new_col_i = ("features", col_i[2], )
    elif col_i[0] == "features" and col_i[2] == "":
        new_col_i = ("features", col_i[1], )
    else:
        print("Woops")

    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_m__mine_2.columns = idx

df_m__mine_2 = df_m__mine_2.dropna(how="any")

# + jupyter={"source_hidden": true}
cols_to_use = df_m__mine_2["features"].columns.tolist()

out_dict = run_kfold_cv_wf(
    df_features_targets=df_m__mine_2,
    cols_to_use=cols_to_use,
    run_pca=True,
    num_pca_comp=num_pca_i,
    k_fold_partition_size=30,
    model_workflow=run_gp_workflow,
    model_settings=dict(
        gp_settings=gp_settings,
        kdict=kdict,
        ),
    )
# #####################################################
df_target_pred = out_dict["df_target_pred"]
MAE = out_dict["MAE"]
R2 = out_dict["R2"]
PCA = out_dict["pca"]
regression_model_list = out_dict["regression_model_list"]

df_target_pred_on_train = out_dict["df_target_pred_on_train"]
MAE_pred_on_train = out_dict["MAE_pred_on_train"]
RM_2 = out_dict["RM_2"]
# #####################################################

if verbose:
    print(
        "MAE: ",
        np.round(MAE, 5),
        " eV",
        sep="")

    print(
        "R2: ",
        np.round(R2, 5),
        sep="")

    print(
        "MAE (predicting on train set): ",
        np.round(MAE_pred_on_train, 5),
        sep="")
# -

# ## `mine_2`

# + jupyter={"source_hidden": true}
# #########################################################
df_ads__mine_2_y = df_ads__mine_2[["g_o", "g_oh", ]]

new_cols = []
for col_i in df_ads__mine_2_y.columns:
    new_col_i = ("targets", col_i, "", )
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_ads__mine_2_y.columns = idx

# #########################################################
df_m__mine_2 = pd.concat([
    df_m_wo_y,
    df_ads__mine_2_y,
    ], axis=1)

df_m__mine_2 = df_m__mine_2.reindex(
    columns=list(df_m__mine_2.columns.levels[0]),
    level=0)

# #########################################################
df_m__mine_2_2 = df_m__mine_2[
    cols_to_keep
    ]

# + jupyter={"source_hidden": true}
adsorbates = ["o", "oh", "ooh", ] 
new_cols = []
for col_i in df_m__mine_2_2.columns:
    # print(col_i)

    new_col_i = None
    if col_i[0] == "targets":
        new_col_i = ("targets", col_i[1], )
    elif col_i[0] == "features" and col_i[1] in adsorbates:
        new_col_i = ("features", col_i[2], )
    elif col_i[0] == "features" and col_i[2] == "":
        new_col_i = ("features", col_i[1], )
    else:
        print("Woops")

    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)

df_m__mine_2_2.columns = idx

df_m__mine_2_2 = df_m__mine_2_2.dropna(how="any")
# -

df_m__mine_2_2.shape

# + jupyter={"source_hidden": true}
cols_to_use = df_m__mine_2_2["features"].columns.tolist()

out_dict = run_kfold_cv_wf(
    df_features_targets=df_m__mine_2_2,
    cols_to_use=cols_to_use,
    run_pca=True,
    num_pca_comp=num_pca_i,
    k_fold_partition_size=30,
    model_workflow=run_gp_workflow,
    model_settings=dict(
        gp_settings=gp_settings,
        kdict=kdict,
        ),
    )
# #####################################################
df_target_pred = out_dict["df_target_pred"]
MAE = out_dict["MAE"]
R2 = out_dict["R2"]
PCA = out_dict["pca"]
regression_model_list = out_dict["regression_model_list"]

df_target_pred_on_train = out_dict["df_target_pred_on_train"]
MAE_pred_on_train = out_dict["MAE_pred_on_train"]
RM_2 = out_dict["RM_2"]
# #####################################################

if verbose:
    print(
        "MAE: ",
        np.round(MAE, 5),
        " eV",
        sep="")

    print(
        "R2: ",
        np.round(R2, 5),
        sep="")

    print(
        "MAE (predicting on train set): ",
        np.round(MAE_pred_on_train, 5),
        sep="")
# -

assert False

# + active=""
#
#
#
#
#
#
# -

# ### Predicting on *OH results

# + active=""
# # FROM OH
# MAE: 0.18735 eV
# R2: 0.70906
# MAE (predicting on train set): 0.14474
#
# # LOW E
# MAE: 0.19039 eV
# R2: 0.7025
# MAE (predicting on train set): 0.10353
#
# # MAGMOM
# MAE: 0.19125 eV
# R2: 0.72463
# MAE (predicting on train set): 0.08905
#
# # MINE
# MAE: 0.18998 eV
# R2: 0.70478
# MAE (predicting on train set): 0.08904
#
# # MINE_2
# MAE: 0.18941 eV
# R2: 0.70577
# MAE (predicting on train set): 0.14718
# -

# ### Predicting on *O results

# + active=""
# # FROM OH
# MAE: 0.19534 eV
# R2: 0.78813
# MAE (predicting on train set): 0.15341
#
# # LOW E
# MAE: 0.18201 eV
# R2: 0.82162
# MAE (predicting on train set): 0.13367
#
# # MAGMOM
# MAE: 0.21635 eV
# R2: 0.7337
# MAE (predicting on train set): 0.17447
#
# # MINE
# MAE: 0.18226 eV
# R2: 0.81959
# MAE (predicting on train set): 0.13481

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# os.environ[""],

# + jupyter={"source_hidden": true}
# # #########################################################
# # Pickling data ###########################################
# directory = os.path.join(
#     root_dir, "out_data")
# if not os.path.exists(directory): os.makedirs(directory)
# with open(os.path.join(directory, "df_ads__magmom.pickle"), "wb") as fle:
#     pickle.dump(df_ads__magmom, fle)
# # #########################################################

# + jupyter={"source_hidden": true}
# df_ads.pickle
# df_dict.pickle

# + jupyter={"source_hidden": true}
# df_ads__from_oh.pickle
# df_ads__low_e.pickle
# df_ads__magmom.pickle

# + jupyter={"source_hidden": true}
# df_m__from_oh.sort_
# df_m__from_oh = 
# df_m__from_oh.reindex(columns=["data", "features", ], level=0)
# df_m__from_oh.reindex(columns=["targets", ], level=0)
# ["targets", ]

# + jupyter={"source_hidden": true}
# list(df_m__from_oh.columns.levels[0])

# + jupyter={"source_hidden": true}
# df_m["targets"]

# df_m.columns.tolist()

# + jupyter={"source_hidden": true}
# for i in new_cols:
#     print(i)

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_j = df_m__from_oh_2

# +
# for name_i, row_i in df_ads__magmom.iterrows():
#     # name_i

#     # #####################################################
#     job_id_o_i = row_i.job_id_o
#     job_id_oh_i = row_i.job_id_oh
#     job_id_bare_i = row_i.job_id_bare
#     # #####################################################

#     # #####################################################
#     row_mine_i = df_ads__mine.loc[name_i]
#     # #####################################################
#     job_id_o_i_2 = row_mine_i.job_id_o
#     job_id_oh_i_2 = row_mine_i.job_id_oh
#     job_id_bare_i_2 = row_mine_i.job_id_bare
#     # #####################################################

#     if not job_id_o_i == job_id_o_i_2:
#         print("IJI")

#     if not job_id_oh_i == job_id_oh_i_2:
#         print("IJI")

#     if not job_id_bare_i == job_id_bare_i_2:
#         print("IJI")

# +
# job_id_

# + active=""
# # FROM_OH
# MAE: 0.18735 eV
# R2: 0.70906
# MAE (predicting on train set): 0.14474
#
# # LOW_E
# MAE: 0.19039 eV
# R2: 0.7025
# MAE (predicting on train set): 0.10353
#
# # MAGMOM
# MAE: 0.19125 eV
# R2: 0.72463
# MAE (predicting on train set): 0.08905

# + active=""
# # FROM OH
# MAE: 0.19001 eV
# R2: 0.71487
# MAE (predicting on train set): 0.13976
#
# # LOW E
# MAE: 0.1893 eV
# R2: 0.70264
# MAE (predicting on train set): 0.11304
#
# # MAGMOM
# MAE: 0.1932 eV
# R2: 0.70798
# MAE (predicting on train set): 0.1057
