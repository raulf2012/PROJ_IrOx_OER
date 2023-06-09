# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Constructing linear model for OER adsorption energies
# ---
#

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

import plotly.graph_objects as go

# #########################################################
from methods import (
    get_df_features_targets,
    get_df_features_targets_seoin,
    )

from methods_models import ModelAgent, GP_Regression

from proj_data import adsorbates
from proj_data import layout_shared
from proj_data import scatter_marker_props
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

root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/model_building/linear_models/my_data")

# ### Script Inputs

# +
target_ads_i = "oh"

feature_ads_i = "o"

use_seoin_data = False

if use_seoin_data:
    feature_ads_i = "o"
# -
quick_easy_settings = False
if quick_easy_settings:
    k_fold_partition_size = 170
    do_every_nth_pca_comp = 8
else:
    k_fold_partition_size = 10
    do_every_nth_pca_comp = 1

# +
# # TEMP
# print(111 * "TEMP | ")
# do_every_nth_pca_comp = 9
# -

# ### Read Data

# +
# #########################################################
df_features_targets = get_df_features_targets()

# #########################################################
df_seoin = get_df_features_targets_seoin()
# -

# ### Combine mine and Seoin's data

if use_seoin_data:
    # Replace multiindex with index of tuples so that my data and Seoin's data can be combined
    indices = df_features_targets.index.tolist()
    df_features_targets.index = indices

    indices = df_seoin.index.tolist()
    df_seoin.index = indices

    # Remove columns that aren't shared by my and Seoin's data
    cols_0 =df_features_targets.columns.tolist()
    cols_1 = df_seoin.columns.tolist()

    cols_comb = cols_0 + cols_1

    cols_comb_unique = []
    for col_i in cols_comb:
        if col_i not in cols_comb_unique:
            cols_comb_unique.append(col_i)

    shared_cols = []
    for col_i in cols_comb_unique:
        if col_i in df_features_targets.columns and col_i in df_seoin.columns:
            shared_cols.append(col_i)

    # Combine data
    df_data = pd.concat([
        df_features_targets[shared_cols],
        df_seoin[shared_cols],
        ], axis=0)
else:
    df_data = df_features_targets

# +
# # TEMP
# print(222 * "TEMP | ")

# df_data = df_data[df_data.data.stoich == "AB3"]
# -

df_data = df_seoin

# +
df_data = df_data[[
    ('targets', 'g_o', ''),
#     ('targets', 'g_oh', ''),
    ('data', 'stoich', ''),

    ('features', 'bulk_oxid_state', ''),
    ('features', 'dH_bulk', ''),
    ('features', 'effective_ox_state', ''),
    ('features', 'volume_pa', ''),
    ('features', 'o', 'active_o_metal_dist'),
    # ('features', 'o', 'angle_O_Ir_surf_norm'),
    ('features', 'o', 'ir_o_mean'),
    ('features', 'o', 'ir_o_std'),
    ('features', 'o', 'octa_vol')

    ]]

df_data

# +
from methods_models import Linear_Regression

L_R = Linear_Regression()

# +
data_dict_list = []
num_feat_cols = df_data.features.shape[1]
# for num_pca_i in range(1, num_feat_cols + 1, do_every_nth_pca_comp):
for num_pca_i in range(4, num_feat_cols + 1, do_every_nth_pca_comp):

    if verbose:
        print("")
        print(40 * "*")
        print(num_pca_i)

    MA = ModelAgent(
        df_features_targets=df_data,
        # Regression=GP_R,
        Regression=L_R,
        Regression_class=Linear_Regression,
        use_pca=True,
        num_pca=num_pca_i,
        adsorbates=adsorbates,
        stand_targets=False,  # True was giving much worse errors, keep False
        )

    MA.run_kfold_cv_workflow(
        k_fold_partition_size=k_fold_partition_size,
        )

    if MA.can_run:
        if verbose:
            print("MAE:", np.round(MA.mae, 4))
            print("MA.r2:", np.round(MA.r2, 4))
            print("MAE (in_fold):", np.round(MA.mae_infold, 4))

    data_dict_i = dict()
    data_dict_i["num_pca"] = num_pca_i
    data_dict_i["MAE"] = MA.mae
    data_dict_i["ModelAgent"] = MA
    data_dict_list.append(data_dict_i)

df_models = pd.DataFrame(data_dict_list)
df_models = df_models.set_index("num_pca")




# #########################################################
# Finding best performing model
row_models_i = df_models.sort_values("MAE").iloc[0]

MA_best = row_models_i.ModelAgent

print(4 * "\n")
if verbose:
    print(
        row_models_i.name,
        " PCA components are ideal with an MAE of ",
        np.round(
        row_models_i.MAE,
            4),
        sep="")

# +
# 9 PCA components are ideal with an MAE of 0.1282



# +
from methods_models import ModelAgent_Plotter

MA_Plot = ModelAgent_Plotter(
    ModelAgent=MA_best,
    layout_shared=layout_shared,
    )

MA_Plot.plot_residuals()
MA_Plot.plot_parity()
MA_Plot.plot_parity_infold()

# # Uncomment to run pca analysis on in-fold regression
# MA.run_pca_analysis()
# -

fig = MA_Plot.plot_residuals__PLT
if show_plot:
    fig.show()

fig = MA_Plot.plot_parity__PLT
if show_plot:
    fig.show()

fig = MA_Plot.plot_parity_infold__PLT
if show_plot:
    fig.show()

from methods_models import plot_mae_vs_pca
plot_mae_vs_pca(
    df_models=df_models,
    layout_shared=layout_shared,
    scatter_marker_props=scatter_marker_props,
    )

# ### Save Data

# +
# Deleting cinv matrix of GP model to save disk space

for num_pca, row_i in df_models.iterrows():
    MA = row_i.ModelAgent
    # MA.cleanup_for_pickle()
# -

data_dict_out = {
    "df_models": df_models,
    "ModelAgent_Plot": MA_Plot,
    }

# Pickling data ###########################################
directory = os.path.join(root_dir, "out_data")
print(directory)
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "modelling_data.pickle"), "wb") as fle:
    pickle.dump(data_dict_out, fle)
# #########################################################

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("model__mine_GP.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#

# + jupyter={"source_hidden": true}
# kdict = [
#     {
#         "type": "gaussian",
#         "dimension": "single",
#         "width": 1.8,
#         "scaling": 0.5,
#         "scaling_bounds": ((0.0001, 10.),),
#         }
#     ]

# GP_R = GP_Regression(
#     kernel_list=kdict,
#     regularization=0.01,
#     optimize_hyperparameters=True,
#     scale_data=False,
#     )

# + jupyter={"source_hidden": true}
# assert False
