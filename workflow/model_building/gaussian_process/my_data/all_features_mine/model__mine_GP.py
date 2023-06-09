# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Constructing linear model for OER adsorption energies
# ---
#

# +
# 12 PCA components are ideal with an MAE of 0.1872
# -

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
    "workflow/model_building/gaussian_process/my_data/all_features_mine")

quick_easy_settings = True
if quick_easy_settings:
    k_fold_partition_size = 30
    do_every_nth_pca_comp = 1
else:
    k_fold_partition_size = 10
    do_every_nth_pca_comp = 1

# ### Read Data

# +
# #########################################################
df_features_targets = get_df_features_targets()

# #########################################################
df_seoin = get_df_features_targets_seoin()
# -

# ### Combine mine and Seoin's data

df_data = df_features_targets

# +
# # TEMP
# print(222 * "TEMP | ")

# df_data = df_data[df_data.data.stoich == "AB3"]

# # df_data = df_data[df_data.data.stoich == "AB2"]
# -

# ### Choosing feature columns

df_data = df_data[[
    # ('targets', 'g_o', ''),
    ('targets', 'g_oh', ''),

    # ('targets', 'e_o', ''),
    # ('targets', 'e_oh', ''),
    # ('targets', 'g_o_m_oh', ''),
    # ('targets', 'e_o_m_oh', ''),

    ('data', 'job_id_o', ''),
    ('data', 'job_id_oh', ''),
    ('data', 'job_id_bare', ''),
    ('data', 'stoich', ''),

    # ('features', 'oh', 'O_magmom'),
    # ('features', 'oh', 'Ir_magmom'),
    # ('features', 'oh', 'active_o_metal_dist'),
    # ('features', 'oh', 'angle_O_Ir_surf_norm'),
    # ('features', 'oh', 'closest_Ir_dist'),
    # ('features', 'oh', 'closest_O_dist'),
    # ('features', 'oh', 'ir_o_mean'),
    # ('features', 'oh', 'ir_o_std'),
    # ('features', 'oh', 'octa_vol'),
    # ('features', 'oh', 'oxy_opp_as_bl'),
    # ('features', 'oh', 'degrees_off_of_straight__as_opp'),
    # ('features', 'oh', 'as_ir_opp_bl_ratio'),

    # ('features', 'o', 'O_magmom'),  # HELPS
    # ('features', 'o', 'Ir_magmom'),
    # ('features', 'o', 'Ir*O_bader'),
    # ('features', 'o', 'Ir_bader'),
    # ('features', 'o', 'O_bader'),
    # ('features', 'o', 'p_band_center'),  # HELPS
    # ('features', 'o', 'Ir*O_bader/ir_o_mean'),

    ('features', 'o', 'active_o_metal_dist'),
    # ('features', 'o', 'angle_O_Ir_surf_norm'),
    # ('features', 'o', 'closest_Ir_dist'),  # NO
    # ('features', 'o', 'closest_O_dist'),  # NO
    ('features', 'o', 'ir_o_mean'),
    # ('features', 'o', 'ir_o_std'), # NO
    ('features', 'o', 'octa_vol'),
    ('features', 'o', 'oxy_opp_as_bl'),
    ('features', 'o', 'degrees_off_of_straight__as_opp'),
    # ('features', 'o', 'as_ir_opp_bl_ratio'),

    ('features', 'dH_bulk', ''),
    # ('features', 'volume_pa', ''),
    ('features', 'bulk_oxid_state', ''),  # KEEP
    ('features', 'effective_ox_state', ''),
    # ('features', 'surf_area', ''),  # NO

    # ('features_pre_dft', 'active_o_metal_dist__pre', ''),  # <-------------------------------
    ('features_pre_dft', 'ir_o_mean__pre', ''), # KEEP
    ('features_pre_dft', 'ir_o_std__pre', ''),  # KEEP
    ('features_pre_dft', 'octa_vol__pre', ''),  # KEEP
    ]]

  # <-------------------------------

# +
# assert False

# +
kdict = [
    {
        "type": "gaussian",
        "dimension": "single",
        "width": 1.8,
        "scaling": 0.5,
        "scaling_bounds": ((0.0001, 10.),),
        }
    ]

GP_R = GP_Regression(
    kernel_list=kdict,
    regularization=0.01,
    optimize_hyperparameters=True,
    scale_data=False,
    )

# +
# df_data.columns.tolist()

# +
new_cols = []
for col_i in df_data.columns:
    if col_i[0] == "features_pre_dft":
        new_col_i = ("features", col_i[1], col_i[2])
    else:
        new_col_i = col_i
    new_cols.append(new_col_i)


# df_data.columns = new_cols
df_data.columns = pd.MultiIndex.from_tuples(new_cols)
# -

df_data

# +
data_dict_list = []
num_feat_cols = df_data.features.shape[1]
# for num_pca_i in range(1, num_feat_cols + 1, do_every_nth_pca_comp):
for num_pca_i in range(8 , num_feat_cols + 1, do_every_nth_pca_comp):

    if verbose:
        print("")
        print(40 * "*")
        print(num_pca_i)

    MA = ModelAgent(
        df_features_targets=df_data,
        Regression=GP_R,
        Regression_class=GP_Regression,
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
# 10 PCA components are ideal with an MAE of 0.1788

# 10 PCA components are ideal with an MAE of 0.1645

# 11 PCA components are ideal with an MAE of 0.158
# 11 PCA components are ideal with an MAE of 0.1594


# +
# 11 PCA components are ideal with an MAE of 0.1703
# 11 PCA components are ideal with an MAE of 0.171

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
    MA.cleanup_for_pickle()
# -

data_dict_out = {
    "df_models": df_models,
    "ModelAgent_Plot": MA_Plot,
    }

assert False

# Pickling data ###########################################
directory = os.path.join(root_dir, "out_data")
print(directory)
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "modelling_data_NEW_88.pickle"), "wb") as fle:
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

# + jupyter={"source_hidden": true} tags=[]
# df_data.features.columns.tolist()

# + jupyter={"source_hidden": true} tags=[]
# df_data.columns.tolist()

# features_pre_dft

# + jupyter={"source_hidden": true} tags=[]
# assert False

# + jupyter={"source_hidden": true} tags=[]
# df_data = df_data[[

#     # ('targets', 'g_o', ''),
#     ('targets', 'g_oh', ''),

#     ('data', 'stoich', ''),
#     ('data', 'job_id_o', ''),
#     ('data', 'job_id_oh', ''),
#     ('data', 'job_id_bare', ''),

#     # ('features', 'o', 'O_magmom'),
#     # ('features', 'o', 'Ir_magmom'),
#     # ('features', 'o', 'Ir*O_bader'),
#     # ('features', 'o', 'Ir_bader'),
#     # ('features', 'o', 'O_bader'),

#     ('features', 'o', 'active_o_metal_dist'),
#     ('features', 'o', 'angle_O_Ir_surf_norm'),
#     ('features', 'o', 'ir_o_mean'),
#     ('features', 'o', 'ir_o_std'),
#     ('features', 'o', 'octa_vol'),
#     ('features', 'o', 'p_band_center'),
#     ('features', 'o', 'Ir*O_bader/ir_o_mean'),
#     ('features', 'dH_bulk', ''),
#     ('features', 'volume_pa', ''),
#     ('features', 'bulk_oxid_state', ''),
#     ('features', 'effective_ox_state', ''),
#     # ('features', 'surf_area', ''),

#     # ('features_pre_dft', 'active_o_metal_dist__pre', ''),
#     # ('features_pre_dft', 'ir_o_mean__pre', ''),
#     # ('features_pre_dft', 'ir_o_std__pre', ''),
#     # ('features_pre_dft', 'octa_vol__pre', ''),

#     ]]
