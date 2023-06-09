# -*- coding: utf-8 -*-
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

# + [markdown] tags=[]
# # Computing covariance matrix for features
# ---
#
# -

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import plotly.express as px

# #########################################################
from proj_data import layout_shared as layout_shared_main
from proj_data import scatter_shared_props as scatter_shared_props_main
from proj_data import (
    stoich_color_dict,
    shared_axis_dict,
    font_tick_labels_size,
    font_axis_title_size__pub,
    font_tick_labels_size__pub,
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

root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/analyse_features/feature_covariance")

# # Script Inputs

# +
target_ads = "o"
# target_ads = "oh"

verbose = True
# verbose = False
# -

# # Read Data

# +
from methods import get_df_features_targets
df_features_targets = get_df_features_targets()
df_i = df_features_targets

from methods import get_df_slab
df_slab = get_df_slab()

# Getting phase > 1 slab ids
df_slab_i = df_slab[df_slab.phase > 1]
phase_2_slab_ids = df_slab_i.slab_id.tolist()
# -

# ### Processing dataframe

# +
if target_ads == "o":
    other_ads = "oh"
elif target_ads == "oh":
    other_ads = "o"

# df_i = df_i.drop("features_stan", axis=1, level=0)

df_i = df_i.drop(other_ads, axis=1, level=1)
# -

# ### Dropping phase 1 slabs

# +
df_index = df_i.index.to_frame()
df_index_i = df_index[
    df_index.slab_id.isin(phase_2_slab_ids)
    ]

print("Dropping phase 1 slabs")
df_i = df_i.loc[
    df_index_i.index
    ]
# -

df_feat = df_i["features"][target_ads]

# ### Standardize columns

# +
df_features = df_feat

df_features_stan = copy.deepcopy(df_features)
for col_i in df_features_stan.columns:
    max_val = df_features_stan[col_i].max()
    mean_val = df_features_stan[col_i].mean()
    std_val = df_features_stan[col_i].std()
    df_features_stan[col_i] = (df_features_stan[col_i]) / max_val
df_feat_stan = df_features_stan

# + active=""
#
#
#
# -

list(df_feat_stan.columns)

# ### Renaming feature columns to be more readable 

# +
#     "bulk_oxid_state":       "F01",
#     "volume_pa":             "F02",
#     "dH_bulk":               "F03",
#     # e_2p  (bulk)
#     "magmom_active_site":    "F05",
#     # Bader
#     "effective_ox_state":    "F07",
#     "octa_vol":              "F08",
#     "angle_O_Ir_surf_norm":  "F09",
#     "ir_o_mean":             "F10",
#     "ir_o_std":              "F11",
#     "active_o_metal_dist":   "F12",

# +
# Sat Apr  3 14:06:38 PDT 2021

#     "bulk_oxid_state":       "F01",
#     "volume_pa":             "F02",
#     "dH_bulk":               "F03",
#     "p_band_center":         "F04",
#     "Ir_magmom":             "F05",
#     "O_magmom":              "F06",
#     "Ir_bader":              "F07",
#     "O_bader":               "F08",
#     "effective_ox_state":    "F09",
#     "octa_vol":              "F10",
#     "ir_o_mean":             "F11",
#     "angle_O_Ir_surf_norm":  "F12",
#     "ir_o_std":              "F13",
#     "active_o_metal_dist":   "F14",

# +
feature_rename_dict = {

    # "magmom_active_site":  "A.S. Magmom",
    # "active_o_metal_dist": "O Metal Dist.",
    # "effective_ox_state":  "Eff. Ox. State",
    # "ir_o_mean":           "Ir-O Mean Dist.",
    # "ir_o_std":            "Ir-O Dist σ",
    # "octa_vol":            "Octa. Vol.",
    # "dH_bulk":             "ΔH Bulk",
    # "volume_pa":           "Vol. P.A.",
    # "bulk_oxid_state":     "Ox. State Bulk",


    "bulk_oxid_state":       "F01",
    "volume_pa":             "F02",
    "dH_bulk":               "F03",
    "p_band_center":         "F04",
    "O_magmom":              "F05",
    "Ir_magmom":             "F06",
    "O_bader":               "F07",
    "Ir_bader":              "F08",
    "effective_ox_state":    "F09",
    "octa_vol":              "F10",
    "ir_o_mean":             "F11",
    "ir_o_std":              "F12",
    "angle_O_Ir_surf_norm":  "F13",
    "active_o_metal_dist":   "F14",

    }

new_cols = []
for col_i in df_feat_stan.columns:
    new_col_i = feature_rename_dict.get(col_i, col_i)
    new_cols.append(new_col_i)

df_feat_stan.columns = new_cols

from misc_modules.pandas_methods import reorder_df_columns
df_feat_stan = reorder_df_columns(
    sorted(list(df_feat_stan.columns)),
    df_feat_stan)

# +
df = df_feat_stan.corr()

df = df.sort_index(
    ascending=False,
    )

# +
cols_to_drop = []
for col_i in df.columns:
    if col_i not in list(feature_rename_dict.values()):
        print(col_i)
        cols_to_drop.append(col_i)

df = df.drop(columns=cols_to_drop)
df = df.drop(index=cols_to_drop)
# -

# ### Setting the diagonal and off-diagonal triangle to 0

# +
# #########################################################
for i in df.columns:
    df.xs(i)[i] = 0

# #########################################################
for i_cnt, i in enumerate(df.columns):
    for j_cnt, j in enumerate(df.columns):        
        # if j_cnt > i_cnt:
        if i_cnt > j_cnt:
            df.xs(i)[j] = 0
# -

max_abs_min_max = np.max(
    [
        np.abs(df.min().min()),
        np.abs(df.max().max()),
        ]
    )

# ### Renaming row, column names, remove the leading 0

# +
feature_rename_dict = {
    "F01": "F1",
    "F02": "F2",
    "F03": "F3",
    "F04": "F4",
    "F05": "F5",
    "F06": "F6",
    "F07": "F7",
    "F08": "F8",
    "F09": "F9",
    # "F10": "F10",
    # "F11": "F11",
    # "F12": "F12",
    }

# Renaming columns
new_cols = []
for col_i in df.columns:
    new_col_i = feature_rename_dict.get(col_i, col_i)
    new_cols.append(new_col_i)
df.columns = new_cols

# Renaming indices
new_inds = []
for index_i in df.index:
    new_ind_i = feature_rename_dict.get(index_i, index_i)
    new_inds.append(new_ind_i)
df.index = new_inds

# +
fig = px.imshow(
    df,
    # x=df.columns, y=df.columns,
    x=df.columns, y=list(reversed(df.columns)),
    title="Feaeture Correlation Matrix",

    # color_continuous_scale="Picnic",    # **
    # color_continuous_scale="aggrnyL",   # ***
    # color_continuous_scale="agsunset",  # ****
    # color_continuous_scale="armyrose",  # ****
    # color_continuous_scale="geyser",    # ********
    # color_continuous_scale="portland",  # ******

    color_continuous_scale="spectral",  # 


    # zmin=-1,
    # zmax=+1,

    zmin=-max_abs_min_max,
    zmax=+max_abs_min_max,

    )

# Modifying layout
fig.layout.title = None

# fig.layout.height = 900
fig.layout.height = 1100
# fig.layout.width = 500

fig.layout.update(dict1=dict(xaxis=shared_axis_dict))
fig.layout.update(dict1=dict(yaxis=shared_axis_dict))

# Color bar
colorbar_dict = dict(
    colorbar=dict(
        outlinecolor="black",
        outlinewidth=1,
        ticks="outside",
        tickvals=[-0.8, -0.4, 0, 0.4, 0.8],
        ticklen=8,

        tickfont=dict(
            size=shared_axis_dict["tickfont"]["size"],
            color="black",
            ),
        )
    )

fig.update_coloraxes(colorbar_dict, row=None, col=None)

if show_plot:
    fig.show()
# -

fig.write_json(
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering/analyse_features/feature_covariance",
        "out_plot/feature_correlation.json"))

# ### Creating publication version of figure

# +
# import copy
fig.update_layout(
    dict(
        height=450,
        width=450,

        xaxis=dict(tickfont=dict(size=font_tick_labels_size__pub)),
        yaxis=dict(tickfont=dict(size=font_tick_labels_size__pub)),
        )
    )

# Color bar
colorbar_dict = dict(
    colorbar=dict(
        tickfont=dict(
            size=font_tick_labels_size__pub,
            ),
        )
    )

fig.update_coloraxes(colorbar_dict, row=None, col=None)

if show_plot:
    fig.show()
# -

assert False

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    save_dir=root_dir,
    place_in_out_plot=True,
    plot_name="corr_matrix__pub",
    write_html=True,
    write_png=True,
    png_scale=6.0,
    write_pdf=True,
    write_svg=False,
    try_orca_write=True,
    verbose=False,
    )
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("feat_correlation.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# +
# fig.write_image("corr_matrix__pub" + ".png", scale=6)
