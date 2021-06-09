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

# # Constructing linear model for OER adsorption energies
# ---
#

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

# import copy

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

import plotly.express as px
import plotly.graph_objects as go

from IPython.display import display

# #########################################################
# from proj_data import scatter_marker_props, layout_shared

# #########################################################
# from local_methods import run_gp_workflow
# -

from methods import get_df_slab
from methods import get_df_features_targets

# +
sys.path.insert(0, 
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/model_building"
        )
    )

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

# ### Read Data

# +
df_features_targets = get_df_features_targets()

df_slab = get_df_slab()

# #########################################################
df_i = df_features_targets

# Getting phase > 1 slab ids
df_slab_i = df_slab[df_slab.phase > 1]
phase_2_slab_ids = df_slab_i.slab_id.tolist()
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

# + active=""
#
#
#
#
# -

# # -------------------------

# # G_O Model

# +
# target_ads_i = "o"

target_ads_i = "oh"
feature_ads_i = "oh"
# + active=""
#
#
#
#

# +
df_j = simplify_df_features_targets(
    df_i,
    target_ads="o",
    feature_ads="oh",
    )

# df_format = df_features_targets[("format", "color", "stoich", )]

# +
# print(
#     "cols_to_use:"
#     "\n",
#     20 * "-",
#     sep="")
# tmp = [print(i) for i in list(df_j["features"].columns)]

cols_to_use = list(df_j["features"].columns)
# -

cols_to_use = [
    'magmom_active_site',
    'active_o_metal_dist',
    'effective_ox_state',
    'ir_o_mean',
    'ir_o_std',
    'octa_vol',
    'dH_bulk',
    'volume_pa',
    'bulk_oxid_state',
    ]

num_pca_comp = 3

# +
df_j = process_feature_targets_df(
    df_features_targets=df_j,
    cols_to_use=cols_to_use,
    )

# #####################################################
# df_pca = process_pca_analysis(
out_dict = process_pca_analysis(
    df_features_targets=df_j,
    num_pca_comp=num_pca_comp,
    )
# #####################################################
df_pca = out_dict["df_pca"]
pca = out_dict["pca"]
# #####################################################
# -

df_pca = df_pca.droplevel(0, axis=1)

# +
df_pca = pd.concat([
    df_features_targets["data"]["overpot"],
    df_pca,
    ], axis=1)

df_pca = df_pca.dropna()

# +
# assert False

# +
fig = px.scatter_3d(
    df_pca,

    x='PCA0',
    y='PCA1',
    z='PCA2',

    color='y',
    )

# fig.show()

# +
# TOP SYSTEMS

# df_features_targets = df_features_targets.loc[

top_indices = [
    ('slac', 'hobukuno_29', 16.0),
    ('sherlock', 'ramufalu_44', 56.0),
    ('slac', 'nifupidu_92', 32.0),
    ('sherlock', 'bihetofu_24', 36.0),
    ('slac', 'dotivela_46', 32.0),
    ('slac', 'vovumota_03', 33.0),
    ('slac', 'ralutiwa_59', 32.0),
    ('sherlock', 'bebodira_65', 16.0),
    ('sherlock', 'soregawu_05', 62.0),
    ('slac', 'hivovaru_77', 26.0),
    ('sherlock', 'vegarebo_06', 50.0),
    ('slac', 'ralutiwa_59', 30.0),
    ('sherlock', 'kamevuse_75', 49.0),
    ('nersc', 'hesegula_40', 94.0),
    ('slac', 'fewirefe_11', 39.0),
    ('sherlock', 'vipikema_98', 60.0),
    ('slac', 'gulipita_22', 48.0),
    ('sherlock', 'rofetaso_24', 48.0),
    ('slac', 'runopeno_56', 32.0),
    ('slac', 'magiwuni_58', 26.0),
    ]

df_pca_2 = df_pca.loc[top_indices]
df_pca_3 = df_pca.drop(labels=top_indices)

# +
fig = px.scatter_3d(
    df_pca_2,

    x='PCA0',
    y='PCA1',
    z='PCA2',

    color='y',
    )

fig.show()

# +
trace_1 = go.Scatter3d(
    x=df_pca_2["PCA0"],
    y=df_pca_2["PCA1"],
    z=df_pca_2["PCA2"],
    mode='markers',
    marker_color="red",
    )
trace_2 = go.Scatter3d(
    x=df_pca_3["PCA0"],
    y=df_pca_3["PCA1"],
    z=df_pca_3["PCA2"],
    mode='markers',
    marker_color="gray",
    )

fig = go.Figure(
    data=[trace_1, trace_2]
    )

fig.show()

# +
trace_1 = go.Scatter3d(
    x=df_pca_2["PCA0"],
    y=df_pca_2["PCA1"],
    z=df_pca_2["PCA2"],
    mode='markers',
    marker_color="red",
    )
trace_2 = go.Scatter3d(
    x=df_pca_3["PCA0"],
    y=df_pca_3["PCA1"],
    z=df_pca_3["PCA2"],
    mode='markers',
    
    marker=go.scatter3d.Marker(
        opacity=0.8,
        colorbar=dict(thickness=20, ),
        color=df_pca_3["overpot"],

        cmin=df_pca["overpot"].min(),
        cmax=0.5,
        reversescale=True,

        ),
    )
layout = go.Layout(
    height=500,
    scene=go.layout.Scene(
        xaxis=go.layout.scene.XAxis(title="PCA0"),
        yaxis=go.layout.scene.YAxis(title="PCA1"),
        zaxis=go.layout.scene.ZAxis(title="PCA2")
        ),
    )

fig = go.Figure(
    data=[
        # trace_1,
        trace_2,
        ],
    layout=layout,
    )

fig.show()
# -

df_pca_4.loc[[("slac", "dotivela_46", 26.0, )]]

# ### Ranking systems by proximity to PCA feature space center

# +
PCA0_mean = df_pca_3.PCA0.mean()
PCA1_mean = df_pca_3.PCA1.mean()
PCA2_mean = df_pca_3.PCA2.mean()

PCA_mean = np.array([
    PCA0_mean,
    PCA1_mean,
    PCA2_mean,
    ])

# +
data_dict_list = []
for index_i, row_i in df_pca_3.iterrows():
    # #####################################################
    index_dict_i = dict(zip(df_pca_3.index.names, index_i,))
    # #####################################################
    PCA_i = row_i[["PCA0", "PCA1", "PCA2", ]].to_numpy()
    dist_from_mean = np.linalg.norm(PCA_mean - PCA_i)

    data_dict_i = dict()
    data_dict_i.update(index_dict_i)
    data_dict_i["dist_from_mean"] = dist_from_mean
    data_dict_list.append(data_dict_i)

df_dist = pd.DataFrame(data_dict_list)

df_dist = df_dist.set_index(["compenv", "slab_id", "active_site", ])
# -

df_pca_4 = pd.concat([
    df_pca_3,
    df_dist,
    ], axis=1)

df_pca_4.sort_values("dist_from_mean", ascending=False).iloc[0:15]

# +

pd.concat([
    df_features_targets.loc[[
        ("slac", "dotivela_46", 26.0, )
        ]],

    df_features_targets.loc[
        df_pca_4.sort_values("dist_from_mean", ascending=False).iloc[0:2].index.tolist()
        ],
    df_features_targets.sample(n=10),


    ], axis=0)
# -



# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_dist = df_dist.sort_values("dist_from_mean", ascending=False)

# df_dist
