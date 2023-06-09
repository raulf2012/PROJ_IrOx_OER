# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

import plotly.express as px
import plotly.graph_objects as go

from IPython.display import display

# #########################################################
from methods import get_df_slab
from methods import get_df_features_targets

from methods_models import pca_analysis
# -

from proj_data import stoich_color_dict

# +
sys.path.insert(0, 
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/model_building"))

from methods_model_building import (
    simplify_df_features_targets,
    run_kfold_cv_wf,
    process_feature_targets_df,
    process_pca_analysis,
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

# target_ads_i = "oh"
# feature_ads_i = "oh"
# + active=""
#
#
#
#

# +
# df_j = simplify_df_features_targets(
#     df_i,
#     target_ads="o",
#     feature_ads="oh",
#     )

# df_format = df_features_targets[("format", "color", "stoich", )]
# -

df_2 = df_features_targets[[
    # ('targets', 'g_o', ''),
    ('targets', 'g_oh', ''),
    ('data', 'job_id_o', ''),
    ('data', 'job_id_oh', ''),
    ('data', 'job_id_bare', ''),
    ('data', 'stoich', ''),

    ('features', 'o', 'active_o_metal_dist'),
    ('features', 'o', 'ir_o_mean'),
    ('features', 'o', 'octa_vol'),
    ('features', 'o', 'oxy_opp_as_bl'),
    ('features', 'o', 'degrees_off_of_straight__as_opp'),
    ('features', 'dH_bulk', ''),
    ('features', 'bulk_oxid_state', ''),
    ('features', 'effective_ox_state', ''),

    # ('features_pre_dft', 'active_o_metal_dist__pre', ''),
    ('features_pre_dft', 'ir_o_mean__pre', ''),
    ('features_pre_dft', 'ir_o_std__pre', ''),
    ('features_pre_dft', 'octa_vol__pre', ''),
    ]]

# +
new_cols = []
for col_i in df_2.columns:
    # print(20*"-")
    # print(col_i)

    col_new_i = None

    if col_i[0] == "features_pre_dft":
        col_new_i = ("features", col_i[1], )
    elif col_i[0] == "features":
        if col_i[1] in ["o", "oh", ]:
            col_new_i = ("features", col_i[2])
        else:
            col_new_i = ("features", col_i[1])
            

    elif col_i[0] == "data":
        col_new_i = ("data", col_i[1], )
    elif col_i[0] == "targets":
        col_new_i = ("targets", col_i[1], )
    else:
        col_new_i = col_i

    # print(col_new_i)
    new_cols.append(col_new_i)

# [print(i) for i in new_cols]

idx = pd.MultiIndex.from_tuples(new_cols)
df_2.columns = idx
# df_2

df_j = df_2

df_j = df_j.dropna()
# -

num_pca_comp = 3

out_dict = process_pca_analysis(
    df_features_targets=df_j,
    num_pca_comp=num_pca_comp,
    )
df_pca = out_dict["df_pca_train"]
pca = out_dict["PCA"]

# +
df_features_targets.columns.tolist()

df_data = df_features_targets[[
    # ('data', 'phase', ''),
    ('data', 'stoich', ''),
    ]]

new_cols = []
for col_i in df_data.columns:
    new_col_i = (col_i[0], col_i[1], )
    new_cols.append(new_col_i)
idx = pd.MultiIndex.from_tuples(new_cols)
df_data.columns = idx
# -

df_pca = pd.concat([
    df_pca,
    df_data,
    ], axis=1)

df_pca = df_pca.droplevel(0, axis=1)

# +
df_pca = pd.concat([
    df_features_targets["data"]["overpot"],
    df_pca,
    ], axis=1)

df_pca = df_pca.dropna()
# -

df_pca["stoich__color"] = df_pca.stoich.map(stoich_color_dict)

# +
fig = px.scatter_3d(
    df_pca,

    x='PCA0',
    y='PCA1',
    z='PCA2',

    color='g_oh',
    )

fig.show()
# -

fig.layout

assert False

# +
# TOP SYSTEMS

# df_features_targets = df_features_targets.loc[

top_indices = [
    ('slac', 'hobukuno_29', 16.0),
    ('sherlock', 'ramufalu_44', 56.0),
    ('slac', 'nifupidu_92', 32.0),
    ('sherlock', 'bihetofu_24', 36.0),
    # ('slac', 'dotivela_46', 32.0),
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
    # ('sherlock', 'vipikema_98', 60.0),
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

    color='g_oh',
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
        # color=df_pca_3["overpot"],
        color=df_pca_3["stoich__color"],
        # df_pca_3.stoich

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

assert False

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

# + active=""
#
#
#

# + jupyter={"source_hidden": true} tags=[]
# df_dist = df_dist.sort_values("dist_from_mean", ascending=False)

# df_dist

# + jupyter={"source_hidden": true} tags=[]

# pd.concat([
#     df_features_targets.loc[[
#         ("slac", "dotivela_46", 26.0, )
#         ]],

#     df_features_targets.loc[
#         df_pca_4.sort_values("dist_from_mean", ascending=False).iloc[0:2].index.tolist()
#         ],
#     df_features_targets.sample(n=10),


#     ], axis=0)
