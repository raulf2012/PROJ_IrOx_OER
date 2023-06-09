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

# # Plotting scatter plot matrix features
# ---
#

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import numpy as np
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

from proj_data import layout_shared
from proj_data import stoich_color_dict, scatter_shared_props, shared_axis_dict
from proj_data import font_axis_title_size__pub, font_tick_labels_size__pub

# # Read Data

# +
from methods import get_df_features_targets
df_features_targets = get_df_features_targets()

from methods import get_df_slab
df_slab = get_df_slab()


# #########################################################
df_i = df_features_targets

# Getting phase > 1 slab ids
df_slab_i = df_slab[df_slab.phase > 1]
phase_2_slab_ids = df_slab_i.slab_id.tolist()
# -

# # Dropping phase 1 slabs

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

df_i = df_i[[
    ('features', 'o', 'active_o_metal_dist'),
    ('features', 'o', 'ir_o_mean'),
    ('features', 'o', 'octa_vol'),
    # ('features', 'o', 'oxy_opp_as_bl'),
    # ('features', 'o', 'degrees_off_of_straight__as_opp'),
    # ('features', 'dH_bulk', ''),
    # ('features', 'bulk_oxid_state', ''),
    # ('features', 'effective_ox_state', ''),
    # ('features_pre_dft', 'active_o_metal_dist__pre', ''),
    # ('features_pre_dft', 'ir_o_mean__pre', ''),
    # ('features_pre_dft', 'ir_o_std__pre', ''),
    # ('features_pre_dft', 'octa_vol__pre', ''),
    ]]

# + tags=[]
new_cols = []
for col_i in df_i.columns:
    col_new_i = [i for i in list(col_i) if i != ""][-1]
    new_cols.append(col_new_i)

df_i.columns = new_cols

# +
import plotly.express as px
fig = px.scatter_matrix(df_i)


# fig.layout.height = 2500
# fig.layout.width = 2500
fig.layout.height = 800
fig.layout.width = 800

# fig.update_xaxes(patch=shared_axis_dict)
# fig.update_yaxes(patch=shared_axis_dict)

tmp = 42

# +
# # px.scatter_matrix?
# -

fig

# +
# print("show_plot:", show_plot)
# if show_plot:
#     fig.show()

# +
# fig.update_layout(dict1=layout_shared, overwrite=True)
# fig.show()

# +
# # fig.update_xaxes?

# +
tmp_int = 2
for i in range(tmp_int):
    for j in range(tmp_int):
        fig.update_xaxes(

            {
                "xaxis" + str(i + 1): dict(
                    # tickangle=-45,
                    showline=True,
                    showgrid=False,
                    mirror=True,
                    linecolor="red",
                    ),
                },

            row=i,
            col=j,
            )
        # fig.update_layout(
        #     )

fig.show()
# -

fig.layout

# +
# fig
# -

shared_axis_dict

# +
# fig = px.scatter_matrix(px.data.tips())
# fig.update_layout({"xaxis"+str(i+1): dict(tickangle = -45) for i in range(2)})
# fig.show()

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    save_dir=os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering/analyse_features/scatter_plot_matrix",
        ),
    write_pdf=True,
    plot_name="scatter_matrix_plot",
    write_html=True,
    )
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("scatter_plot_matrix.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
# -

fig.layout

fig.data
