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

# +
feat_ads = "oh"

df_i = df_i["features_stan"][feat_ads]
# -

df_i.head()

# +
import plotly.express as px
fig = px.scatter_matrix(df_i)

print("show_plot:", show_plot)
if show_plot:
    fig.show()

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    save_dir=os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_vs_features",
        ),
    plot_name="scatter_matrix_plot",
    write_html=True,
    # write_png=False,
    # png_scale=6.0,
    # write_pdf=False,
    # write_svg=False,
    # try_orca_write=False,
    # verbose=False,
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

# + jupyter={"source_hidden": true}
# def method(row_i):
#     # #########################################################
#     name_i = row_i.name
#     # print(name_i)
#     # #########################################################
#     compenv_i = name_i[0]
#     slab_id_i = name_i[1]
#     ads_i = name_i[2]
#     active_site_i = name_i[3]
#     att_num_i = name_i[4]
#     # #########################################################
    
#     job_id_o_i = row_i.data.job_id_o
#     # print(job_id_o_i)

#     if type(job_id_o_i) != str:
#         if np.isnan(job_id_o_i):
#             job_id_o_i = "NaN"

#     name_i = job_id_o_i + "__" + str(int(active_site_i)).zfill(3)
    
#     return(name_i)

# df_i["data", "name_str"] = df_i.apply(
#     method,
#     axis=1)

# df_i = df_i.reindex(columns=[
#     "target_cols", "data", "features", "features_stan", ], level=0)

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_i = df_i.drop(columns=["data", "features", ])

# df_i.columns = df_i.columns.droplevel()

# + jupyter={"source_hidden": true}
# Script Inputs

# verbose = True
# verbose = False
