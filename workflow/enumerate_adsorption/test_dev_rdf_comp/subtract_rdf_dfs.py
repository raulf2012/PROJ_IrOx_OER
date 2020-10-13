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

# # Import Modules

# +
import sys

import numpy as np

import pandas as pd

sys.path.insert(0, "..")
from local_methods import compare_rdf_ij
# -

# #########################################################
import pickle; import os
path_i = os.path.join(
    os.environ["HOME"],
    "__temp__",
    "temp_2.pickle")
with open(path_i, "rb") as fle:
    df_rdf_i, df_rdf_j = pickle.load(fle)
# #########################################################

def create_interp_df(df_i, x_combined):
    """
    """
    r_combined = x_combined

    # df_i = df_rdf_j

    tmp_list = []
    data_dict_list = []
    for r_i in r_combined:
        # print("r_i:", r_i)
        data_dict_i = dict()

        # #################################################
        min_r = df_i.r.min()
        max_r = df_i.r.max()
        # #################################################

        if r_i in df_i.r.tolist():
            row_i = df_i[df_i.r == r_i].iloc[0]
            g_new = row_i.g

        else:
            # print(r_i)
            # tmp_list.append(r_i)

            if (r_i < min_r) or (r_i > max_r):
                g_new = 0.
            else:
                # break

                from scipy.interpolate import interp1d

                inter_fun = interp1d(
                    df_i.r, df_i.g,
                    kind='linear',
                    axis=-1,
                    copy=True,
                    bounds_error=None,
                    # fill_value=None,
                    assume_sorted=False,
                    )


                g_new = inter_fun(r_i)

        data_dict_i["r"] = r_i
        data_dict_i["g"] = g_new
        data_dict_list.append(data_dict_i)

    df_tmp = pd.DataFrame(data_dict_list)

    return(df_tmp)

# +
r_combined = np.sort((df_rdf_j.r.tolist() + df_rdf_i.r.tolist()))
r_combined = np.sort(list(set(r_combined)))


df_interp_i = create_interp_df(df_rdf_i, r_combined)
df_interp_j = create_interp_df(df_rdf_j, r_combined)

compare_rdf_ij(
    df_rdf_i=df_interp_i,
    df_rdf_j=df_interp_j)

# + active=""
#
#
#
#
#

# + jupyter={"source_hidden": true}
# # df_rdf_i.head()
# # df_rdf_j.head()
# print(df_rdf_j.shape[0])
# print(df_rdf_i.shape[0])

# + jupyter={"source_hidden": true}
# len(tmp_list)

# len(r_combined)

# for i in r_combined:
#     print(i)

# + jupyter={"source_hidden": true}
# for i_cnt, row_i in df_rdf_i.iterrows():
#     tmp = 42

# row_i.r

# + jupyter={"source_hidden": true}
# import plotly.graph_objs as go
# trace = go.Scatter(
#     x=df_tmp.r,
#     y=df_tmp.g,
#     )
# data = [trace]

# fig = go.Figure(data=data)
# fig.show()

# + jupyter={"source_hidden": true}
# # r_i
# # min_r

# id_min = df_i[df_i.r > r_i].r.idxmin()
# row_i = df_i.loc[id_min]

# r_1 = row_i.r
# g_1 = row_i.g

# + jupyter={"source_hidden": true}
# id_max = df_i[df_i.r < r_i].r.idxmax()

# row_i = df_i.loc[id_max]

# r_0 = row_i.r
# g_0 = row_i.g



# + jupyter={"source_hidden": true}
# r_i = 0.7222222222222222

# dr = 0.2

# # #########################################################


