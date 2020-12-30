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
#     display_name: Python [conda env:PROJ_irox]
#     language: python
#     name: conda-env-PROJ_irox-py
# ---

# # Clean jupyter notebooks (remove output)
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

from pathlib import Path
from json import dump, load
from shutil import copyfile

import plotly.graph_objs as go
import pandas as pd

# #########################################################
from jupyter_modules.jupyter_methods import (
    clean_ipynb,
    get_ipynb_notebook_paths,
    )

from jupyter_modules.jupyter_methods import get_df_jupyter_notebooks
# -

# ### Script Inputs

dry_run = True

# +
PROJ_irox_path = os.environ["PROJ_irox_oer"]
df = get_df_jupyter_notebooks(path=PROJ_irox_path)

# Removing this notebook from dataframe
df = df[df.file_name != "clean_jup.ipynb"]

# Removing notebooks that are in not relevent places
df = df[df.in_bad_place == False]
# -

df.head()

# +
# assert False
# -

# # Listing notebooks without paired python script

# +
df_tmp = df.sort_values("file_path")
df_tmp[df_tmp.py_file_present == False].style.set_properties(**{"text-align": "left"})

df_i = df_tmp[df_tmp.py_file_present == False]

print(
    "Number of jupyter notebooks without paired .py file:",
    "\n",
    df_i.shape[0]
    )

tmp = df_i.file_path_short.tolist()
tmp1 = [print(i) for i in tmp]
# -

# # Cleaning notebooks larger than 0.1 MB in size

# +
# #########################################################
# clean_notebooks = True
# #########################################################

df_big = df[df.file_size__mb > 0.04]

print("Number of notebooks to clean:", df_big.shape[0])

for ind_i, row_i in df_big.iterrows():
    file_path_i = row_i.file_path
 
    print("file_path_i:", file_path_i)
    # if clean_notebooks:
    if not dry_run:
        tmp = 42
        # clean_ipynb(file_path_i, True)
# -

# # Plotting file size, ordered high to low

# +
# #########################################################
layout = go.Layout(
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(text="NaN"),
        ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(text="File Size (MB)"),
        ),
    )

# #########################################################
trace = go.Scatter(
    x=df.index.tolist(),
    y=df.file_size__mb.tolist(),
    )
data = [trace]

fig = go.Figure(data=data, layout=layout)
# fig.show()

# + active=""
#
#
#
# -

df

# + jupyter={"source_hidden": true}
# import chart_studio.plotly as py
# import plotly.graph_objs as go

# import os

# x_array = [0, 1, 2, 3]
# y_array = [0, 1, 2, 3]


# trace = go.Scatter(
#     x=x_array,
#     y=y_array,
#     mode="markers",
#     opacity=0.8,
#     marker=dict(

#         symbol="circle",
#         color='LightSkyBlue',

#         opacity=0.8,

#         # color=z,
#         colorscale='Viridis',
#         colorbar=dict(thickness=20),

#         size=20,
#         line=dict(
#             color='MediumPurple',
#             width=2
#             )
#         ),

#     line=dict(
#         color="firebrick",
#         width=2,
#         dash="dot",
#         ),

#     error_y={
#         "type": 'data',
#         "array": [0.4, 0.9, 0.3, 1.1],
#         "visible": True,
#         },

#     )

# data = [trace]

# fig = go.Figure(data=data)
# fig.show()

# + jupyter={"source_hidden": true}
# # #########################################################
# data_dict_list = []
# # #########################################################
# dirs_list = get_ipynb_notebook_paths(PROJ_irox_path=PROJ_irox)
# for file_i in dirs_list:
#     data_dict_i = dict()

#     file_size_i = Path(file_i).stat().st_size
#     file_size_mb_i =  file_size_i / 1000 / 1000
#     file_name_i = file_i.split("/")[-1]
#     file_path_short_i = file_i[len(PROJ_irox) + 1:]

#     # #####################################################
#     in_bad_place = False
#     if ".virtual_documents" in file_i:
#         in_bad_place = True

#     # #####################################################
#     if "." in file_name_i:
#         ext_i = file_name_i.split(".")[-1]
#     else:
#         ext_i = "NaN"

#     # #####################################################
#     py_file_i = os.path.join(
#         "/".join(file_i.split("/")[0:-1]),
#         file_name_i.split(".")[0] + ".py"
#         )

#     my_file = Path(py_file_i)
#     if my_file.is_file():
#         py_file_present_i = True
#     else:
#         py_file_present_i = False


#     # #####################################################
#     data_dict_i["file_path"] = file_i
#     data_dict_i["file_path_short"] = file_path_short_i
#     data_dict_i["file_name"] = file_name_i
#     data_dict_i["file_ext"] = ext_i
#     data_dict_i["file_size__b"] = file_size_i
#     data_dict_i["file_size__mb"] = file_size_mb_i
#     data_dict_i["in_bad_place"] = in_bad_place
#     data_dict_i["py_file_present"] = py_file_present_i
#     # data_dict_i[""] = 
#     # #####################################################
#     data_dict_list.append(data_dict_i)
#     # #####################################################

# # #########################################################
# df = pd.DataFrame(data_dict_list)
# df = df.sort_values("file_size__b", ascending=False)
# df = df.reset_index(drop=True)

# # Removing this notebook from dataframe
# df = df[df.file_name != "clean_jup.ipynb"]

# # Removing notebooks that are in not relevent places
# df = df[df.in_bad_place == False]
# # #########################################################
