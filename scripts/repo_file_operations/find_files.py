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

# # Find files within project directories
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

import pandas as pd

# #########################################################

# +
# #########################################################
data_dict_list = []
# #########################################################
root_dir = os.path.join(os.environ["PROJ_irox_oer"])
for subdir, dirs, files in os.walk(root_dir):
    find_ind = subdir.find(root_dir)
    for file in files:

        # Getting file extension
        ext_i = None
        if "." in file:
            ext_i = file.split(".")[-1]

        # Get relative path
        subdir_rel = None
        if find_ind != -1:
            subdir_rel = subdir[
                find_ind + len(root_dir) + 1:]
        path_rel_i = os.path.join(subdir_rel, file)



        # #################################################
        data_dict_i = dict()
        # #################################################
        data_dict_i["file_ext"] = ext_i
        data_dict_i["file_name"] = file
        data_dict_i["path_rel"] = subdir_rel
        data_dict_i["file_path_rel"] = path_rel_i
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

# #########################################################
df_files = pd.DataFrame(data_dict_list)
# #########################################################

# +
df_i = df_files[df_files.file_ext == "pickle"]

df_i = df_i[df_i.file_name.str.contains("df_")]

df_i = df_i[~df_i.file_name.str.contains("old")]
df_i = df_i[~df_i.path_rel.str.contains("old")]

df_i = df_i.sort_values(["path_rel", "file_name"])
# -

dfStyler = df_i.style.set_properties(**{'text-align': 'left'})
dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

df_i[df_i.file_name.duplicated(keep=False)]

# df_i.to_string()
df_i.file_path_rel.tolist()

assert False

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # os.environ[]

# root_dir

# + jupyter={"source_hidden": true}
# find_ind = subdir.find(root_dir)

# subdir_rel = None
# if find_ind != -1:
#     tmp = 42

#     subdir_rel = subdir[
#         find_ind + len(root_dir) + 1:
#         ]

# + jupyter={"source_hidden": true}

# subdir.find("isdjfisd")

# + jupyter={"source_hidden": true}
# for ind_i, row_i in df_i.iterrows():
#     tmp = 42
# df_i['Names'].str.contains('Mel')
# row_i.path_rel

# + jupyter={"source_hidden": true}
# # Test data
# df = pd.DataFrame(
#     {
#         'text': ['fooooooooooo', 'bar'],
#         'number': [1, 2],
#         }
#     )

# dfStyler = df.style.set_properties(**{'text-align': 'left'})
# dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
