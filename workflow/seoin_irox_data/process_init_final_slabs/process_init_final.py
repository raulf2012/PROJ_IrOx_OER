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

# # Parse init and final slabs for magmoms
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

import pandas as pd

# +
# #########################################################
data_dict_list = []
# #########################################################
root_dir = os.path.join(
    os.environ["PROJ_DATA"],
    "PROJ_IrOx_OER/initial_final_structures")
# #########################################################
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if ".traj" in file:

            last_dir = subdir.split("/")[-1]
            init_or_final = last_dir.split("_")[-1]

            # print(
            #     file.split("_")
            #     )

            crystal_i = file.split("_")[0]
            facet_i = file.split("_")[1]
            coverage_i = file.split("_")[2]
            # print(tmp)

            frag_i = coverage_i + "_covered"

            find_ind = file.find(frag_i)

            file_2 = file[find_ind + len(frag_i):]

            file_3 = file_2.split(".")[0]

            print(file_3)

            # active_site_i = file.split(".")[0].split("_")[-1]
            # termination_i = file.split(".")[0].split("_")[-2]


            # #############################################
            data_dict_i = dict()
            # #############################################
            data_dict_i["crystal"] = crystal_i
            data_dict_i["facet"] = facet_i
            data_dict_i["coverage_i"] = coverage_i

            # data_dict_i["active_site"] = active_site_i
            # data_dict_i["termination"] = termination_i

            data_dict_i["init_final"] = init_or_final
            data_dict_i["file_name"] = file
            # #############################################
            data_dict_list.append(data_dict_i)
            # #############################################

# #########################################################
df = pd.DataFrame(
    data_dict_list
    )
# #########################################################
# -



# +
# termination	active_site	
# -

df

['rutile', '012', 'O', 'covered', '0', 'OER', 'OOH', '3', '0.traj']
['columbite', '121', 'O', 'covered', '1', 'OER', 'OH', '5', '1.traj']
['columbite', '120', 'O', 'covered', '0', 'OER', 'OOH', '0', '0.traj']
['pyrite', '100', 'O', 'covered', '0', 'OER', 'OOH', '0', '0.traj']
['anatase', '110', 'O', 'covered', '0', 'OER', 'OH', '3', '2.traj']
['columbite', '121', 'O', 'covered', '3', 'OER', 'OH', '1', '1.traj']
['pyrite', '120', 'O', 'covered', '0', 'OER', 'O', 'covered', '0.traj']
['columbite', '121', 'O', 'covered', '1', 'OER', 'OH', '1', '0.traj']
['brookite', '120', 'O', 'covered', '1', 'OER', 'OH', '1', '1.traj']
