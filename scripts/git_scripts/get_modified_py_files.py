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

# # Import Modules

# +
import os
print(os.getcwd())
import sys


import subprocess

import pandas as pd
# -

os.chdir(
    os.environ["PROJ_irox_oer"]
    )

# +
# res = subprocess.check_output(
#     "git rev-parse --show-toplevel".split(" ")
#     )
# root_git_path = res.decode("UTF-8").strip()

# +
res = subprocess.check_output(
    # ["git", "ls-files", "--others", "--exclude-standard"]
    "git status -s --porcelain".split(" ")
    )

data_dict_list = []
out_list = [i.decode("UTF-8") for i in res.splitlines()]
for line_i in out_list:
    data_dict_i = dict()

    if line_i[0] == " ":
        line_i = line_i[1:]

    status_i = line_i.split(" ")[0]
    path_i = line_i.split(" ")[1]

    data_dict_i["status"] = status_i
    data_dict_i["path"] = path_i

    filename_i = path_i.split("/")[-1]
    data_dict_i["filename"] = filename_i

    if len(line_i.split(" ")) > 2:
        print("More than 2 columns here, problem")        
        break
        
    data_dict_list.append(data_dict_i)

df = pd.DataFrame(data_dict_list)

df.iloc[15:35]
# -

df

# + active=""
#
#
#
#
#
# -

assert False

# +
sys.path.insert(0,
    os.path.join(
        os.environ["PROJ_irox"],
        "scripts/repo_file_operations"))
from methods import get_ipynb_notebook_paths

ipynb_files_list = get_ipynb_notebook_paths(relative_path=True)
# ipynb_files_list

# +
df = df[df.path.isin(ipynb_files_list)]
df = df[df.status == "M"]

df["path_python"] = [i.replace(".ipynb", ".py") for i in df.path.tolist()]

" ".join(df.path_python.tolist())

# +
" ".join(df.path.tolist())

# df.path
