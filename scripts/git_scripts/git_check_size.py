# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_IrOx_Active_Learning_OER]
#     language: python
#     name: conda-env-PROJ_IrOx_Active_Learning_OER-py
# ---

# +
import os
print(os.getcwd())
import sys

import pandas as pd
# -

os.chdir(
    os.environ["PROJ_irox"]
    )

# +
import subprocess
res = subprocess.check_output(
    ["git", "ls-files", "--others", "--exclude-standard"]
    )

data_dict_list = []
for path_i in res.splitlines():
    data_dict_i = dict()

    path_i = path_i.decode('UTF-8')
    data_dict_i["path"] = path_i

    file_name_i = path_i.split("/")[-1]
    data_dict_i["file"] = file_name_i

    size_i = os.stat(path_i).st_size / 1000  # Size in KB
    data_dict_i["size_kb"] = size_i
    
    data_dict_list.append(data_dict_i)
    
df = pd.DataFrame(data_dict_list)

# +
df["size_mb"] = df.size_kb / 1000


df = df.sort_values("size_kb", ascending=False)

df.iloc[0:20]
# -

df.shape
