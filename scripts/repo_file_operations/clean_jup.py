# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox] *
#     language: python
#     name: conda-env-PROJ_irox-py
# ---

# +
import os
print(os.getcwd())
import sys

from json import dump, load
from shutil import copyfile

# #########################################################
from methods import clean_ipynb
from methods import get_ipynb_notebook_paths

# +
cwd = os.getcwd()

find_str = "PROJ_IrOx_Active_Learning_OER/"
find_int = cwd.find(find_str)

PROJ_irox = cwd[:find_int] + find_str
# -

dirs_list = get_ipynb_notebook_paths(PROJ_irox_path=PROJ_irox)
for file_i in dirs_list:
    tmp = 42
    clean_ipynb(file_i, True)
