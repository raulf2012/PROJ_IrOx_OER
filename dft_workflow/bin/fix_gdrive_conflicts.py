# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Fix conflicts to GDrive and Dropbox (project folder) filesystem
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

from local_methods import remove_conflicted_path_dir
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# +
# remove_files_folders = False
remove_files_folders = True

if not remove_files_folders:
    print(
        10 * "Dry run, not actually removing stuff, just checking \n"
        )

# +
compenv = os.environ.get("COMPENV", "wsl")

if compenv == "wsl":
    jobs_root_dir = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        "dft_workflow")
elif compenv == "nersc" or compenv == "sherlock" or compenv == "slac":
    jobs_root_dir = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow")

# + active=""
#
#
#
# -

# ## Cleanup Dropbox project dirs
# ---

# + jupyter={"outputs_hidden": true}
for subdir, dirs, files in os.walk(os.environ["PROJ_irox_oer"]):
    for file_i in files:
        if "conflicted copy" in file_i:
            file_path_i = os.path.join(subdir, file_i)
            if remove_files_folders:
                print("Removing:", file_path_i)
                os.remove(file_path_i)

# +
# assert False
# -

# ## Cleanup GDrive dirs
# ---

# ### Remove conflicted files with " (" syntax

for subdir, dirs, files in os.walk(jobs_root_dir):
    for file_i in files:
        if " (" in file_i:
            file_path_i = os.path.join(subdir, file_i)

            if remove_files_folders:
                print("Removing:", file_path_i)
                os.remove(file_path_i)

# ### Remove conflicted files "[Conflict " syntax

for subdir, dirs, files in os.walk(jobs_root_dir):
    for file_i in files:
        if "[Conflict" in file_i:
            file_path_i = os.path.join(subdir, file_i)

            print("Removing:", file_path_i)
            if remove_files_folders:
                os.remove(file_path_i)

# ### Remove conflicted folders

for subdir, dirs, files in os.walk(jobs_root_dir):
    remove_conflicted_path_dir(path=subdir, remove_files_folders=remove_files_folders)

# + active=""
#
#
#
