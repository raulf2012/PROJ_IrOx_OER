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

# # Overview of jupyter notebooks in project
# ---
# I will compile list of notebooks and check against the script that runs all notebooks. I want to make sure that everything is being run and no notebook is being neglected.

# ### Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys


from IPython.display import display

import pandas as  pd
pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
pd.options.display.max_colwidth = 100

# #########################################################
from jupyter_modules.jupyter_methods import get_df_jupyter_notebooks

# #########################################################
from local_methods import get_notebooks_run_in__jobs_update
from local_methods import get_files_exec_in_run_all

# -

# ### Script Inputs

# +
black_listed_dirs = [
    "sandbox",
    ".virtual_documents",
    ]


# These notebooks are not intended to be run periodically, they are one-time use
notebooks_to_ignore = [

    # Trivial model copies
    "workflow/model_building/predict_mean_of_pop/trivial_models-Copy1.ipynb",
    "workflow/model_building/predict_mean_of_pop/trivial_models-Copy2.ipynb",
    "workflow/model_building/predict_mean_of_pop/trivial_models-Copy3.ipynb",

    "workflow/seoin_irox_data/analyze_dataset/trivial_eff_ox_state_model.ipynb",
    "workflow/seoin_irox_data/compare_mine_seoin_oer/compare_mine_seoin.ipynb",
    "workflow/seoin_irox_data/featurize_data/featurize_data.ipynb",
    "workflow/seoin_irox_data/match_si_and_seoin_data.ipynb",
    "workflow/seoin_irox_data/plot_data/eff_ox_vs_oer.ipynb",
    "workflow/seoin_irox_data/process_SI_oer_table.ipynb",
    "workflow/seoin_irox_data/process_bulk_data/process_bulk.ipynb",
    "workflow/seoin_irox_data/process_init_final_slabs/process_init_final.ipynb",
    "workflow/seoin_irox_data/read_seoin_irox_data.ipynb",

    # Only needs to be run manually, will set in motion many downstream jobs
    "dft_workflow/run_slabs/run_o_covered/setup_dft.ipynb",

    # #####################################################
    # Slab creation notebooks
    "workflow/creating_slabs/create_slabs.ipynb",
    "workflow/creating_slabs/creating_symm_slabs/creat_symm_slabs.ipynb",
    "workflow/creating_slabs/modify_df_slab.ipynb",
    "workflow/creating_slabs/process_slabs.ipynb",
    "workflow/creating_slabs/quality_control/slab_size_xy_rep.ipynb",
    "workflow/creating_slabs/selecting_bulks/select_bulks.ipynb",
    "workflow/creating_slabs/slab_similarity/slab_similarity.ipynb",
    "workflow/enumerate_adsorption/get_all_active_sites.ipynb",
    "workflow/enumerate_adsorption/modify_df_active_sites.ipynb",
    "workflow/enumerate_adsorption/test_dev_rdf_comp/opt_rdf_comp.ipynb",
    "workflow/enumerate_adsorption/test_dev_rdf_comp/subtract_rdf_dfs.ipynb",


    # #####################################################
    # Some notebooks in job_analysis
    "dft_workflow/job_analysis/id_resource_waste/id_resource_waste.ipynb",
    "dft_workflow/job_analysis/measure_comp_resources/measure_cpu_hours.ipynb",
    "dft_workflow/job_analysis/prepare_oer_sets/prepare_oer_sets.ipynb",


    # #####################################################
    # Some notebooks in `compare_magmoms` aren't that useful anymore
    "dft_workflow/job_analysis/compare_magmoms/anal_magmoms.ipynb",
    "dft_workflow/job_analysis/compare_magmoms/new_O_bare_magmom_comp/comp_new_O_bare_magmoms.ipynb",


    # #####################################################
    # DFT workflow bin scripts that are 'on occasion' use  only
    "dft_workflow/bin/anal_job_out.ipynb",
    "dft_workflow/bin/delete_unsub_jobs.ipynb",
    "dft_workflow/bin/run_unsub_jobs.ipynb",


    # #####################################################
    # One-time bulk processing scripts
    "workflow/process_bulk_dft/create_final_df_dft.ipynb",
    "workflow/process_bulk_dft/get_bulk_coor_env.ipynb",
    "workflow/process_bulk_dft/manually_classify_bulks/classify_bulks.ipynb",
    "workflow/process_bulk_dft/read_json_to_new_ase.ipynb",
    "workflow/process_bulk_dft/standardize_bulks/stan_bulks.ipynb",
    "workflow/process_bulk_dft/write_atoms_json.ipynb",

    "workflow/xrd_bulks/plot_xrd_patterns/plot_xrd_patterns.ipynb",
    "workflow/xrd_bulks/xrd_bulks.ipynb",


    # #####################################################
    # These weren't too fruitful so no need to rerun
    "workflow/oer_vs_features/bivariate_combs_feat/bivariate_combs.ipynb",
    "workflow/oer_vs_features/feature_covariance/feat_covar.ipynb",


    # #####################################################
    # Most scripts in scripts/ are meant to be run on a one-off basis
    "scripts/git_scripts/get_modified_py_files.ipynb",
    "scripts/git_scripts/git_check_size.ipynb",
    "scripts/repo_file_operations/clean_jup.ipynb",
    "scripts/repo_file_operations/jup_notebook_overview.ipynb",
    "scripts/repo_file_operations/find_files.ipynb",


    # #####################################################
    # Don't run these things
    "workflow/__misc__/analysis_for_jens/analysis.ipynb",
    "workflow/feature_engineering/generate_features/catkit_form_oxid/catkit_form_oxid.ipynb",


    # #####################################################
    # MISC
    "__misc__/00_group_meeting/group_meeting.ipynb",
    "collated_plots.ipynb",
    "dft_workflow/run_slabs/setup_jobs_from_oh/id_new_O_with_orig_OH/new_O_orig_OH.ipynb",


    ]


# +
TEMP_FILES_TO_IGNORE = [
    "dft_workflow/__misc__/finding_nonconstrained_mistakes/find_unconstr_slabs.ipynb",
    "dft_workflow/job_analysis/slab_struct_drift/slab_struct_drift.ipynb",
    "dft_workflow/job_analysis/systems_fully_ready/sys_fully_ready.ipynb",
    "dft_workflow/job_processing/clean_dft_dirs.ipynb",

    # "workflow/oer_analysis/oer_scaling/oer_scaling.ipynb",
    ]

notebooks_to_ignore.extend(TEMP_FILES_TO_IGNORE)
# -

# ### Read Data

# +
# Read jupyter notebook dataframe
PROJ_irox_path = os.environ["PROJ_irox_oer"]

# Get datafame of all jupyter notebooks in project
df_ipynb = get_df_jupyter_notebooks(path=PROJ_irox_path)
df_ipynb_i = df_ipynb

# Read bash update method (calls all notebooks)
df_ipynb_update = get_notebooks_run_in__jobs_update()

df_run_all = get_files_exec_in_run_all()

# + active=""
#
#
# -

# ## Printing notebooks that aren't being run in: `PROJ_irox_oer__comm_overnight`

print(
    "\n",
    "Notebooks that aren't being run in PROJ_irox_oer__comm_overnight:",

    "\n",
    65 * "-",
    sep="")
# #########################################################
for index_i, row_i in df_ipynb_i.iterrows():
    # #####################################################
    file_path_short_i = row_i.file_path_short
    file_name_i = row_i.file_name
    # #####################################################

    # Skipping certain rows
    # if "sandbox" in file_name_i:
    if "sandbox" in file_path_short_i:
        continue
    if "old." in file_path_short_i:
        continue
    if file_path_short_i in notebooks_to_ignore:
        continue

    black_listed = False
    for dir_i in black_listed_dirs:
        if file_path_short_i.find(dir_i) == 0:
            black_listed = True
    if black_listed:
        continue


    # #####################################################
    df_i = df_run_all[df_run_all.path_short == file_path_short_i]
    if df_i.shape[0] == 0:
        print(file_path_short_i)

# + active=""
# dft_workflow/job_analysis/prepare_oer_sets/write_oer_sets.ipynb
#
# -

df_ipynb_i.head()

# + active=""
#
#
#
