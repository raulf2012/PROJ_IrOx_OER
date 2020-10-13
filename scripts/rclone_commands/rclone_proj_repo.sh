#!/bin/bash

#| - General rclone command
rclone --transfers=2 sync \
$rclone_dropbox:01_norskov/00_git_repos/PROJ_IrOx_OER \
$wd/PROJ_IrOx_OER \
--exclude ".git/**" \
--exclude "__pycache__/**" \
--exclude ".ipynb_checkpoints/**" \
--exclude "__old__/**" \
--exclude "__temp__/**" \
--exclude "out_data/**" \
--exclude "out_data*/**" \
--exclude "out_plot/**" \
--exclude "out_plot*/**" \
--verbose \
#__|




PROJ_irox_oer_rc=01_norskov/00_git_repos/PROJ_IrOx_OER

# #########################################################
# IrOx OER slabs dataframe
rclone copy  \
$rclone_dropbox:$PROJ_irox_oer_rc/workflow/creating_slabs/out_data/df_slab_final.pickle \
$wd/PROJ_IrOx_OER/workflow/creating_slabs/out_data \
-v


# #########################################################
#| - DFT settings json file
rclone copy  \
$rclone_dropbox:$PROJ_irox_oer_rc/dft_workflow/dft_scripts/out_data/dft_calc_settings.json \
$wd/PROJ_IrOx_OER/dft_workflow/dft_scripts/out_data \
-v &

rclone copy  \
$rclone_dropbox:$PROJ_irox_oer_rc/dft_workflow/dft_scripts/out_data/easy_dft_calc_settings.json \
$wd/PROJ_IrOx_OER/dft_workflow/dft_scripts/out_data \
-v &
#__|


# #########################################################
# DFT jobs dataframe, needed to not run jobs from other clusters
rclone copy  \
$rclone_dropbox:$PROJ_irox_oer_rc/dft_workflow/job_processing/out_data/ \
$wd/PROJ_IrOx_OER/dft_workflow/job_processing/out_data/ \
-v &



# #########################################################
# df_atoms_sorted_ind
rclone copy  \
$rclone_dropbox:$PROJ_irox_oer_rc/dft_workflow/job_analysis/atoms_indices_order/out_data/ \
$wd/PROJ_IrOx_OER/dft_workflow/job_analysis/atoms_indices_order/out_data/ \
-v &


# #########################################################
# df_active_sites
rclone copy  \
$rclone_dropbox:$PROJ_irox_oer_rc/workflow/enumerate_adsorption/out_data/df_active_sites.pickle \
$wd/PROJ_IrOx_OER/workflow/enumerate_adsorption/out_data/ \
-v &

# #########################################################
# df_slabs_oh
rclone copy  \
$rclone_dropbox:$PROJ_irox_oer_rc/dft_workflow/job_analysis/create_oh_slabs/out_data/df_slabs_oh.pickle \
$wd/PROJ_IrOx_OER/dft_workflow/job_analysis/create_oh_slabs/out_data/ \
-v &

# workflow/enumerate_adsorption/out_data/df_active_sites.pickle

# path_i = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/enumerate_adsorption",
#     "out_data/df_active_sites.pickle")

# /nfs/slac/g/suncatfs/flores12/PROJ_IrOx_OER/
# dft_workflow/job_analysis/atoms_indices_order/out_data/df_atoms_sorted_ind.pickle


echo "Sync done!"



#| - __old__


# --exclude "*voro_temp*" \
# --exclude "00_abx_al_runs/out_data/**" \
# --exclude "01_abx_al_runs_new/out_data/**" \


# #########################################################
#| - dft_workflow/job_processing/out_data

# # #########################################################
# # DFT jobs dataframe, needed to not run jobs from other clusters
# rclone copy  \
# $rclone_dropbox:$PROJ_irox_oer_rc/dft_workflow/job_processing/out_data/df_jobs_combined.pickle \
# $wd/PROJ_IrOx_OER/dft_workflow/job_processing/out_data/ \
# -v &
#
# # #########################################################
# # TEMP
# rclone copy  \
# $rclone_dropbox:$PROJ_irox_oer_rc/dft_workflow/job_processing/out_data/df_jobs_paths.pickle \
# $wd/PROJ_IrOx_OER/dft_workflow/job_processing/out_data/ \
# -v &
#
# # #########################################################
# # DFT jobs data dataframe
# rclone copy  \
# $rclone_dropbox:$PROJ_irox_oer_rc/dft_workflow/job_processing/out_data/df_jobs_data.pickle \
# $wd/PROJ_IrOx_OER/dft_workflow/job_processing/out_data/ \
# -v &
#
# # #########################################################
# # DFT data dataframe
# rclone copy  \
# $rclone_dropbox:$PROJ_irox_oer_rc/workflow/process_bulk_dft/out_data/df_dft.pickle \
# $wd/PROJ_IrOx_OER/workflow/process_bulk_dft/out_data/ \
# -v &
#
# # #########################################################
# # DFT jobs data dataframe
# rclone copy  \
# $rclone_dropbox:$PROJ_irox_oer_rc/dft_workflow/job_processing/out_data/job_id_mapping.csv \
# $wd/PROJ_IrOx_OER/dft_workflow/job_processing/out_data/ \
# -v &

#__|


#__|
