# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_IrOx_Active_Learning_OER]
#     language: python
#     name: conda-env-PROJ_IrOx_Active_Learning_OER-py
# ---

# +
"""
"""

# | - Import Modules
import os
import sys

from json import dump, load
from shutil import copyfile
#__|

# + jupyter={"source_hidden": true}
# dirs_list = [
#     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/CatHub_MPContribs_upload/CatHub_upload/read_db_file.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/CatHub_MPContribs_upload/MPContribs_upload/creating_dataset_for_mp_runs.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/CatHub_MPContribs_upload/MPContribs_upload/mpcontribs_upload.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/CatHub_MPContribs_upload/MPContribs_upload/updating_dataset_props.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/CatHub_MPContribs_upload/MPContribs_upload/duplicate_MP_entries/find_duplicates_in_mp.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/dft_structures/01_bulk_structures/parse_oer_structures.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/parse_dft_data/parse_all_data_new.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/python_classes/dev_new_al_class/comp_perf_plots.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/python_classes/dev_new_al_class/dev_notebook_1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/python_classes/dev_new_al_class/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/run_nersc_vasp/ml_bulk_opt/run_all_bulks/rerun_jobs.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/run_nersc_vasp/ml_bulk_opt/run_all_bulks/setup_dirs.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/scripts/convert_jup_to_pyth.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/scripts/git_check_size.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/00_notebook_directory.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/an_data_processing_test.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/01_surface_energies/01_surface_e_convergence/an_surf_e_convergence__v4.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/01_surface_energies/01_surface_e_convergence/test_an_surf_e_convergence__v4.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/01_surface_energies/02_surface_e_pourb_plot/an_surface-energy_pourbaix__v4.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/01_surface_energies/02_surface_e_pourb_plot/SANDBOX.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/01_surface_energies/02_surface_e_pourb_plot/first_method/an_surface-energy_pourbaix__v1-Copy1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/01_surface_energies/02_surface_e_pourb_plot/first_method/an_surface-energy_pourbaix__v1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/01_surface_energies/02_surface_e_pourb_plot/first_method/creating_legend.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/01_surface_energies/iro3_100_o_covered_weird_stoich/Untitled.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/01_2d_oer_volc/an_2d_volcano_plot.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/01_2d_oer_volc/an_2d_volcano_plot_temp.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/01_2d_oer_volc/real_2d_volc.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/02_oer_volc/an_irox_volcano_plotly__v1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/02_oer_volc/kinetic_volcano_colin__v1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/03_ads_e_scaling/an_irox_scaling__v2.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/03_ads_e_scaling/01_calc_scal_fixed_slope/an_irox_scaling__v2.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/05_oer_fed/an_irox_fed_tmp.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/misc_analysis/get_diff_between_iro2_iro3.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/misc_analysis/get_slab_vacuum_dist/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/02_oer_analysis/oer_data_table/create_latex_table.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/03_combined_oer_surf_e/an_surface-energy_pourbaix__v4.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/04_old_surf_pourb/an_irox_pourbaix_tmp.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/06_quality_control/an_irox_quality_contr.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/07_bulk_pourbaix/01_pourbaix_scripts/an_pourbaix_plot.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/07_bulk_pourbaix/01_pourbaix_scripts/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/07_bulk_pourbaix/01_pourbaix_scripts/sandbox_pourb_transitions.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/07_bulk_pourbaix/01_pourbaix_scripts/01_pourb_anim/create_pourb_anim.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/07_bulk_pourbaix/01_pourbaix_scripts/old.02_small_toc_fig/create_toc_pourb.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/07_bulk_pourbaix/02_free_energy_calc/an_energy_treatment.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/07_bulk_pourbaix/02_free_energy_calc/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/07_bulk_pourbaix/03_data_table/bulk_pourb_latex_table.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/07_bulk_pourbaix/looking_into_IrO4_ion/get_ion_info.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/08_measuring_comp_resourc/an_meas_comp_resourc.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/bonding_theory_analysis/octa_hedra_bonding_model.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/dos_pdos_irox/a_iro3/rapiDOS_analysis.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/dos_pdos_irox/b_iro3/rapiDOS_analysis.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/dos_pdos_irox/r_iro2/rapiDOS_analysis.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/dos_pdos_irox/r_iro3/rapiDOS_analysis.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/energy_treatment_deriv/energy_derivation.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/energy_treatment_deriv/calc_references/calc_O_Ir_refs__H_G.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/metastability_limit_murat/metastability.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/metastability_limit_murat/create_data_table/create_data_table.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/ml_methods__test.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/00_abx_al_runs/run_al_multiple_times.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/00_abx_al_runs/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/01_abx_al_runs_new/run_al_multiple_times.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/01_abx_al_runs_new/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/al_animation/create_animations.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/al_animation/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/al_plots_for_figure/ml_plots__v4.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/al_plots_for_figure/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/combined_al_plot/create_subplots__v5.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/get_duplicates_from_al/get_duplicates_list.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/get_duplicates_from_al/old.verify_duplicates.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/parity_plots/parity_post_dft.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/parity_plots/parity_pre_dft.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/parity_plots/plotting_results.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/performance_comp/most_stable_when_disc/get_ave_gen_of_top_disc.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/performance_comp/pca_analysis_sandbox/pca_var_covered.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/performance_comp/top_10_disc_vs_dft/comp_perf_plots__v2.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/00_ml_workflow/performance_comp/top_10_disc_vs_dft/01_plot_all_runs/plot_all_runs.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/ccf_similarity_analysis/analyse_dij_matrix.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/ccf_similarity_analysis/calc_diff_pre_post_dft.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/ccf_similarity_analysis/compute_ccf_and_dij_matrix/compute_ccf__v1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/ccf_similarity_analysis/compute_ccf_and_dij_matrix/get_d_ij_matrix__v1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/creating_irox_dataset_for_xrd/create_irox_datasets.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/energy_vs_volume/analyse_environment_raul.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/energy_vs_volume/create_coord_env_df.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/energy_vs_volume/load_raul.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/energy_vs_volume/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/energy_vs_volume/00_main_plotting_script/00_plotting_kirsten.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/manually_classify_dft_structures/analyze_oxy_coord.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/manually_classify_dft_structures/coordination_env.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/manually_classify_dft_structures/manually_analyze_structures.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/manually_classify_dft_structures/old.analyze_coord_motiffs.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/manually_classify_dft_structures/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/opt_mae_err_gp_model/cv_error__v3.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/opt_mae_err_gp_model/test_send_email.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/opt_mae_err_gp_model/plotting/plot_cv_error.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/collect_all_bulk_dft_data.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/comparing_raul_chris_dft/comp_chris_raul_calcs.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/comparing_raul_chris_dft/comp_chris_raul_calcs__v1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/creating_final_dataset_for_upload/create_final_dft.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/figuring_size_of_final_final_dataset/analysis__v0.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/figuring_size_of_final_final_dataset/analysis__v1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/figuring_size_of_final_final_dataset/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/out_data/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/parse_chris_bulk_dft/parse_chris_dirs.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/parse_chris_bulk_dft/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/parse_my_bulk_dft/parse_my_bulk_dft.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/parse_my_bulk_dft/track_ids__v0.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/parse_my_bulk_dft/track_ids__v1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/parse_my_bulk_dft/track_ids__v2.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/parse_my_bulk_dft/track_ids__v3.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/parse_my_oer_bulk_dft/an_parse_oer_bulk_systems.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/parse_oqmd_data/prepare_oqmd_data.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/prototype_classification/classify_prototypes.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/static_prototypes_structures/create_atoms_df.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/static_prototypes_structures/elim_high_num_atom_cnt.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/static_prototypes_structures/process_dupl_proto.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/static_prototypes_structures/reoptimized_struct_kirsten.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/static_prototypes_structures/sandbox.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/symmetry_analysis/reduce_atoms.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/processing_bulk_dft/symmetry_analysis_dft_static/classify_symm.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/prototypes_of_oer_systems/getting_prototype_names-Copy1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/visualizing_data/plot_features.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/visualizing_data/plot_features_pre_post_training.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/voronoi_featurize/01_data_coll_featurization__v4.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/voronoi_featurize/02_fingerprints_pre_opt.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/voronoi_featurize/cumulative_performance.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/voronoi_featurize/feature_analysis.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/ml_modelling/__misc__/compare_alpha_phases/compare_alpha_phases_with_ml.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/190311_testing_ads_structures.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/190314_playing_with_ads_df.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/an_IrOx_sandbox_1.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/an_IrOx_sandbox_2.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/an_IrOx_sandbox_4.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/an_IrOx_sandbox_6_ads_fe_corr.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/sandbox_190103.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/test_new_df.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/test_old_df.ipynb",
# #     "/mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_Active_Learning_OER/workflow/tmp_sandbox/tmp_test_df_meth.ipynb",
#     ]

# +
dirs_to_ignore = [
    ".ipynb_checkpoints",
    "__old__",
    "__temp__",
    "__test__",
    "__bkp__",
    ]

root_dir = os.path.join(
    os.environ["PROJ_irox"],
    # "workflow/tmp_sandbox",
    )

dirs_list = []
for subdir, dirs, files in os.walk(root_dir):

    # | - Pass over ignored directories
    ignore_subdir = False
    for ignore_dir in dirs_to_ignore:
        if ignore_dir in subdir:
            ignore_subdir = True
    if ignore_subdir:
        continue
    #__|

    for file in files:
        if ".swp" in file:
            # pass
            continue

        if ".ipynb" in file:
            file_path = os.path.join(subdir, file)
            full_dir_i = os.path.join(subdir, file)

            # print(file_path)
            # print(os.path.join(subdir, file))

            dirs_list.append(full_dir_i)
# -

for file_i in dirs_list:
    print(file_i)

    bash_comm = "jupytext --to py" + " " + file_i
    os.system(bash_comm)
