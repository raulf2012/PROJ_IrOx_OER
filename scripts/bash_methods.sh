#!/bin/bash

# #########################################################
# Location of bash aliases and function for project
#
# #########################################################


PROJ_irox_oer__comm_rclone_sync()
#| - PROJ_irox_oer__comm_rclone_sync
{
bash $PROJ_irox_oer/scripts/rclone_commands/rclone_proj_repo.sh
}
#__|

# #########################################################
#| - Job methods

PROJ_irox_oer__comm_jobs_update()
#| - PROJ_irox_oer__comm_jobs_update
{

if [[ "$COMPENV" == "wsl" ]]; then
    # | - If in wsl

    # $PROJ_irox_oer/dft_workflow/job_processing/analyze_jobs.py
    # $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.py
    # $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py

    # $PROJ_irox_oer/dft_workflow/job_analysis/atoms_indices_order/correct_atom_indices_order.py
    # $PROJ_irox_oer/dft_workflow/job_analysis/collect_collate_dft_data/collect_collate_dft.py
    # $PROJ_irox_oer/dft_workflow/job_analysis/create_oh_slabs/create_oh_slabs.py
    # $PROJ_irox_oer/dft_workflow/job_analysis/df_coord_for_post_dft/coord_env_for_post_dft.py
    # $PROJ_irox_oer/dft_workflow/job_analysis/get_init_slabs_bare_oh/get_init_slabs_bare_oh.py


    #| -  Light scripts

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.py"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.py

    # dft_workflow/job_processing/parse_job_dirs.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.py"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/analyze_jobs.py"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/analyze_jobs.py
    #__|

    echo ""
    echo ""
    echo ""
    echo "pass 'all' flag to run all notebooks"

    if [[ $1 == "all" ]]; then

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/collect_collate_dft_data/collect_collate_dft.py"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/collect_collate_dft_data/collect_collate_dft.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/get_init_slabs_bare_oh/get_init_slabs_bare_oh.py"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/get_init_slabs_bare_oh/get_init_slabs_bare_oh.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/atoms_indices_order/correct_atom_indices_order.py"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/atoms_indices_order/correct_atom_indices_order.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/df_coord_for_post_dft/coord_env_for_post_dft.py"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/df_coord_for_post_dft/coord_env_for_post_dft.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/create_oh_slabs/create_oh_slabs.py"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/create_oh_slabs/create_oh_slabs.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/analyze_oh_jobs/anal_oh_slabs.py"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/analyze_oh_jobs/anal_oh_slabs.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/compare_magoms.py"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/compare_magoms.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/decide_jobs_magmoms.py"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/decide_jobs_magmoms.py

    fi


    # echo ""
    # echo ""
    # echo ""
    # echo "****************************************"
    # echo "python $PROJ_irox_oer/"
    # echo "****************************************"
    # python $PROJ_irox_oer/


    # __|
elif [[ "$COMPENV" == "nersc" ]] || [[ "$COMPENV" == "slac" ]] || [[ "$COMPENV" == "sherlock" ]]; then
    # | - If on computing cluster
    echo "We're in a computer cluster"

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "bash $PROJ_irox_oer/scripts/rclone_commands/rclone_proj_repo.sh > /dev/null 2>&1"
    echo "****************************************"
    bash $PROJ_irox_oer/scripts/rclone_commands/rclone_proj_repo.sh > /dev/null 2>&1

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.py"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.py"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "PROJ_irox_oer__comm_rclone_sync > /dev/null 2>&1"
    echo "****************************************"
    PROJ_irox_oer__comm_rclone_sync > /dev/null 2>&1

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.py"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py


    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "PROJ_irox_oer__comm_rclone_sync"
    echo "****************************************"
    PROJ_irox_oer__comm_rclone_sync > /dev/null 2>&1

    # __|
fi

}
#__|


PROJ_irox_oer__comm_features_update()
#| - PROJ_irox_oer__comm_features_update
{

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/octahedra_volume/octa_volume.py"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/octahedra_volume/octa_volume.py

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/oxid_state/oxid_state.py"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/oxid_state/oxid_state.py

  # #######################################################
  # #######################################################

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/collect_feature_data.py"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/collect_feature_data.py

}
#__|




PROJ_irox_oer__comm_jobs_clean()
#| - PROJ_irox_oer__comm_jobs_clean
{
python $PROJ_irox_oer/dft_workflow/job_processing/clean_dft_dirs.py
}
#__|

PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus()
#| - PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus
{
python $PROJ_irox_oer/dft_workflow/bin/sync_unsub_jobs_to_clus.py
# dft_workflow/job_processing/clean_dft_dirs.py
}
#__|

PROJ_irox_oer__comm_jobs_run_unsub_jobs()
#| - PROJ_irox_oer__comm_jobs_run_unsub_jobs
{
# bash $PROJ_irox_oer/dft_workflow/bin/run_unsub_jobs $1
bash $PROJ_irox_oer/dft_workflow/bin/run_unsub_jobs "$@"

echo "TEMP"
echo "$@"
echo "TEMP"
}
#__|


#__|
# #########################################################
