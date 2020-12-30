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
# #########################################################
# Job methods
# #########################################################
# #########################################################

PROJ_irox_oer__comm_jobs_update()
#| - PROJ_irox_oer__comm_jobs_update
{

if [[ "$COMPENV" == "wsl" ]]; then
    # | - If in wsl

    # 08saduijf | Parser Start


    #| -  Light scripts

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.py


    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/analyze_jobs.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/analyze_jobs.py
    #__|

    echo ""
    echo ""
    echo ""
    echo "pass 'all' flag to run all notebooks"

    if [[ $1 == "all" ]]; then
      #| - all other scripts
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/collect_collate_dft_data/collect_collate_dft.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/collect_collate_dft_data/collect_collate_dft.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/get_init_slabs_bare_oh/get_init_slabs_bare_oh.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/get_init_slabs_bare_oh/get_init_slabs_bare_oh.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/atoms_indices_order/correct_atom_indices_order.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/atoms_indices_order/correct_atom_indices_order.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/df_coord_for_post_dft/coord_env_for_post_dft.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/df_coord_for_post_dft/coord_env_for_post_dft.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/create_oh_slabs/create_oh_slabs.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/create_oh_slabs/create_oh_slabs.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/analyze_oh_jobs/anal_oh_slabs.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/analyze_oh_jobs/anal_oh_slabs.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/workflow/creating_slabs/maintain_df_coord/fix_df_coord.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/workflow/creating_slabs/maintain_df_coord/fix_df_coord.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "$PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/compare_magoms.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/compare_magoms.py

      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "$PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/decide_jobs_magmoms.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/decide_jobs_magmoms.py

      #__|

    # 08saduijf | parser end
  fi

    # __|
elif [[ "$COMPENV" == "nersc" ]] || [[ "$COMPENV" == "slac" ]] || [[ "$COMPENV" == "sherlock" ]]; then
    # | - if on computing cluster
    echo "we're in a computer cluster"

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
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "proj_irox_oer__comm_rclone_sync > /dev/null 2>&1"
    echo "****************************************"
    PROJ_irox_oer__comm_rclone_sync > /dev/null 2>&1


    # __|
fi

}
#__|


PROJ_irox_oer__comm_features_update()
#| - PROJ_irox_oer__comm_features_update
{

  # 5jfds7u8ik | Parser Start

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/octahedra_volume/octa_volume.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/octahedra_volume/octa_volume.py

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/oxid_state/oxid_state.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/oxid_state/oxid_state.py

  # #######################################################
  # #######################################################

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/collect_feature_data.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/collect_feature_data.py


  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/combine_features_targets.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/combine_features_targets.py


  # 5jfds7u8ik | Parser End

}
#__|


PROJ_irox_oer__comm_plotting_update()
#| - PROJ_irox_oer__comm_plotting_update
{

  # 5jfds7u8ik | Parser Start






  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_analysis/oer_analysis.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_analysis/oer_analysis.py

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_analysis/oer_scaling/oer_scaling.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_analysis/oer_scaling/oer_scaling.py

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_vs_features/new_oer_vs_features.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_vs_features/new_oer_vs_features.py

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_vs_features/oer_lin_model.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_vs_features/oer_lin_model.py

  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_vs_features/scatter_plot_matrix/scatter_plot_matrix.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_vs_features/scatter_plot_matrix/scatter_plot_matrix.py







  # echo ""
  # echo ""
  # echo ""
  # echo "****************************************"
  # echo "python $PROJ_irox_oer/"
  # echo "****************************************"
  # python $PROJ_irox_oer/


  # 5jfds7u8ik | Parser End

}
#__|












PROJ_irox_oer__comm_overnight()
#| - PROJ_irox_oer__comm_overnight
{

PROJ_irox_oer__comm_overnight_wrap "$@" | tee $PROJ_irox_oer/log_files/PROJ_irox_oer__comm_overnight_wrap.log

cp \
$PROJ_irox_oer/log_files/PROJ_irox_oer__comm_overnight_wrap.log \
$PROJ_irox_oer/log_files/PROJ_irox_oer__comm_overnight_wrap.log.old

}
#__|

PROJ_irox_oer__comm_overnight_wrap()
#| - PROJ_irox_oer__comm_overnight
{
sleep_time=5m
if [[ $1 == "no_wait" ]]; then
    sleep_time=5
fi

if [[ -n "$2" ]]
    then
        iter_num=$2
    else
        iter_num=3
fi

echo "iter_num"
echo $iter_num

END=$iter_num
for ((i=1; i<=END; i++))
# for i in {1..$iter_num}

do

echo ""
echo ""
echo ""
echo ""
echo ""
echo "Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i"
echo "Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i"
echo "Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i"
echo "Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i"
echo "Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i"
echo "Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i    Loop Number: $i"
echo ""
echo ""
echo ""
echo ""
echo ""

#| - TEMP | All the guts
if [[ "$COMPENV" == "wsl" ]]; then
    # | - If in wsl

    echo ""
    echo "############################################################################################################################################"
    echo "bash $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh to_local"
    echo "############################################################################################################################################"
    if [[ $GDRIVE_DAEMON == False ]]; then
        bash $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh to_local no_verbose
    else
        echo "Skipping $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh"
        echo "GDrive is running"
    fi
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_clean"
    echo "############################################################################################################################################"
    PROJ_irox_oer__comm_jobs_clean
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "python $PROJ_irox_oer/dft_workflow/bin/fix_gdrive_conflicts.ipynb"
    echo "############################################################################################################################################"
    python $PROJ_irox_oer/dft_workflow/bin/fix_gdrive_conflicts.py
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_update all"
    echo "############################################################################################################################################"
    PROJ_irox_oer__comm_jobs_update all
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "bash setup_new_jobs.sh"
    echo "############################################################################################################################################"
    cd $PROJ_irox_oer/dft_workflow/run_slabs
    bash setup_new_jobs.sh
    cd $PROJ_irox_oer
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus"
    echo "############################################################################################################################################"
    PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "bash $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh to_local"
    echo "############################################################################################################################################"
    if [[ $GDRIVE_DAEMON == False ]]; then
        bash $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh to_local no_verbose
    else
        echo "Skipping $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh"
        echo "GDrive is running"
    fi
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_features_update"
    echo "############################################################################################################################################"
    PROJ_irox_oer__comm_features_update
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_plotting_update"
    echo "############################################################################################################################################"
    PROJ_irox_oer__comm_plotting_update
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    # __|
elif [[ "$COMPENV" == "nersc" ]] || [[ "$COMPENV" == "slac" ]] || [[ "$COMPENV" == "sherlock" ]]; then
    # | - If on computing cluster


    # #####################################################

    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_update"
    echo "############################################################################################################################################"
    PROJ_irox_oer__comm_jobs_update
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus"
    echo "############################################################################################################################################"
    PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_run_unsub_jobs run frac_of_jobs_to_run=1."
    echo "############################################################################################################################################"
    # PROJ_irox_oer__comm_jobs_run_unsub_jobs run frac_of_jobs_to_run=1.
    PROJ_irox_oer__comm_jobs_run_unsub_jobs run frac_of_jobs_to_run=1. verbose=False
    echo "############################################################################################################################################"

    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_clean"
    echo "############################################################################################################################################"
    PROJ_irox_oer__comm_jobs_clean
    echo "############################################################################################################################################"





    # __|
fi
#__|

done

}
#__|
















PROJ_irox_oer__comm_jobs_clean()
#| - PROJ_irox_oer__comm_jobs_clean
{

echo ""
echo "############################################################################################################################################"
echo "python $PROJ_irox_oer/dft_workflow/job_processing/clean_dft_dirs.py"
echo "############################################################################################################################################"
python $PROJ_irox_oer/dft_workflow/job_processing/clean_dft_dirs.py
echo "############################################################################################################################################"

}
#__|

PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus()
#| - PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus
{

echo ""
echo "############################################################################################################################################"
echo "python $PROJ_irox_oer/dft_workflow/bin/sync_unsub_jobs_to_clus.ipynb"
echo "############################################################################################################################################"
python $PROJ_irox_oer/dft_workflow/bin/sync_unsub_jobs_to_clus.py
echo "############################################################################################################################################"

}
#__|

PROJ_irox_oer__comm_jobs_run_unsub_jobs()
#| - PROJ_irox_oer__comm_jobs_run_unsub_jobs
{
bash $PROJ_irox_oer/dft_workflow/bin/run_unsub_jobs "$@"

# echo "TEMP"
# echo "$@"
# echo "TEMP"
}
#__|
