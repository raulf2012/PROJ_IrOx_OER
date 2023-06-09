#!/bin/bash



#| - __old__











TEMP_BASH()
{
echo "TEMP BASH FUNCTION"
}








foo()
{

    foo_usage() { echo "foo: [-a <arg>]" 1>&2; exit; }

    local OPTIND o a
    while getopts ":a:" o; do
        case "${o}" in


            a)
                a="${OPTARG}"

                echo "TM djfsifjds 89u9f9sd"

                ;;
            *)
                foo_usage
                ;;
        esac

        echo "HERE IIII"

    done

    shift $((OPTIND-1))

    echo "a: [${a}], non-option arguments: $*"

}






foo_2()
{

a_flag='false'
b_flag='false'
files='false'
verbose='false'

print_usage() {
  printf "Usage: ..."
}

while getopts 'abf:v' flag; do
  case "${flag}" in
    a) a_flag='true' ;;
    b) b_flag='true' ;;
    f) files="${OPTARG}" ;;
    v) verbose='true' ;;
    *) print_usage
       exit 1 ;;
  esac
done


echo a_flag $a_flag
echo b_flag $b_flag
echo files $files
echo verbose $verbose

}




# __|



# #########################################################
# Location of bash aliases and function for project
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

PROJ_irox_oer__comm_overnight()
#| - PROJ_irox_oer__comm_overnight
{

echo ""
echo "Example usage:"
echo "  PROJ_irox_oer__comm_overnight -s 5 -i 2"


echo ""
echo "Parameter values:"
echo "  s: sleep_time"
echo "  i: iter_num"
echo "  m: run_misc"
echo ""

# | - Argument Parsing
sleep_time=5
iter_num=1
run_misc="false"

print_usage() {
  printf "Usage: ..."
}

while getopts 's:i:m' flag; do
  case "${flag}" in
    s) sleep_time="${OPTARG}" ;;
    i) iter_num="${OPTARG}" ;;
    m) run_misc="true" ;;
    *) print_usage
       exit 1 ;;
  esac
done


if [[ $sleep_time == "no_wait" ]]; then
  sleep_time=5
fi

echo sleep_time $sleep_time
echo iter_num $iter_num
echo run_misc $run_misc


# | - __old__
# echo "----"
# echo $sleep_time
# echo "----"
# if [[ -n "$2" ]]
#     then
#         iter_num=$2
#     else
#         iter_num=3
# fi

# echo "sleep_time"
# echo $sleep_time
#
# echo "iter_num"
# echo $iter_num
# __|


# __|


if [[ "$COMPENV" == "wsl" ]]; then
  filename_suff=""
elif [[ "$COMPENV" == "nersc" ]] || [[ "$COMPENV" == "slac" ]] || [[ "$COMPENV" == "sherlock" ]]; then
  filename_suff=__$COMPENV
fi


echo ""
echo "filename_suff"
echo $filename_suff
echo ""


filename="PROJ_irox_oer__comm_overnight_wrap"$filename_suff".log"


echo "Filename:"
echo $filename
echo ""


# #########################################################
echo "Running PROJ_irox_oer__comm_overnight_wrap"
# PROJ_irox_oer__comm_overnight_wrap "$@" | tee \
# $PROJ_irox_oer/log_files/$filename

PROJ_irox_oer__comm_overnight_wrap $sleep_time $iter_num $run_misc | tee \
$PROJ_irox_oer/log_files/$filename



# Copy the current output to .old file to preserve in case of overwrite
echo "Copy output to .old file"
cp \
$PROJ_irox_oer/log_files/$filename \
$PROJ_irox_oer/log_files/$filename.old


# Rclone copy file to Dropbox (if run on cluster)
echo "Rclone file to dropbox"
rclone copy \
$PROJ_irox_oer/log_files/$filename \
$rclone_dropbox:$rclone_dropbox_path_to_dir/log_files \

}
#__|

PROJ_irox_oer__comm_overnight_wrap()
#| - PROJ_irox_oer__comm_overnight
{

# | - Printing date and time
now=$(date)

echo ""
echo "--------------------"
echo "CURRENT TIME AND DATE"
printf "%s\n" "$now"
echo "--------------------"
echo ""
# __|


sleep_time=$1
iter_num=$2
run_misc=$3


#| - OLD | Arg parsing
# sleep_time=5m
# if [[ $1 == "no_wait" ]]; then
#     sleep_time=5
# fi
#
# if [[ -n "$2" ]]
#     then
#         iter_num=$2
#     else
#         iter_num=3
# fi
#
# echo "sleep_time"
# echo $sleep_time
#
# echo "iter_num"
# echo $iter_num
#__|




END=$iter_num
for ((i=1; i<=END; i++))
do

#| - Echo header
echo ""
echo ""
echo ""
echo ""
echo ""
echo "#| - Fold start | Loop: $i"
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
#__|

start_time="$(date -u +%s)"

#| - All the guts

# echo "All the guts TEMP TEST"


if [[ "$COMPENV" == "wsl" ]]; then
    # | - If in wsl

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/cluster_scripts/collect_df_from_clust.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/cluster_scripts/collect_df_from_clust.py


    echo ""
    echo ""
    echo ""
    echo ""

    #| - $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "bash $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh to_local"
    echo "############################################################################################################################################"
    echo "#| - $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh"
    echo ""
    echo ""
    echo ""
    echo ""

    # if [[ $GDRIVE_DAEMON == False ]]; then
    #     bash $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh to_local no_verbose
    # else
    #     echo "Skipping $PROJ_irox_oer/scripts/rclone_commands/rclone_gdrive_dell.sh"
    #     echo "GDrive is running"
    # fi

    end_time_i="$(date -u +%s)"
    elapsed_i="$(($end_time_i-$start_time))"
    elapsed_min_i=$(($elapsed_i / 60))

    echo ""
    echo ""
    echo ""
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "rclone_gdrive_dell.sh"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Run time: $elapsed_i s"
    echo "Run time: $elapsed_min_i min"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    echo ""
    echo "#__|"
    echo "############################################################################################################################################"


    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - PROJ_irox_oer__comm_jobs_clean
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_clean"
    echo "############################################################################################################################################"
    echo "#| - PROJ_irox_oer__comm_jobs_clean"
    echo ""
    echo ""
    echo ""
    echo ""

    PROJ_irox_oer__comm_jobs_clean

    end_time_i="$(date -u +%s)"
    elapsed_i="$(($end_time_i-$start_time))"
    elapsed_min_i=$(($elapsed_i / 60))

    echo ""
    echo ""
    echo ""
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "PROJ_irox_oer__comm_jobs_clean"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Run time: $elapsed_i s"
    echo "Run time: $elapsed_min_i min"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    echo ""
    echo "#__|"
    echo "############################################################################################################################################"


    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - $PROJ_irox_oer/dft_workflow/bin/fix_gdrive_conflicts.ipynb
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "python $PROJ_irox_oer/dft_workflow/bin/fix_gdrive_conflicts.ipynb"
    echo "############################################################################################################################################"
    echo "#| - $PROJ_irox_oer/dft_workflow/bin/fix_gdrive_conflicts.ipynb"
    echo ""
    echo ""
    echo ""
    echo ""

    python $PROJ_irox_oer/dft_workflow/bin/fix_gdrive_conflicts.py

    end_time_i="$(date -u +%s)"
    elapsed_i="$(($end_time_i-$start_time))"
    elapsed_min_i=$(($elapsed_i / 60))

    echo ""
    echo ""
    echo ""
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "fix_gdrive_conflicts.ipynb"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Run time: $elapsed_i s"
    echo "Run time: $elapsed_min_i min"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    echo ""
    echo "#__|"
    echo "############################################################################################################################################"


    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - PROJ_irox_oer__comm_jobs_update
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_update all"
    echo "############################################################################################################################################"
    echo "#| - PROJ_irox_oer__comm_jobs_update"
    echo ""
    echo ""
    echo ""
    echo ""

    PROJ_irox_oer__comm_jobs_update all

    end_time_i="$(date -u +%s)"
    elapsed_i="$(($end_time_i-$start_time))"
    elapsed_min_i=$(($elapsed_i / 60))

    echo ""
    echo ""
    echo ""
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "PROJ_irox_oer__comm_jobs_update"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Run time: $elapsed_i s"
    echo "Run time: $elapsed_min_i min"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    echo ""
    echo "#__|"
    echo "############################################################################################################################################"


    echo ""
    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - setup_new_jobs.sh
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "bash setup_new_jobs.sh"
    echo "############################################################################################################################################"
    echo "#| - setup_new_jobs.sh"
    echo ""
    echo ""
    echo ""
    echo ""

    cd $PROJ_irox_oer/dft_workflow/run_slabs
    bash setup_new_jobs.sh
    cd $PROJ_irox_oer

    end_time_i="$(date -u +%s)"
    elapsed_i="$(($end_time_i-$start_time))"
    elapsed_min_i=$(($elapsed_i / 60))

    echo ""
    echo ""
    echo ""
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "setup_new_jobs.sh"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Run time: $elapsed_i s"
    echo "Run time: $elapsed_min_i min"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    echo ""
    echo "#__|"
    echo "############################################################################################################################################"


    echo ""
    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus"
    echo "############################################################################################################################################"
    echo "#| - PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus"
    echo ""
    echo ""
    echo ""
    echo ""

    PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus

    end_time_i="$(date -u +%s)"
    elapsed_i="$(($end_time_i-$start_time))"
    elapsed_min_i=$(($elapsed_i / 60))

    echo ""
    echo ""
    echo ""
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Run time: $elapsed_i s"
    echo "Run time: $elapsed_min_i min"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    echo ""
    echo "#__|"
    echo "############################################################################################################################################"


    echo ""
    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - PROJ_irox_oer__comm_features_update

    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_features_update"
    echo "############################################################################################################################################"
    echo "#| - PROJ_irox_oer__comm_features_update"
    echo ""
    echo ""
    echo ""
    echo ""

    PROJ_irox_oer__comm_features_update

    end_time_i="$(date -u +%s)"
    elapsed_i="$(($end_time_i-$start_time))"
    elapsed_min_i=$(($elapsed_i / 60))

    end_time_i="$(date -u +%s)"
    elapsed_i="$(($end_time_i-$start_time))"
    elapsed_min_i=$(($elapsed_i / 60))

    echo ""
    echo ""
    echo ""
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "PROJ_irox_oer__comm_features_update"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Run time: $elapsed_i s"
    echo "Run time: $elapsed_min_i min"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    echo ""
    echo "#__|"
    echo "############################################################################################################################################"


    echo ""
    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - PROJ_irox_oer__comm_plotting_update
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_plotting_update"
    echo "############################################################################################################################################"
    echo "#| - PROJ_irox_oer__comm_plotting_update"
    echo ""
    echo ""
    echo ""
    echo ""

    PROJ_irox_oer__comm_plotting_update


    end_time_i="$(date -u +%s)"
    elapsed_i="$(($end_time_i-$start_time))"
    elapsed_min_i=$(($elapsed_i / 60))

    echo ""
    echo ""
    echo ""
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "PROJ_irox_oer__comm_plotting_update"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo "Run time: $elapsed_i s"
    echo "Run time: $elapsed_min_i min"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    echo ""
    echo "#__|"
    echo "############################################################################################################################################"


    echo ""
    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|



    if [[ "$run_misc" == "true" ]]; then

      #| - PROJ_irox_oer__comm_misc_update
      echo ""
      echo ""
      echo ""
      echo ""
      echo "run_misc=true"
      echo ""
      echo ""
      echo "############################################################################################################################################"
      echo "PROJ_irox_oer__comm_misc_update"
      echo "############################################################################################################################################"
      echo "#| - PROJ_irox_oer__comm_misc_update"
      echo ""
      echo ""
      echo ""
      echo ""

      PROJ_irox_oer__comm_misc_update


      end_time_i="$(date -u +%s)"
      elapsed_i="$(($end_time_i-$start_time))"
      elapsed_min_i=$(($elapsed_i / 60))

      echo ""
      echo ""
      echo ""
      echo ""
      echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
      echo "PROJ_irox_oer__comm_misc_update"
      echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
      echo "Run time: $elapsed_i s"
      echo "Run time: $elapsed_min_i min"
      echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

      echo ""
      echo "#__|"
      echo "############################################################################################################################################"


      echo ""
      echo ""
      echo "Sleeping... for: $sleep_time"
      sleep $sleep_time

      echo ""
      echo ""
      echo ""
      #__|

    elif [[ "$run_misc" == "false" ]]; then
      echo "run_misc=false"
    fi



    # __|
elif [[ "$COMPENV" == "nersc" ]] || [[ "$COMPENV" == "slac" ]] || [[ "$COMPENV" == "sherlock" ]]; then
    # | - If on computing cluster

    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/cluster_scripts/get_jobs_running.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/cluster_scripts/get_jobs_running.py


    #| - PROJ_irox_oer__comm_jobs_update
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_update"
    echo "############################################################################################################################################"
    echo "#| - PROJ_irox_oer__comm_jobs_update"
    echo ""
    echo ""
    echo ""
    echo ""

    PROJ_irox_oer__comm_jobs_update

    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"


    echo ""
    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus"
    echo "############################################################################################################################################"
    echo "#| - PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus"
    echo ""
    echo ""
    echo ""
    echo ""

    PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus

    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"


    echo ""
    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - PROJ_irox_oer__comm_jobs_run_unsub_jobs
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_run_unsub_jobs run frac_of_jobs_to_run=1."
    echo "############################################################################################################################################"
    echo "#| - PROJ_irox_oer__comm_jobs_run_unsub_jobs"
    echo ""
    echo ""
    echo ""
    echo ""

    # PROJ_irox_oer__comm_jobs_run_unsub_jobs run frac_of_jobs_to_run=1.
    PROJ_irox_oer__comm_jobs_run_unsub_jobs run frac_of_jobs_to_run=1. verbose=False

    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"


    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    #| - PROJ_irox_oer__comm_jobs_clean
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"
    echo "PROJ_irox_oer__comm_jobs_clean"
    echo "############################################################################################################################################"
    echo "#| - PROJ_irox_oer__comm_jobs_clean"
    echo ""
    echo ""
    echo ""
    echo ""

    PROJ_irox_oer__comm_jobs_clean

    echo ""
    echo ""
    echo ""
    echo ""
    echo "############################################################################################################################################"


    echo ""
    echo ""
    echo "Sleeping... for: $sleep_time"
    sleep $sleep_time

    echo ""
    echo ""
    echo ""
    #__|

    # __|
fi

#__|

end_time="$(date -u +%s)"

elapsed="$(($end_time-$start_time))"

elapsed_min=$(($elapsed / 60))

echo ""
echo "Total of $elapsed seconds elapsed for this loop number"
echo "Total of $elapsed_min minutes elapsed for this loop number"


echo ""
echo "#__| - Fold End"

done

}
#__|



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

      # | - $PROJ_irox_oer/dft_workflow/job_analysis/collect_collate_dft_data/collect_collate_dft.ipynb
      echo ""
      echo ""
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/collect_collate_dft_data/collect_collate_dft.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/collect_collate_dft_data/collect_collate_dft.py
      # __|

      # | - $PROJ_irox_oer/dft_workflow/job_analysis/get_init_slabs_bare_oh/get_init_slabs_bare_oh.ipynb
      echo ""
      echo ""
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/get_init_slabs_bare_oh/get_init_slabs_bare_oh.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/get_init_slabs_bare_oh/get_init_slabs_bare_oh.py
      # __|

      # | - $PROJ_irox_oer/dft_workflow/job_analysis/atoms_indices_order/correct_atom_indices_order.ipynb
      echo ""
      echo ""
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/atoms_indices_order/correct_atom_indices_order.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/atoms_indices_order/correct_atom_indices_order.py
      # __|

      # | - $PROJ_irox_oer/dft_workflow/job_analysis/df_coord_for_post_dft/coord_env_for_post_dft.ipynb
      echo ""
      echo ""
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/df_coord_for_post_dft/coord_env_for_post_dft.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/df_coord_for_post_dft/coord_env_for_post_dft.py
      # __|

      # | - $PROJ_irox_oer/dft_workflow/job_analysis/create_oh_slabs/create_oh_slabs.ipynb
      echo ""
      echo ""
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/create_oh_slabs/create_oh_slabs.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/create_oh_slabs/create_oh_slabs.py
      # __|

      # | - $PROJ_irox_oer/dft_workflow/job_analysis/analyze_oh_jobs/anal_oh_slabs.ipynb
      echo ""
      echo ""
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/analyze_oh_jobs/anal_oh_slabs.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/analyze_oh_jobs/anal_oh_slabs.py
      # __|

      # | - $PROJ_irox_oer/workflow/creating_slabs/maintain_df_coord/fix_df_coord.ipynb
      echo ""
      echo ""
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/workflow/creating_slabs/maintain_df_coord/fix_df_coord.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/workflow/creating_slabs/maintain_df_coord/fix_df_coord.py
      # __|

      # | - $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/compare_magoms.ipynb
      echo ""
      echo ""
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/compare_magoms.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/compare_magoms.py
      # __|

      # | - $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/decide_jobs_magmoms.ipynb
      echo ""
      echo ""
      echo ""
      echo ""
      echo ""
      echo "****************************************"
      echo "python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/decide_jobs_magmoms.ipynb"
      echo "****************************************"
      python $PROJ_irox_oer/dft_workflow/job_analysis/compare_magmoms/decide_jobs_magmoms.py
      # __|

      #__|

    # 08saduijf | parser end

  fi

    # __|
elif [[ "$COMPENV" == "nersc" ]] || [[ "$COMPENV" == "slac" ]] || [[ "$COMPENV" == "sherlock" ]]; then
    # | - if on computing cluster
    echo "we're in a computer cluster"

    # | - $PROJ_irox_oer/scripts/rclone_commands/rclone_proj_repo.sh
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "bash $PROJ_irox_oer/scripts/rclone_commands/rclone_proj_repo.sh > /dev/null 2>&1"
    echo "****************************************"
    bash $PROJ_irox_oer/scripts/rclone_commands/rclone_proj_repo.sh > /dev/null 2>&1
    # __|

    # | - $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.ipynb
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_dirs.py
    # __|

    # | - $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.ipynb
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/collect_job_dirs_data.py
    # __|

    # | - $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.ipynb
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.ipynb"
    echo "****************************************"
    python $PROJ_irox_oer/dft_workflow/job_processing/parse_job_data.py
    # __|

    # | - proj_irox_oer__comm_rclone_sync
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "****************************************"
    echo "proj_irox_oer__comm_rclone_sync > /dev/null 2>&1"
    echo "****************************************"
    PROJ_irox_oer__comm_rclone_sync > /dev/null 2>&1
    # __|

    echo ""
    echo ""
    echo ""

    # __|
fi

}
#__|


PROJ_irox_oer__comm_features_update()
#| - PROJ_irox_oer__comm_features_update
{

  # 5jfds7u8ik | Parser Start

  # | - $PROJ_irox_oer/workflow/feature_engineering/generate_features/octahedra_volume/octa_volume.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/generate_features/octahedra_volume/octa_volume.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/generate_features/octahedra_volume/octa_volume.py
  # __|

  # | - $PROJ_irox_oer/workflow/feature_engineering/generate_features/oxid_state/oxid_state.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/generate_features/oxid_state/oxid_state.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/generate_features/oxid_state/oxid_state.py
  # __|

  # | - $PROJ_irox_oer/workflow/feature_engineering/generate_features/active_site_angles/AS_angles.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/generate_features/active_site_angles/AS_angles.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/generate_features/active_site_angles/AS_angles.py
  # __|

  # | - $PROJ_irox_oer/workflow/dos_analysis/collect_dos_data.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/dos_analysis/collect_dos_data.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/dos_analysis/collect_dos_data.py
  # __|

  # | - $PROJ_irox_oer/workflow/feature_engineering/generate_features/pdos_features/pdos_feat.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/generate_features/pdos_features/pdos_feat.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/generate_features/pdos_features/pdos_feat.py
  # __|

  # | - $PROJ_irox_oer/workflow/feature_engineering/generate_features/bader_features/bader_feat.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/generate_features/bader_features/bader_feat.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/generate_features/bader_features/bader_feat.py

  # workflow/feature_engineering/generate_features/bader_features/bader_feat.py
  # __|




  # #######################################################
  # Running the feature collection and combination notebooks
  # #######################################################

  # | - $PROJ_irox_oer/workflow/feature_engineering/collect_feature_data.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/collect_feature_data.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/collect_feature_data.py
  # __|

  # | - $PROJ_irox_oer/workflow/feature_engineering/combine_features_targets.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/combine_features_targets.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/combine_features_targets.py
  # __|

  echo ""
  echo ""
  echo ""
  echo ""
  echo ""

  # 5jfds7u8ik | Parser End

}
#__|


# #########################################################
# Run scripts that create plots
PROJ_irox_oer__comm_plotting_update()
#| - PROJ_irox_oer__comm_plotting_update
{

  # 5jfds7u8ik | Parser Start

  # | - $PROJ_irox_oer/workflow/oer_analysis/oer_analysis.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_analysis/oer_analysis.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_analysis/oer_analysis.py
  # __|

  # | - $PROJ_irox_oer/workflow/oer_analysis/oer_scaling/oer_scaling.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_analysis/oer_scaling/oer_scaling.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_analysis/oer_scaling/oer_scaling.py
  # __|

  # | - $PROJ_irox_oer/workflow/oer_vs_features/new_oer_vs_features.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_vs_features/new_oer_vs_features.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_vs_features/new_oer_vs_features.py
  # __|

  # | - $PROJ_irox_oer/workflow/oer_vs_features/oer_lin_model.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_vs_features/oer_lin_model.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_vs_features/oer_lin_model.py
  # __|

  # | - $PROJ_irox_oer/workflow/feature_engineering/analyse_features/scatter_plot_matrix/scatter_plot_matrix.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/analyse_features/scatter_plot_matrix/scatter_plot_matrix.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/analyse_features/scatter_plot_matrix/scatter_plot_matrix.py
  # __|

  # | - $PROJ_irox_oer/workflow/oer_analysis/volcano_1d/volcano_1d.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_analysis/volcano_1d/volcano_1d.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_analysis/volcano_1d/volcano_1d.py
  # __|

  # | - $PROJ_irox_oer/workflow/oer_analysis/volcano_2d/volcano_2d.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/oer_analysis/volcano_2d/volcano_2d.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/oer_analysis/volcano_2d/volcano_2d.py
  # __|

  # | - $PROJ_irox_oer/workflow/feature_engineering/analyse_features/feature_covariance/feat_correlation.ipynb
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/workflow/feature_engineering/analyse_features/feature_covariance/feat_correlation.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/workflow/feature_engineering/analyse_features/feature_covariance/feat_correlation.py
  # __|

  # 5jfds7u8ik | Parser End

}
#__|

# #########################################################
# Misc scripts, ideally should be able to be run after everything else
# Minimal dependencies, etc.
PROJ_irox_oer__comm_misc_update()
#| - PROJ_irox_oer__comm_misc_update
{

  # | - $PROJ_irox_oer/dft_workflow/job_analysis/prepare_oer_sets/write_oer_sets.py
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
  echo "****************************************"
  echo "python $PROJ_irox_oer/dft_workflow/job_analysis/prepare_oer_sets/write_oer_sets.ipynb"
  echo "****************************************"
  python $PROJ_irox_oer/dft_workflow/job_analysis/prepare_oer_sets/write_oer_sets.py
  # __|

}
#__|










PROJ_irox_oer__comm_jobs_clean()
#| - PROJ_irox_oer__comm_jobs_clean
{

echo ""
echo ""
echo ""
echo ""
echo "############################################################################################################################################"
echo "python $PROJ_irox_oer/dft_workflow/job_processing/clean_dft_dirs.ipynb"
echo "############################################################################################################################################"
python $PROJ_irox_oer/dft_workflow/job_processing/clean_dft_dirs.py
echo "############################################################################################################################################"
echo ""
echo ""
echo ""

}
#__|

PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus()
#| - PROJ_irox_oer__comm_jobs_sync_unsub_jobs_too_clus
{

echo ""
echo ""
echo ""
echo ""
echo "############################################################################################################################################"
echo "python $PROJ_irox_oer/dft_workflow/bin/sync_unsub_jobs_to_clus.ipynb"
echo "############################################################################################################################################"
python $PROJ_irox_oer/dft_workflow/bin/sync_unsub_jobs_to_clus.py
echo "############################################################################################################################################"
echo ""
echo ""
echo ""

}
#__|

PROJ_irox_oer__comm_jobs_run_unsub_jobs()
#| - PROJ_irox_oer__comm_jobs_run_unsub_jobs
{

echo ""
echo ""
echo ""
bash $PROJ_irox_oer/dft_workflow/bin/run_unsub_jobs "$@"
echo ""
echo ""
echo ""

}
#__|
