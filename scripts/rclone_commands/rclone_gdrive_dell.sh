#!/bin/bash

if [[ $1 == "to_local" ]]; then
    # | - to_local

    # echo "ijidsjfisd98jsf9jds8fh8seufisdjdifjds"
    # echo $2
    # echo "ijidsjfisd98jsf9jds8fh8seufisdjdifjds"

    if [[ $2 == "no_verbose" ]]; then
        echo "Turned off verbose flag"
        verbose_flag=""
    else
        verbose_flag="--verbose"
    fi

    # rclone sync \
    rclone copy \
    $rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/run_slabs/ \
    $PROJ_irox_oer_gdrive/dft_workflow/run_slabs/ \
    --exclude "out_data.old/**" \
    --exclude vasprun.xml \
    --exclude PROCAR \
    --exclude DOSCAR \
    --transfers=80 --checkers=80 --tpslimit=80 \
    --stats-file-name-length=150 \
    $verbose_flag 


    # __|
elif [[ $1 == "to_cloud" ]]; then
    # | - to_cloud
    rclone copy \
    $PROJ_irox_oer_gdrive/dft_workflow/run_slabs/ \
    $rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/run_slabs/ \
    --transfers=80 --checkers=80 --tpslimit=80 --verbose

    # /home/raulf2012/rclone_temp/PROJ_irox_oer/dft_workflow/run_slabs/ \

    # __|
else
    echo "Didn't choose an option"
    echo "Either sync GDrive cloud with local dirs (to_local)"
    echo "or"
    echo "Sync local changes back to cloud (to_cloud)"
fi

format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }

echo "Script completed in $(format_time $SECONDS)"

