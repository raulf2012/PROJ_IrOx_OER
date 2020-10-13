#!/bin/bash

"""
"""

if [[ $1 == "to_local" ]]; then
    # | - to_local
    rclone sync \
    $rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/run_slabs/ \
    /home/raulf2012/rclone_temp/PROJ_irox_oer/dft_workflow/run_slabs/ \
    --exclude "out_data.old/**" \
    --transfers=40 --checkers=40 --verbose --tpslimit=10 

    # --exclude "/*.old/**"
    # --exclude *.old

    # $rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/run_slabs/run_o_covered/out_data \
    # /home/raulf2012/rclone_temp/PROJ_irox_oer/dft_workflow/run_slabs/run_o_covered/out_data \

    # __|
elif [[ $1 == "to_cloud" ]]; then
    # | - to_cloud
    # rclone sync \
    rclone copy \
    /home/raulf2012/rclone_temp/PROJ_irox_oer/dft_workflow/run_slabs/ \
    $rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/run_slabs/ \
    --transfers=40 --checkers=40 --verbose --tpslimit=10

    # /home/raulf2012/rclone_temp/PROJ_irox_oer/dft_workflow/run_slabs/run_o_covered/out_data \
    # $rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/run_slabs/run_o_covered/out_data \
    # __|
else
    echo "Didn't choose an option"
    echo "Either sync GDrive cloud with local dirs (to_local)"
    echo "or"
    echo "Sync local changes back to cloud (to_cloud)"
fi

