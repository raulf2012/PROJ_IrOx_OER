#!/bin/bash

if [[ $2 == "no_verbose" ]]; then
    echo "Turned off verbose flag"
    verbose_flag=""
else
    verbose_flag="--verbose"
fi

# $rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/sherlock/zimixdvdxd/2-1-11/active_site__61/01_attempt/_01/ \
# $PROJ_irox_oer_gdrive/dft_workflow/run_dos_bader/run_o_covered/out_data/dft_jobs/sherlock/zimixdvdxd/2-1-11/active_site__61/01_attempt/_01/ \

# $rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/ \
# $PROJ_irox_oer_gdrive/dft_workflow/ \

rclone sync \
$rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/ \
$PROJ_irox_oer_gdrive/ \
--exclude "out_data.old/**" \
--exclude vasprun.xml \
--exclude PROCAR \
--exclude DOSCAR \
--transfers=80 --checkers=80 --tpslimit=80 \
--stats-file-name-length=150 \
$verbose_flag

format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }

echo "Script completed in $(format_time $SECONDS)"
