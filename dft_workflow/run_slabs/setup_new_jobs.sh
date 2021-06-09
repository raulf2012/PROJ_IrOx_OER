#!/bin/bash


# 08saduijf | Parser Start

echo " "
echo " "
echo " "
echo " "
echo "****************************************"
echo "$PROJ_irox_oer/dft_workflow/manually_analyze_slabs/manually_analyze_slabs.ipynb"
echo "****************************************"
python $PROJ_irox_oer/dft_workflow/manually_analyze_slabs/manually_analyze_slabs.py

echo " "
echo " "
echo " "
echo " "
echo "****************************************"
echo "$PROJ_irox_oer/dft_workflow/run_slabs/setup_new_jobs.ipynb"
echo "****************************************"
python $PROJ_irox_oer/dft_workflow/run_slabs/setup_new_jobs.py

echo " "
echo " "
echo " "
echo " "
echo "****************************************"
echo "$PROJ_irox_oer/dft_workflow/run_slabs/run_bare_oh_covered/setup_dft_bare.ipynb"
echo "****************************************"
python $PROJ_irox_oer/dft_workflow/run_slabs/run_bare_oh_covered/setup_dft_bare.py


echo " "
echo " "
echo " "
echo " "
echo "****************************************"
echo "$PROJ_irox_oer/dft_workflow/run_slabs/run_oh_covered/setup_dft_oh.ipynb"
echo "****************************************"
python $PROJ_irox_oer/dft_workflow/run_slabs/run_oh_covered/setup_dft_oh.py


echo ""
echo ""
echo ""
echo "****************************************"
echo "python $PROJ_irox_oer/dft_workflow/run_slabs/setup_jobs_from_oh/setup_new_jobs_from_oh.ipynb"
echo "****************************************"
python $PROJ_irox_oer/dft_workflow/run_slabs/setup_jobs_from_oh/setup_new_jobs_from_oh.py


# 08saduijf | Parser End
