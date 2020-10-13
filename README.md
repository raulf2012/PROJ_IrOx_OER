# Project path variables and commands

### Paths:
`$PROJ_irox_oer`
`$PROJ_irox_oer_gdrive`
`$PROJ_irox_oer_data`
`$PROJ_irox_oer_paper` 

### Commands:
`PROJ_irox_oer__comm_rclone_sync`


# Import links
Github repo for project code:
https://github.com/raulf2012/PROJ_IrOx_OER

Github Dropbox location:
https://www.dropbox.com/sh/2457uwjlftnict4/AAA5Thi4UYwcwbHqI-od4sM0a?dl=0

Workflowy link:
https://beta.workflowy.com/s/irox-surface-structu/3WVcm1z8egYdgfgv

Dropbox project folder:
https://www.dropbox.com/sh/5ajpxzefmqcmf6c/AAB_u5xINSMRS7zkhzR4z9vEa?dl=0

Article Overleaf link:
https://www.overleaf.com/8152133417ngprszxttkws

Article @Github location:
https://github.com/raulf2012/PAPER_IrOx_OER

# Setting up environment

Create conda environment

`conda create --name PROJ_irox_oer python=3.6.10 --no-default-packages`

Python version: 3.6.10

    conda install \
    -c plotly \
    -c conda-forge \
    -c anaconda \
    jupyterlab jupytext \
    scikit-learn=0.23.1 matplotlib=3.2.1 scipy=1.4.1 pandas=1.0.4 \
    plotly=4.8.1 chart-studio=1.1.0 plotly-orca psutil=5.7.0 colormap colorlover \
    ase=3.19.1 pymatgen=2020.4.29 gpflow \
    nodejs=10  tensorflow \
    yaml nbclean notedown ipywidgets nb_conda_kernels nodejs dictdiffer

Run these jupyter commands to install useful labextensions

    jupyter labextension install @rahlir/theme-gruvbox --no-build
    jupyter labextension install axelfahy/jupyterlab-vim --no-build
    jupyter labextension install @jupyter-widgets/jupyterlab-manager


TODO Move this

This was needed to get the `tqdm` python module to work properly with Jupyterlab

    jupyter nbextension enable --py widgetsnbextension
    conda install -c conda-forge ipywidgets
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
