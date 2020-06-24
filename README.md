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

`conda create --name PROJ_irox_oer --no-default-packages`

Python version: 3.6.10

    conda install \
    -c plotly \
    -c conda-forge \
    -c anaconda \
    nodejs jupyterlab jupytext \
    scikit-learn matplotlib scipy pandas \
    plotly chart-studio plotly-orca psutil colormap colorlover \
    ase pymatgen=2020.4.29 gpflow \
    nodejs=10 nb_conda_kernels tensorflow \
    yaml nbclean notedown

Run these jupyter commands to install useful labextensions

    jupyter labextension install @rahlir/theme-gruvbox --no-build
    jupyter labextension install axelfahy/jupyterlab-vim --no-build

TODO Move this

This was needed to get the `tqdm` python module to work properly with Jupyterlab

    jupyter nbextension enable --py widgetsnbextension
    conda install -c conda-forge ipywidgets
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
