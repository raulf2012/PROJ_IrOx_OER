# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Setup/continue jobs
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import dictdiffer
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.options.display.max_colwidth = 120

from ase import io

# #########################################################
from methods import (
    get_df_jobs_data,
    get_df_jobs,
    get_df_jobs_paths,
    get_df_jobs_anal,
    get_df_slab,
    cwd,
    )

# #########################################################
from dft_workflow_methods import (
    is_job_understandable,
    job_decision,
    transfer_job_files_from_old_to_new,
    is_job_compl_done,
    )
# -

from methods import get_df_jobs_on_clus__all

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Script Inputs

# +
# TEST_no_file_ops = True  # True if just testing around, False for production mode
TEST_no_file_ops = False

# Slac queue to submit to
slac_sub_queue = "suncat3"  # 'suncat', 'suncat2', 'suncat3'
# -

# ### Read Data

# +
df_jobs = get_df_jobs()

df_jobs_data = get_df_jobs_data(drop_cols=False)

df_jobs_paths = get_df_jobs_paths()

df_jobs_anal = get_df_jobs_anal()

df_slab = get_df_slab()
# -

df_jobs_on_clus__all = get_df_jobs_on_clus__all()

# ### Filtering down to `oer_adsorbate` jobs

# +
# # Uncomment this to only rerun `oer_adsorbate` jobs
# # Otherwise both dos_bader and oer_adsorbate jobs are rerun

# df_ind = df_jobs_anal.index.to_frame()

# df_jobs_anal = df_jobs_anal.loc[
#     df_ind[df_ind.job_type == "oer_adsorbate"].index
#     ]

# +
# Removing systems that were marked to be ignored
from methods import get_systems_to_stop_run_indices

indices_to_stop_running = get_systems_to_stop_run_indices(df_jobs_anal=df_jobs_anal)
df_jobs_anal = df_jobs_anal.drop(index=indices_to_stop_running)

df_resubmit = df_jobs_anal
# -

# ### Filter `df_resubmit` to only rows that are to be resubmitted

job_ids_to_force_resubmit = [
    # Wed May 12 18:57:44 PDT 2021
    # "luhegesu_97",

    # Sat May 22 22:28:51 PDT 2021
    "renonero_79",
    "sahutoho_38",
    "nasowihu_34",
    "muvawosa_27",
    "gudehuhu_68",
    "kuwimolu_53",
    "digemobe_04",
    "peremese_97",
    "hiramama_87",
    "wusifiha_94",
    "tapidudi_05",
    "resopuke_29",
    "rinisohe_79",
    "garahipo_46",
    "gogihaba_12",
    "ratorage_16",


    # # #####################################################
    # # "fukudiko_66",
    # # "fipipida_61",
    # # "tibunane_36",

    # # #####################################################
    # "tisakuri_50",
    # "hukemena_85",
    # "sewedawu_95",
    # "rugagumu_31",
    # "gitogahu_48",
    # "bigufoha_89",
    # "pubahadu_79",
    # "hebomume_93",
    # "makaledi_83",
    # "fevupilo_12",
    # "ruvanaka_31",
    # "wirurabi_88",
    # "ludekatu_27",
    # "hogigova_47",
    # "koboguhi_40",
    # "kalidasa_91",
    # "larodaru_75",
    # # ############
    # "nonogase_08",
    # "gorofiwe_14",
    # "gatevehu_95",
    # "kebolapu_40",
    # "pesusero_02",
    # # ############
    # "parihobe_18",
    # "fuhegara_35",
    # "ropirema_45",
    # "sisipule_40",
    # "valagane_51",
    # "revotaho_43",

    # "pigenipi_49",
    # "nitisule_36",

    # # ############
    # "budubadi_27",
    # "mudebupo_43",

    # # ############
    # "nunidaha_71",
    # "kegifopa_85",

    # # ############
    # "liludigu_62",
    # "sahutoho_38",

    ]

# ### Reading resubmit data from `rerun_unfinished_slabs.ipynb`

# #########################################################
import pickle; import os
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/run_slabs/rerun_unfinished_slabs",
    "out_data")
path_i = os.path.join(
    directory,
    "df_to_rerun__not_force_conv.pickle")
with open(path_i, "rb") as fle:
    df_to_rerun__not_force_conv = pickle.load(fle)
# #########################################################

# +
df_resubmit_tmp = copy.deepcopy(df_resubmit)

# #########################################################
mask_list = []

# for i in df_resubmit.decision.tolist():
for name_i, row_i in df_resubmit.iterrows():
    decision_i = row_i.decision
    job_id_max_i = row_i.job_id_max

    # if "resubmit" in decision_i or \
    #    job_id_max_i in job_ids_to_force_resubmit or \
    #    name_i in df_to_rerun__not_force_conv.index:
    #     mask_list.append(True)

        
    if "resubmit" in decision_i or job_id_max_i in job_ids_to_force_resubmit:
        mask_list.append(True)
    elif name_i in df_to_rerun__not_force_conv.index:
        row_rerun_i = df_to_rerun__not_force_conv.loc[name_i]
        job_id_max__rerun = row_rerun_i.job_id_max

        if job_id_max_i == job_id_max__rerun:
            mask_list.append(True)
        else:
            mask_list.append(False)

    else:
        mask_list.append(False)

df_resubmit = df_resubmit_tmp[mask_list]
df_nosubmit = df_resubmit_tmp[np.invert(mask_list)]

# print("df_resubmit.shape:", df_resubmit.shape)
# print("df_nosubmit.shape:", df_nosubmit.shape)
# -

# ### Processing `systems_to_stop_running`

# +
df_i = df_nosubmit[df_nosubmit.job_completely_done == False]

index_mask = []
for name_i, row_i in df_i.iterrows():
    decision_i = row_i.decision
    job_id_max_i = row_i.job_id_max

    if job_id_max_i not in df_jobs_on_clus__all.job_id:

        if len(decision_i) == 0:
            index_mask.append(name_i)
        else:
            add_name_to_mask = False
            for decision_str_j in decision_i:
                str_frags = ["not understandable", ]
                for str_i in str_frags:
                    if str_i in decision_str_j:
                        add_name_to_mask = True

            if add_name_to_mask:
                index_mask.append(name_i)

df_i = df_i.loc[index_mask]





# #########################################################
# Only rerunning from slab generations > 1
def method(row_i):
    job_id_max_i = row_i.job_id_max
    row_jobs_i = df_jobs.loc[job_id_max_i]
    slab_id_i = row_jobs_i.slab_id
    row_slab_i = df_slab.loc[slab_id_i]
    phase_i = row_slab_i.phase
    return(phase_i)
df_i["phase"] = df_i.apply(method,axis=1)
df_i = df_i[df_i.phase > 1]

# #########################################################
if df_i.shape[0] > 0:
    print("There are jobs being left idle, nothing to do, fix it")
    print(df_i.job_id_max.tolist())
    print("")

    job_ids = df_i.job_id_max.tolist()

    df_jobs_i = df_jobs.loc[job_ids]

    for job_id_i, row_i in df_jobs_i.iterrows():
        row_path_i = df_jobs_paths.loc[job_id_i]
        gdrive_path_i = row_path_i.gdrive_path

        path_i = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            gdrive_path_i,
            "job.err")

        if verbose:
            print(20 * "*")
            print("gdrive_path_i:", gdrive_path_i)
            print("job_id_i:", job_id_i)
            print("")
            print("")


            if verbose:
                my_file = Path(path_i)
                if my_file.is_file():

                    with open(path_i, "r") as f:
                        lines = f.read().splitlines()

                    tmp = [print(i) for i in lines[-10:]]
                    print("")

# +
# assert False
# -

# ### Creating new job directories and initializing

# +
data_dict_list = []
for i_cnt, (name_i, row_i) in enumerate(df_resubmit.iterrows()):

    data_dict_i = dict()
    if verbose:
        print(40 * "*")
        print(name_i)

    # #####################################################
    job_type_i = name_i[0]
    compenv_i = name_i[1]
    slab_id_i = name_i[2]
    ads_i = name_i[3]
    active_site_i = name_i[4]
    att_num_i = name_i[5]
    # #####################################################

    # #####################################################
    job_id_max_i = row_i.job_id_max
    dft_params_new = row_i.dft_params_new
    # #####################################################

    # #####################################################
    df_jobs_i = df_jobs[df_jobs.compenv == compenv_i]
    row_jobs_i = df_jobs_i.loc[job_id_max_i]
    # #####################################################
    rev_num = row_jobs_i.rev_num
    # #####################################################

    # #####################################################
    df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv_i]
    row_paths_max_i = df_jobs_paths_i.loc[job_id_max_i]
    # #####################################################
    gdrive_path = row_paths_max_i.gdrive_path
    # #####################################################

    # #####################################################
    df_jobs_data_i = df_jobs_data[df_jobs_data.compenv == compenv_i]
    row_data_max_i = df_jobs_data_i.loc[job_id_max_i]
    # #####################################################
    num_steps = row_data_max_i.num_steps
    incar_params = row_data_max_i.incar_params
    # #####################################################


    path_i = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        gdrive_path)


    from pathlib import Path
    outcar_is_there = False
    my_file = Path(os.path.join(path_i, "OUTCAR"))
    if my_file.is_file():
        outcar_is_there = True


    # #####################################################
    # Copy files to new job dir
    new_path_i = "/".join(path_i.split("/")[0:-1] + ["_" + str(rev_num + 1).zfill(2)])

    if not TEST_no_file_ops:
        if not os.path.exists(new_path_i):
            print(new_path_i)
            os.makedirs(new_path_i)

            
    files_to_transfer_for_new_job = [

        [
            os.path.join(path_i, "model.py"),
            "model.py"],
        [
            "dir_dft_params/dft-params.json",
             "dft-params.json"],

        "WAVECAR",
        "dft-params.json",
        "data_dict.json",
        ]


    contcar_is_there = False
    my_file = Path(os.path.join(path_i, "CONTCAR"))
    if my_file.is_file():
        contcar_is_there = True

    not_ready = False
    if num_steps > 0 and not contcar_is_there:
        print("num_steps > 0 but CONTCAR is not avail", name_i)
        not_ready = True


    if not not_ready:
        # #####################################################
        with cwd(path_i):
            if num_steps > 0:
                atoms = io.read("CONTCAR")
                atoms.write("contcar_out.traj")
                files_to_transfer_for_new_job.append(
                    ["contcar_out.traj", "init.traj"])


            else:
                atoms = io.read("init.traj")
                files_to_transfer_for_new_job.append(
                    "init.traj"
                    )

            # If spin-polarized calculation then get magmoms from prev. job and pass to new job
            if outcar_is_there:
                if incar_params["ISPIN"] == 2:
                    if num_steps > 0:
                        atoms_outcar = io.read("OUTCAR")
                        magmoms_i_tmp = atoms_outcar.get_magnetic_moments()

                        data_path = os.path.join("out_data/magmoms_out.json")
                        with open(data_path, "w") as outfile:
                            json.dump(magmoms_i_tmp.tolist(), outfile)

                        files_to_transfer_for_new_job.append(
                            ["out_data/magmoms_out.json", "magmoms.json"])
                # If previous job was the non-spin-pol calc, then copy through the magmoms.json file
                elif incar_params["ISPIN"] == 1:
                    files_to_transfer_for_new_job.append("magmoms.json")
            else:
                files_to_transfer_for_new_job.append("magmoms.json")


            num_atoms = atoms.get_global_number_of_atoms()



        # #####################################################
        if not TEST_no_file_ops:
            transfer_job_files_from_old_to_new(
                path_i=path_i,
                new_path_i=new_path_i,
                files_to_transfer_for_new_job=files_to_transfer_for_new_job,
                )

        # #####################################################
        if not TEST_no_file_ops:
            dft_params_path_i = os.path.join(
                new_path_i,
                "dft-params.json")
            with open(dft_params_path_i, "r") as fle:
                dft_params_current = json.load(fle)

            # Update previous DFT parameters with new ones
            dft_params_current.update(dft_params_new)

            with open(dft_params_path_i, "w") as outfile:
                json.dump(dft_params_current, outfile, indent=2)

        # #####################################################
        data_dict_i["path_i"] = new_path_i
        data_dict_i["num_atoms"] = num_atoms
        data_dict_list.append(data_dict_i)

        # else:
        #     print("OUTCAR file wasn't where it should be, probably need rclone")
        #     print("name_i:", name_i)

# #########################################################
df_sub = pd.DataFrame(data_dict_list)
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("setup_new_jobs.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
