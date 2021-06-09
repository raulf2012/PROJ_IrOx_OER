"""DFT jobs workflow related methods.
"""

#| - Import Modules
import os
import sys

import random
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

# #########################################################
from proj_data import compenv

# #########################################################
from dft_job_automat.compute_env import ComputerCluster
#__|

from proj_data import compenvs


# #########################################################

def is_job_understandable(
    timed_out=None,
    completed=None,
    error=None,
    job_state=None,
    ):
    #| - is_job_understandable
    understandable_job = False

    if timed_out:
        understandable_job = True

    if completed:
        understandable_job = True

    if error:
        understandable_job = True

    acceptable_job_states = [
        "PENDING", "SUCCEEDED",
        "SUCCEEDED", "RUNNING", "FAILED"]
    if job_state in acceptable_job_states:
        understandable_job = True

    # if understandable_job:
    #     print("This job is UNDERSTANDABLE")

    return(understandable_job)
    #__|

def job_decision(
    error=None,
    error_type=None,
    timed_out=None,
    completed=None,
    submitted=None,
    job_understandable=None,
    ediff_conv_reached=None,
    incar_params=None,
    brmix_issue=None,
    num_nonconv_scf=None,
    num_conv_scf=None,
    true_false_ratio=None,
    frac_true=None,
    job_state=None,
    job_completely_done=None,
    ):
    """
    Possible decisions
    """
    #| - job_decision
    possible_decisions = [
        "resubmit",
        "nothing",
        # "",
        ]

    dft_params_new = dict()
    message = ""

    # decision_i = "nothing"
    decision_i = []


    data_root_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/dft_scripts/out_data")
    data_path = os.path.join(data_root_path, "conservative_mixing_params.json")
    with open(data_path, "r") as fle:
        dft_calc_sett_mix = json.load(fle)


    # #####################################################
    #| - Closed decision logic

    #| - If job not submitted yet, do nothing
    if submitted is False:
        decision_i = ["nothing", "Not submitted"]
        out_dict = dict(decision=decision_i, dft_params=dict())
        return(out_dict)
    #__|

    #| - If the job is not understandable then don't continue
    if not job_understandable:
        decision_i = ["nothing", "not understandable"]
        out_dict = dict(decision=decision_i, dft_params=dft_params_new)
        return(out_dict)
    #__|

    #| - If completed and ISPIN < 2 then resubmit with ISPIN=2
    if completed:
        ispin = incar_params.get("ISPIN", None)
        # I added inequality because I accidentally ran a ISPIN=0 job, which defaults to ISPIN=1
        if ispin <= 1:
            message += "Finished ISPIN=1 calculation, move onto ISPIN=2"
            decision_i = ["resubmit", ]
            dft_params_new["ispin"] = 2

            out_dict = dict(decision=decision_i, dft_params=dft_params_new)
            return(out_dict)
    #__|

    #| - If job_state is pending or running then do nothing
    acceptable_job_states = ["PENDING", "RUNNING"]
    if job_state in acceptable_job_states:
        decision_i = ["nothing", job_state]
        out_dict = dict(decision=decision_i, dft_params=dict())
        return(out_dict)
    #__|

    #| - If job is completely done, do nothing
    if job_completely_done:
        decision_i = ["nothing", "All done"]
        out_dict = dict(decision=decision_i, dft_params=dict())
        return(out_dict)
    #__|

    #__|

    # #####################################################
    #| - If the job simply timed out and is not complete then resubmit
    if (True is True) and \
        (timed_out is True) and \
        (completed is not True) and \
        (True is True):
        decision_i.append("resubmit")
        # decision_i = "resubmit"
    #__|

    # #####################################################
    #| - If most of scf cycles are unconverged and the final ediff convergence is not achieved then tweak parameters

    # This is essentially the same as the if statement below it except it doesn't have the frac_true requirement
    # I had a job that finished  a bunch of SCF cycles and then hit a brmix isssue
    if not ediff_conv_reached and brmix_issue:
        decision_i.append("tweak_params")

        # data_root_path = os.path.join(
        #     os.environ["PROJ_irox_oer"],
        #     "dft_workflow/dft_scripts/out_data")
        # data_path = os.path.join(data_root_path, "conservative_mixing_params.json")
        # with open(data_path, "r") as fle:
        #     dft_calc_sett_mix = json.load(fle)

        dft_params_new.update(dft_calc_sett_mix)


    if frac_true < 0.2 and not ediff_conv_reached:
        decision_i.append("tweak_params")

        #| - If it's the BRMIX issue then reduce the mixing paramters
        if brmix_issue:
            dft_params_new.update(dft_calc_sett_mix)
            # dft_params_new.update(dict(ispin=0))
        #__|

        if brmix_issue != True:

            algo_i = incar_params.get("ALGO", None)
            if algo_i == "Normal":
                algo_new = "Fast"
                dft_params_new.update(dict(algo=algo_new))

            prec_i = incar_params.get("PREC", None)
            if prec_i == "Normal":
                prec_new = "Accurate"
                dft_params_new.update(dict(prec=prec_new))


    #__|

    # #####################################################
    #| - If cgroup out-of-memory handler error
    if error and error_type == "out-of-memory handler":
        decision_i.append("out of memory")
        decision_i.append("resubmit")
        dft_params_new.update(dict(
            kpar=10,
            npar=6,
            ))
    #__|

    # #####################################################
    #| - If POSMAP symmetry error occurs
    if error and error_type == "POSMAP symm err":
        decision_i.append("tweak_params")
        decision_i.append("resubmit")
        dft_params_new.update(dict(symprec=1.0e-4))
    #__|

    # #####################################################
    #| - If weird SLAC core failure error
    if error and error_type == "weird SLAC cluster error, core failed":
        decision_i.append("resubmit")

    if error and error_type == "weird SLAC cluster error, core failed (type 2)":
        decision_i.append("resubmit")

    if error and error_type == "mpirun has exited":
        decision_i.append("resubmit")
    #__|

    # print(error)
    # print(error_type)

    if len(decision_i) == 0 and len(dft_params_new) == 0:
        if error and error_type == "calculation blown up":
            decision_i.append("resubmit")
            dft_params_new.update(dft_calc_sett_mix)


    if error and error_type == "Segmentation fault":
        decision_i.append("resubmit")
    if error and error_type == "slurm allocation issue":
        decision_i.append("resubmit")



    if job_state == "FAILED":
        decision_i.append("resubmit")

    # #####################################################
    if "tweak_params" in decision_i:
        if "resubmit" not in decision_i:
            decision_i.append("resubmit")

    out_dict = dict(decision=decision_i, dft_params=dft_params_new)
    return(out_dict)
    #__|

def is_job_compl_done(ispin=None, completed=None):
    """
    """
    #| - is_job_compl_done
    job_compl = False

    cond_0 = (ispin == 2)
    cond_1 = (completed)

    if cond_0 and cond_1:
        # print("Job is completely done")
        job_compl = True

    return(job_compl)
    #__|


def transfer_job_files_from_old_to_new(
    path_i=None,
    new_path_i=None,
    files_to_transfer_for_new_job=None,
    ):
    """
    """
    #| - transfer_job_files_from_old_to_new
    for transfer_i in files_to_transfer_for_new_job:
        if type(transfer_i) == list:
            # print("temp 0")
            my_file = Path(os.path.join(path_i, transfer_i[0]))
            if my_file.is_file():
                my_file = Path(os.path.join(new_path_i, transfer_i[1]))
                if not my_file.is_file():
                    copyfile(
                        os.path.join(path_i, transfer_i[0]),
                        os.path.join(new_path_i, transfer_i[1]))
        elif type(transfer_i) == str:
            my_file = Path(os.path.join(path_i, transfer_i))
            if my_file.is_file():
                my_file = Path(os.path.join(new_path_i, transfer_i))
                if not my_file.is_file():
                    copyfile(
                        os.path.join(path_i, transfer_i),
                        os.path.join(new_path_i, transfer_i))
    #__|

def get_job_spec_dft_params(
    compenv=None,
    slac_sub_queue="suncat2",
    ):
    """
    """
    #| - get_job_spec_dft_params
    dft_params_dict = dict()

    kpar = 4
    npar = 4
    if compenv == "nersc":
        #| - nersc
        # kpar = 10
        # npar = 6

        kpar = 11
        npar = 1
        #__|
    if compenv == "sherlock":
        #| - sherlock
        npar = 4
        #__|
    if compenv == "slac":
        #| - slac
        if slac_sub_queue == "suncat":
            # 8 cores per node
            kpar = 2
            npar = 4
        elif slac_sub_queue == "suncat2":
            # 12 cores per node
            kpar = 3
            npar = 4
        elif slac_sub_queue == "suncat3":
            # 16 cores per node
            kpar = 4
            npar = 4
        else:
            print("Didn't parse slac submission queue properly")
            kpar = 2
            npar = 4
        #__|

    dft_params_dict["kpar"] = kpar
    dft_params_dict["npar"] = npar

    return(dft_params_dict)
    #__|

def get_job_spec_scheduler_params(
    compenv=None,
    ):
    """
    """
    #| - get_job_spec_scheduler_params
    out_dict = dict()

    #| - wall time calculation
    if compenv == "nersc":
        wall_time_factor = 45.
    elif compenv == "sherlock":
        wall_time_factor = 30.
    elif compenv == "slac":
        # wall_time_factor = 30.
        wall_time_factor = 90.
    else:
        print("Couldn't catch the compenv and set  wall_time_factor correctly")
        wall_time_factor = 1.
    #__|

    out_dict["wall_time_factor"] = wall_time_factor

    return(out_dict)
    #__|

def submit_job(
    nodes_i=None,
    path_i=None,
    num_atoms=None,
    wall_time_factor=1.0,
    queue=None,
    ):
    #| - submit_job

    print("nodes_i:", nodes_i)
    print("path_i:", path_i)
    print("num_atoms:", num_atoms)
    print("wall_time_factor:", wall_time_factor)
    print("queue:", queue)



    #| - Wall time
    # wall_time_i = calc_wall_time(num_atoms, factor=wall_time_factor)
    wall_time_i = calc_wall_time(num_atoms=num_atoms, num_scf=50)

    print("wall_time_i:", wall_time_i)
    wall_time_i = wall_time_factor * wall_time_i
    print("wall_time_i:", wall_time_i)
    #__|

    CC = ComputerCluster()
    if os.environ["COMPENV"] == "nersc":
        #| - nersc
        # nodes_i = 1
        nodes_i = 10

        dft_params_file = os.path.join(path_i, "dft-params.json")
        my_file = Path(dft_params_file)
        if my_file.is_file():

            with open(dft_params_file, "r") as fle:
                dft_calc_settings = json.load(fle)

            kpar_i = dft_calc_settings.get("kpar", None)
            npar_i = dft_calc_settings.get("npar", None)

            if kpar_i == 10 and npar_i == 6:
                nodes_i = 10
            else:
                nodes_i = 1

        def_params = {
            "wall_time": wall_time_i,
            # "queue": "premium",
            "queue": "regular",
            "architecture": "knl",
            "nodes": nodes_i,
            "path_i": path_i}
        #__|

    elif os.environ["COMPENV"] == "sherlock":
        #| - sherlock
        def_params = {
            "wall_time": wall_time_i,
            # "nodes": nodes_i,
            "nodes": 1,
            "path_i": path_i}
        #__|

    elif os.environ["COMPENV"] == "slac":
        #| - slac
        print("queue:", queue)

        if queue == "suncat":
            cpus_i = 8
        elif queue == "suncat2":
            cpus_i = 12
        elif queue == "suncat3":
            cpus_i = 16

        def_params = {
            "wall_time": wall_time_i,
            "cpus": cpus_i,
            "queue": queue,
            "path_i": path_i}
        #__|

    else:
        def_params = {
            "wall_time": wall_time_i,
            # "queue": "premium",
            "queue": "regular",
            # "architecture": "knl",
            "nodes": 1,
            "path_i": path_i}

    CC.submit_job(**def_params)
    #__|

def scf_per_min_model(num_atoms=None):
    """
    """
    #| - scf_per_min_model
    # m0 = lg.intercept_
    # m1 = lg.coef_[1]
    # m2 = lg.coef_[2]

    m0 = +6.47968662
    m1 = -0.12968329
    m2 = +0.00068543


    # min_x = -(m1 / (2 * m2))
    x_min = -(m1 / (2 * m2))

    X_min = np.array([x_min]).reshape(-1, 1)

    y_min = scf_poly_model(x_min, m0=m0, m1=m1, m2=m2)


    # Linear region paramters
    max_lin_x = 200  # End of linear region
    x0 = x_min
    y0 = y_min
    x1 = max_lin_x
    y1 = 0.06
    if num_atoms >= x_min and num_atoms <= max_lin_x:
        m = (y1 - y0) / (x1 - x0)
        b = (x1 * y0 - x0 * y1) / (x1 - x0)

        scf_per_min_i = m * num_atoms + b

        # print("min_y:", min_y)
        # scf_per_min_i = y_min / 2

    elif num_atoms > max_lin_x:
        scf_per_min_i = y1

    else:
        scf_per_min_i = scf_poly_model(num_atoms, m0=m0, m1=m1, m2=m2)

        # X_test = np.array([num_atoms, ]).reshape(-1, 1)
        # X_test_ = poly.fit_transform(X_test)
        # scf_per_min_i = lg.predict(X_test_)[0]

    return(scf_per_min_i)
    #__|

def calc_wall_time(num_atoms=None, num_scf=None):
    """
    """
    #| - calc_wall_time
    scf_per_min_i = scf_per_min_model(num_atoms=num_atoms)
    wall_time_i = num_scf / scf_per_min_i

    return(wall_time_i)
    #__|

def scf_poly_model(num_atoms, m0=None, m1=None, m2=None):
    """
    """
    #| - scf_poly_model
    # X_min_ = poly.fit_transform(num_atoms)
    # scf_per_min_i = lg.predict(X_min_)[0]

    # print("m0:", m0, "m1:", m1, "m2:", m2)

    # #####################################################
    scf_per_min_i = 0 + \
        (m0) * (num_atoms ** 0) + \
        (m1) * (num_atoms ** 1) + \
        (m2) * (num_atoms ** 2)
    return(scf_per_min_i)
    #__|

def print_dft_calc_changes(group=None, df_jobs_data=None):
    """
    """
    #| - print_dft_calc_changes
    ind_list = group.index.tolist()
    group = group.sort_values("rev_num")
    for i_cnt, (job_id_i, row_i) in enumerate(group.iterrows()):

        # #####################################################
        rev_num = row_i.rev_num
        # #####################################################
        row_data_i = df_jobs_data.loc[job_id_i]
        incar_params_i = row_data_i.incar_params
        # #####################################################

        print(40 * "-")
        print("rev_num:", rev_num)

        if i_cnt == 0:
            continue

        # #####################################################
        job_id_im1 = ind_list[i_cnt - 1]
        row_data_im1 = df_jobs_data.loc[job_id_im1]
        incar_params_im1 = row_data_im1.incar_params
        # #####################################################

        # TEMP
        # print(incar_params_im1)

        second_dict = incar_params_i
        first_dict = incar_params_im1

        for diff in list(dictdiffer.diff(first_dict, second_dict)):
            print(diff)
    #__|


#| - Job path processing

def parse_job_dirs(root_dir=None):
    """Parse for job directories in arbitrary directories.
    """
    #| - parse_job_dirs
    data_dict_list = []
    for subdir, dirs, files in os.walk(root_dir):

        data_dict_i = dict()
        data_dict_i["path_full"] = subdir

        last_dir = root_dir.split("/")[-1]

        path_i = os.path.join(last_dir, subdir[len(root_dir) + 1:])
        if path_i[-1] == "/":
            path_i = path_i[:-1]

        # #####################################################
        if "dft_jobs" not in subdir: continue
        if ".old" in subdir: continue
        if path_i == "": continue
        if " (" in subdir: continue
        # #####################################################
        # print(path_i)

        path_rel_to_proj = get_path_rel_to_proj(subdir)
        data_dict_i["path_rel_to_proj"] = path_rel_to_proj

        # #################################################
        out_dict = get_job_paths_info(path_i)
        data_dict_i.update(out_dict)

        is_rev_dir = data_dict_i["is_rev_dir"]
        if is_rev_dir:
            data_dict_list.append(data_dict_i)
            # print(30 * "*")
        # #################################################


    # #####################################################
    df = pd.DataFrame(data_dict_list)

    # #####################################################
    is_submitted_list = []
    is_empty_list = []
    for index_i, row_i in df.iterrows():
        path_full = row_i.path_full

        is_submitted = False
        my_file = Path(os.path.join(path_full, ".SUBMITTED"))
        if my_file.is_file():
            is_submitted = True
        is_submitted_list.append(is_submitted)

        is_empty = False
        dir_list = os.listdir(path_full)
        if len(dir_list) == 0:
            is_empty = True
        is_empty_list.append(is_empty)

    # #####################################################
    df["is_submitted"] = is_submitted_list
    df["is_empty"] = is_empty_list

    return(df)
    #__|

def is_attempt_dir(dir_i):
    """
    """
    #| - is_attempt_dir
    out_dict = dict()

    is_attempt_dir_i = False
    att_num = None

    if "_" in dir_i:

        dir_split = dir_i.split("_")
        if len(dir_split) == 2:
            if dir_split[0].isnumeric():
                att_num = int(dir_split[0])
                if dir_split[1] == "attempt":
                    is_attempt_dir_i = True

    out_dict["is_attempt_dir"] = is_attempt_dir_i
    out_dict["att_num"] = att_num

    return(out_dict)
    #__|

def get_path_rel_to_proj(full_path):
    """
    """
    #| - get_path_rel_to_proj
    subdir = full_path

    PROJ_dir = os.environ["PROJ_irox_oer"]
    ind_tmp = subdir.find(PROJ_dir.split("/")[-1])
    path_rel_to_proj = subdir[ind_tmp:]
    path_rel_to_proj = "/".join(path_rel_to_proj.split("/")[1:])

    return(path_rel_to_proj)
    #__|

def is_rev_dir(dir_i):
    """
    """
    #| - is_rev_dir
    # print("dir_i:", dir_i)

    out_dict = dict()

    assert dir_i is not None, "dir_i is None"

    is_rev_dir_i = False
    rev_num = None

    if dir_i[0] == "_":
        dir_split = dir_i.split("_")
        if len(dir_split) == 2:
            if dir_split[1].isnumeric():
                rev_num = int(dir_split[1])
                is_rev_dir_i = True

    out_dict["is_rev_dir"] = is_rev_dir_i
    out_dict["rev_num"] = rev_num

    return(out_dict)
    #__|

def get_job_paths_info(path_i):
    """
    """
    #| - get_job_paths_info
    out_dict = dict()

    # #####################################################
    start_ind_to_remove = None
    rev_num_i = None
    att_num_i = None
    path_job_root_i = None
    path_job_root_w_att_rev = None
    is_rev_dir_i_i = None
    is_attempt_dir_i = None

    path_job_root_w_att = None
    gdrive_path = None
    # #####################################################


    # #########################################################
    #  Getting the compenv
    # compenvs = ["slac", "sherlock", "nersc", ]

    compenv_out = None
    got_compenv_from_path = False
    for compenv_i in compenvs:
        compenv_in_path = compenv_i in path_i
        if compenv_in_path:
            compenv_out = compenv_i
            got_compenv_from_path = True
    if compenv_out is None:
        compenv_out = os.environ["COMPENV"]


    # #########################################################
    path_split_i = path_i.split("/")
    # print("path_split_i:", path_split_i)  # TEMP
    for i_cnt, dir_i in enumerate(path_split_i):

        out_dict_i = is_rev_dir(dir_i)
        is_rev_dir_i = out_dict_i["is_rev_dir"]
        rev_num_i = out_dict_i["rev_num"]
        if is_rev_dir_i:
            dir_im1 = path_split_i[i_cnt - 1]

            out_dict_i = is_attempt_dir(dir_im1)
            is_attempt_dir_i = out_dict_i["is_attempt_dir"]
            att_num_i = out_dict_i["att_num"]
            if is_attempt_dir_i:
                start_ind_to_remove = i_cnt - 1

    if start_ind_to_remove:
        path_job_root_i = path_split_i[:start_ind_to_remove]
        path_job_root_i = "/".join(path_job_root_i)

        path_job_root_w_att_rev = path_split_i[:start_ind_to_remove + 2]
        path_job_root_w_att_rev = "/".join(path_job_root_w_att_rev)

        path_job_root_w_att = path_split_i[:start_ind_to_remove + 1]
        path_job_root_w_att = "/".join(path_job_root_w_att)


        # print(compenv_out)
        # if compenv_out is not None:
        if got_compenv_from_path:
            # compenv_out = os.environ["COMPENV"]
            gdrive_path = path_job_root_w_att_rev
        else:
            gdrive_path = get_gdrive_job_path(path_job_root_w_att_rev)


        #  print("path_job_root_w_att_rev:", path_job_root_w_att_rev)
        #  print("path_job_root_i:", path_job_root_i)
    else:
        pass



    out_dict["compenv"] = compenv_out
    out_dict["path_job_root"] = path_job_root_i
    out_dict["path_job_root_w_att_rev"] = path_job_root_w_att_rev
    out_dict["att_num"] = att_num_i
    out_dict["rev_num"] = rev_num_i
    out_dict["is_rev_dir"] = is_rev_dir_i
    out_dict["is_attempt_dir"] = is_attempt_dir_i
    out_dict["path_job_root_w_att"] = path_job_root_w_att
    out_dict["gdrive_path"] = gdrive_path

    return(out_dict)
    #__|

def get_gdrive_job_path(path_job_root_w_att_rev):
    """
    Example of `path_job_root_w_att_rev` path variable:
        'dft_workflow/run_slabs/run_o_covered/out_data/dft_jobs/vh9gx4n2xg/001/01_attempt/_01'
    """
    #| - get_gdrive_job_path
    from proj_data import compenvs

    compenv_already_in_path = False
    for compenv_i in compenvs:
        if compenv_i in path_job_root_w_att_rev:
            compenv_already_in_path = True

    if not compenv_already_in_path:
        found_dft_jobs = False
        path_list = []; path_list_after = []
        for i in path_job_root_w_att_rev.split("/"):
            if not found_dft_jobs:
                path_list.append(i)
            else:
                path_list_after.append(i)
            if i == "dft_jobs":
                found_dft_jobs = True
        gdrive_path = os.path.join("/".join(path_list), compenv, "/".join(path_list_after))
        out_path = gdrive_path
    else:
        out_path = path_job_root_w_att_rev

    if "slac/slac" in out_path:
        print("out_path:", out_path)
    elif "sherlock/sherlock" in out_path:
        print("out_path:", out_path)

    return(out_path)
    #__|

#__|


#| - Magmom Setting Methods

def set_magmoms(
    atoms=None,
    mode=None,  # "set_magmoms_M_O", "set_magmoms_random"
    set_from_file_if_avail=True,
    ):
    """
    """
    #| - set_magmoms
    import json

    #| - Set magmoms from file
    my_file = Path("magmoms.json")
    if my_file.is_file():
        # #################################################
        data_path = os.path.join("magmoms.json")
        with open(data_path, "r") as fle:
            magmoms_i = json.load(fle)
        # #################################################
        if set_from_file_if_avail:
            atoms.set_initial_magnetic_moments(magmoms=magmoms_i)
            return
    #__|

    if mode == "set_magmoms_M_O":
        set_magmoms_M_O(
            atoms=atoms,
            O_magmom=0.2,
            M_magmom=1.2,
            )

    elif mode == "set_magmoms_random":
        set_magmoms_random(
            atoms=atoms,
            max_magmom=2.,
            )
    #__|

def set_magmoms_M_O(
    atoms=None,
    O_magmom=0.2,
    M_magmom=1.2,
    ):
    """
    """
    #| - set_magmoms_M_O
    magmoms_i = []
    for atom in atoms:
        if atom.symbol == "O":
            magmom_i = O_magmom
        else:
            magmom_i = M_magmom
        magmoms_i.append(magmom_i)

    # #########################################################
    atoms.set_initial_magnetic_moments(magmoms=magmoms_i)
    #__|

def set_magmoms_random(
    atoms=None,
    max_magmom=2.,
    ):
    """
    """
    #| - set_magmoms_M_O
    magmoms_i = []
    for atom in atoms:
        magmom_i = max_magmom * random.uniform(0, 1)
        magmoms_i.append(magmom_i)

    # #####################################################
    atoms.set_initial_magnetic_moments(magmoms=magmoms_i)
    #__|


#__|
