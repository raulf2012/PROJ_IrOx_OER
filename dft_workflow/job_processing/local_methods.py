"""
"""

#| - Import Modules
import os
import sys

import copy
import time
import pickle
import subprocess
from pathlib import Path

# from contextlib import contextmanager

import numpy as np

from ase import io

# #########################################################
from vasp.vasp_methods import parse_incar, read_incar
# from vasp.vasp_methods import read_incar, get_irr_kpts_from_outcar
from vasp.parse_oszicar import parse_oszicar
# from vasp.vasp_methods import

# #########################################################
from dft_job_automat.compute_env import ComputerCluster

# #########################################################
from proj_data import compenv
from methods import cwd
#__|



# #########################################################
#| - Job path processing
from dft_workflow_methods import is_attempt_dir
from dft_workflow_methods import is_rev_dir
from dft_workflow_methods import get_job_paths_info
from dft_workflow_methods import get_gdrive_job_path
#__|

# #########################################################
# | - Job parsing methods

def parse_job_err(path, compenv=None):
    """
    """
    #| - parse_job_err
    # print(path)

    status_dict = {
        "timed_out": None,
        "error": None,
        "error_type": None,
        "brmix_issue": None,
        }

    if compenv is None:
        compenv = os.environ["COMPENV"]

    # print("123jk8999df9s9")


    # | - Parsing SLAC job
    if compenv == "slac":
        job_out_file_path = os.path.join(path, "job.out")
        my_file = Path(job_out_file_path)
        if my_file.is_file():
            with open(job_out_file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:

                #| - Checking if job timed out
                phrase_i = "job killed after reaching LSF run time limit"
                if phrase_i in line:
                    status_dict["timed_out"] = True
                #__|

                #| - Checking for POSMAP symmetry error
                phrase_i = "POSMAP internal error: symmetry equivalent atom not found"
                if phrase_i in line:
                    status_dict["error"] = True
                    status_dict["error_type"] = "POSMAP symm err"
                #__|

                #| - Checking for weird mpirun exit error
                phrase_i = "mpirun has exited due to process rank"
                if phrase_i in line:
                    status_dict["error"] = True
                    status_dict["error_type"] = "mpirun has exited"
                #__|

                #| - Checking for another mpi failure error
                phrase_i = "Primary job  terminated normally, but"
                if phrase_i in line:
                    status_dict["error"] = True
                    status_dict["error_type"] = "weird SLAC cluster error, core failed (type 2)"
                #__|

                #| - Checking for mysterious sudden termination (Signal received. Exiting ...)
                phrase_i = "Signal received. Exiting ..."
                if phrase_i in line:
                    status_dict["error"] = True
                    status_dict["error_type"] = "Weird mysterious sudden termination"

                # print("TEMP | aesrtrdxytcfiuyvi7es5zdtif7gyiuo5seyardiugyobihb")
                #__|



            #| - Checking for weird cluster error
            # mpirun detected that one or more processes exited with non-zero status, thus causing
            # the job to be terminated. The first process to do so was:
            phrase_0_pres = False
            phrase_1_pres = False
            for line in lines:

                phrase_0 = "mpirun detected that one or more processes exited with non-zero status"
                if phrase_0 in line:
                    phrase_0_pres = True

                phrase_1 = "The first process to do so was"
                if phrase_1 in line:
                    phrase_1_pres = True

            if phrase_0_pres and phrase_1_pres:
                status_dict["error"] = True
                status_dict["error_type"] = "weird SLAC cluster error, core failed"
            #__|

    #__|


    # print("compenv:", compenv)

    #| - Parsing NERSC jobs
    if compenv == "nersc":
        job_out_file_path = os.path.join(path, "job.err")
        my_file = Path(job_out_file_path)
        if my_file.is_file():
            with open(job_out_file_path, 'r') as f:
                lines = f.readlines()

            # print("")
            # print("")
            # print(lines)
            # print("")
            # print("")
            #
            # print("111111111")
            for line in lines:

                #| - Checking if job timed out
                # Some of your processes may have been killed by the cgroup out-of-memory handler.
                phrase_i = "killed by the cgroup out-of-memory handler"
                if phrase_i in line:
                    status_dict["error"] = True
                    status_dict["error_type"] = "out-of-memory handler"
                #__|

    # __|


    # | - Parsing error file
    job_err_file_path = os.path.join(path, "job.err")
    my_file = Path(job_err_file_path)
    if my_file.is_file():
        with open(job_err_file_path, 'r') as f:
            lines = f.readlines()

        # else:
        for line in lines:
            if "DUE TO TIME LIMIT" in line:
                status_dict["timed_out"] = True

            if "Traceback (most recent call last):" in line:
                status_dict["error"] = True

            if "ValueError: could not convert string to float" in line:
                status_dict["error"] = True
                status_dict["error_type"] = "calculation blown up"

            if "Signal: Segmentation fault" in line:
                status_dict["error"] = True
                status_dict["error_type"] = "Segmentation fault"

            if "Unable to confirm allocation for job" in line:
                status_dict["error"] = True
                status_dict["error_type"] = "slurm allocation issue"

    #__|


    # print("KJSDFJ(SDi787rt676tfg76t78ty897t7)")

    # | - Parsing out file

    #| - old parser here, keeping for now
    if status_dict["error"] is True:
        job_out_file_path = os.path.join(path, "job.out")
        my_file = Path(job_out_file_path)
        if my_file.is_file():
            # print("Going to attempt to open the file here")
            with open(job_out_file_path, "r") as f:
                lines = f.readlines()
                # print("If you see this then the file was read ok")

            # print("len(lines):", len(lines))

            for line in lines:
                err_i = "VERY BAD NEWS! internal error in subroutine SGRCON:"
                if err_i in line:
                    status_dict["error_type"] = "Error in SGRCON (symm error)"
                    break
    #__|


    # print("98998887876767")

    my_file_0 = Path(os.path.join(path, "job.out"))
    my_file_1 = Path(os.path.join(path, "job.out.short"))
    if my_file_0.is_file():
        job_out_file = my_file_0
    elif my_file_1.is_file():
        job_out_file = my_file_1
    else:
        job_out_file = None

    # print("098765454543432")


    if job_out_file is not None:
        # print("Going to attempt to open the file here")

        with open(job_out_file, "r") as f:
            for line in f:
                err_i = "BRMIX: very serious problems"
                if err_i in line:
                    status_dict["brmix_issue"] = True
                    status_dict["error"] = True


        # with open(job_out_file, 'r') as f:
        #     lines = f.readlines()
        #     print("DONE!!!!")
        # print("len(lines):", len(lines))
        # #| - Checking for BRMIX error
        # for line in lines:
        #     err_i = "BRMIX: very serious problems"
        #     if err_i in line:
        #         status_dict["brmix_issue"] = True
        #         status_dict["error"] = True
        # #__|

    #__|

    return(status_dict)
    #__|

def parse_finished_file(path):
    """
    """
    #| - parse_finished_file
    status_dict = {
        "completed": False,
        }

    job_err_file_path = os.path.join(path, ".FINISHED")
    my_file = Path(job_err_file_path)
    if my_file.is_file():
        status_dict["completed"] = True

    return(status_dict)
    #__|

def parse_job_state(path):
    """
    """
    #| - parse_job_state
    CC = ComputerCluster()
    job_state = CC.cluster.job_state(path_i=path)

    return({"job_state": job_state})
    #__|

def is_job_submitted(path):
    """
    """
    #| - is_job_submitted
    status_dict = {
        "submitted": False,
        }

    submitted_file_path = os.path.join(path, ".SUBMITTED")
    my_file = Path(submitted_file_path)
    if my_file.is_file():
        status_dict["submitted"] = True
    return(status_dict)
    #__|

def is_job_started(path):
    """
    """
    #| - is_job_submitted
    status_dict = {
        "job_started": False,
        }

    file_path = os.path.join(path, "job.out")
    my_file = Path(file_path)
    if my_file.is_file():
        status_dict["job_started"] = True
    return(status_dict)
    #__|

def get_isif_from_incar(path):
    """
    """
    #| - get_isif_from_incar
    isif = None

    incar_file_path = os.path.join(path, "INCAR")
    my_file = Path(incar_file_path)
    if my_file.is_file():
        with open(incar_file_path, 'r') as f:
            lines = f.readlines()

            isif = parse_incar(lines)["ISIF"]
    return({"isif": isif})
    #__|

def get_number_of_ionic_steps(path):
    """
    """
    #| - get_number_of_ionic_steps
    status_dict = {"num_steps": None}

    outcar_path = os.path.join(path, "OUTCAR")
    try:
        traj = io.read(outcar_path, index=":")
        status_dict["num_steps"] = len(traj)
    except:
        status_dict["num_steps"] = 0
        pass

    return(status_dict)
    #__|

def analyze_oszicar(path):
    """
    """
    #| - analyze_oszicar
    def ediff_conv_reached(df_scf, ediff=None):
        """
        """
        #| - ediff_conv_reached

        row_last = df_scf.iloc[-1]
        # row_i = row_last

        dE_i = row_last.dE
        if np.abs(dE_i) < ediff:
            # print("Last dE is less than ediff")
            ediff_conv_reached = True
        else:
            # print("Last dE does not reach required ediff convergence")
            ediff_conv_reached = False

        return(ediff_conv_reached)
        #__|

    path_i = path

    out_dict = parse_oszicar(vasp_dir=path_i)

    if out_dict is not None:
        #| - __temp__
        ion_step_conv_dict = out_dict["ion_step_conv_dict"]
        num_N_dict = out_dict["num_N_dict"]
        N_tot = out_dict["N_tot"]
        if N_tot is not None:
            N_tot = int(N_tot)

        # #########################################################
        keys_list = list(ion_step_conv_dict.keys())
        num_scf_cycles = len(keys_list)


        # #########################################################
        # Read INCAR file
        incar_dict = read_incar(path_i)
        if incar_dict is None:
            ediff = 2E-5
        else:
            ediff = incar_dict["EDIFF"]


        # #########################################################
        # Checking last SCF cycle for convergence
        # df_scf_last = ion_step_conv_dict[keys_list[-1]]
        # ediff_conv_reached_last = ediff_conv_reached(df_scf_last, ediff=ediff)
        if len(ion_step_conv_dict) > 0:
            df_scf_last = ion_step_conv_dict[keys_list[-1]]
            ediff_conv_reached_last = ediff_conv_reached(df_scf_last, ediff=ediff)
        else:
            ediff_conv_reached_last = None


        # #########################################################
        # Checking ediff convergence of all scf cycles
        ediff_conv_reached_dict = dict()
        for key, df_scf_i in ion_step_conv_dict.items():
            ediff_conv_reached_i = ediff_conv_reached(df_scf_i, ediff=ediff)
            ediff_conv_reached_dict[key] = ediff_conv_reached_i


        #| - Getting ratio of successful to unsuccessful scf runs
        # ediff_conv_reached_dict = ion_step_conv_dict
        # (list(ediff_conv_reached_dict.values()))

        true_list = [i for i in list(ediff_conv_reached_dict.values()) if i == True]
        num_true = len(true_list)

        false_list = [i for i in list(ediff_conv_reached_dict.values()) if i == False]
        num_false = len(false_list)

        if num_false == 0:
            true_false_ratio = 9999
        else:
            true_false_ratio = num_true / num_false

        # If both num_true and num_false are 0, then just return None
        if (num_true + num_false) > 0:
            frac_true = num_true / (num_true + num_false)
            frac_false = num_false / (num_true + num_false)
        else:
            frac_true = None
            frac_false = None
        #__|


        # #########################################################
        # Preparing out_dict
        out_dict = dict(
            ediff_conv_reached=ediff_conv_reached_last,
            ediff_conv_reached_dict=ediff_conv_reached_dict,
            num_scf_cycles=num_scf_cycles,
            N_tot=N_tot,
            true_false_ratio=true_false_ratio,
            frac_true=frac_true,
            frac_false=frac_false,
            num_nonconv_scf=num_false,
            num_conv_scf=num_true,
            )

        #__|
    else:

        # #########################################################
        # Preparing out_dict
        out_dict = dict(
            ediff_conv_reached=None,
            ediff_conv_reached_dict=None,
            num_scf_cycles=None,
            N_tot=None,
            true_false_ratio=None,
            num_nonconv_scf=None,
            num_conv_scf=None,
            )
    return(out_dict)
    #__|

def read_data_pickle(path=None):
    """
    """
    #| - read_data_pickle
    out_dict = dict(
        run_start_time=None,
        time_str=None,
        run_time=None,
        )

    # #########################################################
    file_path_i = os.path.join(
        path,
        "out_data/pre_out_data.pickle")
    my_file = Path(file_path_i)
    if my_file.is_file():
        #| - Read pre_out_data
        # #####################################################
        with open(file_path_i, "rb") as fle:
            pickle_data = pickle.load(fle)
        # #####################################################

        atoms = pickle_data.get("atoms", None)
        dft_calc_settings = pickle_data.get("dft_calc_settings", None)
        VP = pickle_data.get("VP", None)
        calc = pickle_data.get("calc", None)
        local_dft_settings = pickle_data.get("local_dft_settings", None)
        t0 = pickle_data.get("t0", None)

        # #################################################
        # time_i = time.localtime(t0[0])
        time_i = time.localtime(t0)
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time_i)


        # #################################################
        out_dict["run_start_time"] = time_i
        out_dict["time_str"] = time_str
        #__|



    # #########################################################
    # #########################################################
    # #########################################################



    # #########################################################
    file_path_i = os.path.join(
        path,
        "out_data/out_data.pickle")
    my_file = Path(file_path_i)
    if my_file.is_file():
        #| - Read out_data
        # #####################################################
        with open(file_path_i, "rb") as fle:
            pickle_data = pickle.load(fle)
        # #####################################################

        # tmp = [print(i) for i in list(pickle_data.keys())]

        # #########################################################
        atoms = pickle_data.get("atoms", None)
        dft_calc_settings = pickle_data.get("dft_calc_settings", None)
        VP = pickle_data.get("VP", None)
        calc = pickle_data.get("calc", None)
        local_dft_settings = pickle_data.get("local_dft_settings", None)
        run_time = pickle_data.get("run_time", None)
        t0 = pickle_data.get("t0", None)
        tf = pickle_data.get("tf", None)

        # #########################################################
        out_dict["run_time"] = run_time
        #__|

    return(out_dict)
    #__|

def get_final_atoms(path):
    """
    """
    #| - get_final_atoms
    atoms = None

    file_path_i = os.path.join(
        path, "final_with_calculator.traj")
    my_file = Path(file_path_i)
    if my_file.is_file():
        atoms = io.read(file_path_i)

    return(atoms)
    #__|

def get_init_atoms(path):
    """
    """
    #| - get_final_atoms
    atoms = None

    file_path_i = os.path.join(
        path, "init.traj")
    my_file = Path(file_path_i)
    if my_file.is_file():
        atoms = io.read(file_path_i)

    return(atoms)
    #__|

def get_magmoms_from_job(path_i):
    """
    """
    #| -  get_magmoms_from_job
    incar_params = read_incar(path_i)
    if incar_params is not None:
        ispin_i = incar_params["ISPIN"]

        path_i = os.path.join(
            path_i,
            "OUTCAR")

        magmoms_i = None
        if ispin_i == 2:
            try:
                atoms = io.read(path_i)
                atoms_read_properly = True
            except:
                atoms_read_properly = False

            if atoms_read_properly:
                magmoms_i = atoms.get_magnetic_moments()
    else:
        magmoms_i = None

    return(magmoms_i)
    #__|

def get_ads_from_path(path_i):
    """
    """
    #| - get_ads_from_path
    ads_i = None
    # if "run_o_covered" in path_rel_to_proj_i:
    if "run_o_covered" in path_i:
        ads_i = "o"
    elif "run_bare_oh_covered" in path_i:
        if "/bare/" in path_i:
            ads_i = "difjsi"
        # elif "/oh"

    elif "difjsi" in path_i:
        ads_i = "difjsi"


    return(ads_i)
    #__|

def get_forces_info(path_i):
    """
    """
    #| - get_forces_info
    from ase_modules.ase_methods import max_force
    from pathlib import Path

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["force_largest"] = None
    out_dict["force_sum"] = None
    # #####################################################


    full_path_i = os.path.join(path_i, "final_with_calculator.traj")
    my_file = Path(full_path_i)
    if my_file.is_file():
        atoms = io.read(full_path_i)

        force_largest, force_sum = max_force(atoms)

        num_atoms = atoms.get_global_number_of_atoms()

        force_sum_per_atom = force_sum / num_atoms

        out_dict["force_largest"] = force_largest
        out_dict["force_sum"] = force_sum
        out_dict["force_sum_per_atom"] = force_sum_per_atom
        # #################################################

    return(out_dict)
    # __|

#__|

# #########################################################
#| - Job cleanup and processing

def process_large_job_out(path_i, job_out_size_limit=10):
    """
    """
    #| - process_large_job_out
    proj_dir = os.environ.get("PROJ_irox_oer", None)

    bash_script_path = os.path.join(
        proj_dir,
        "dft_workflow/job_processing",
        "shorten_job_out.sh")

    if "job.out.short" not in os.listdir(path=path_i):
        Path_i = Path(os.path.join(path_i, "job.out"))
        if Path_i.is_file():
            size_mb_i = Path_i.stat().st_size / 1000 / 1000
            if size_mb_i > job_out_size_limit:
                with cwd(path_i):
                    result = subprocess.run(
                        [bash_script_path],
                        stdout=subprocess.PIPE)
                    out_lines = result.stdout.splitlines()

    else:
        # Already processed
        pass
    #__|

def rclone_sync_job_dir(
    path_job_root_w_att_rev=None,
    path_rel_to_proj=None,
    verbose=True,
    ):
    """
    """
    #| - rclone_sync_job_dir
    # #########################################################
    gdrive_path = get_gdrive_job_path(path_job_root_w_att_rev)

    # #########################################################
    rclone_gdrive_stanford = os.environ["rclone_gdrive_stanford"]
    PROJ_irox_oer_gdrive = os.environ["PROJ_irox_oer_gdrive"]
    gdrive = os.environ["gdrive"]

    # #########################################################
    ind_i = PROJ_irox_oer_gdrive.find(gdrive)
    PROJ_irox_oer_gdrive_rel = PROJ_irox_oer_gdrive[ind_i + len(gdrive) + 1:]

    # #########################################################
    gdrive_path_full = os.path.join(
        PROJ_irox_oer_gdrive_rel,
        gdrive_path)



    # #########################################################
    vasp_files_to_exclude = [
        "CHG",
        "CHGCAR",
        "WAVECAR",
        # "",
        ]

    exclude_str = ""
    for file_i in vasp_files_to_exclude:
            exc_str_i = " --exclude " + file_i
            exclude_str += exc_str_i

    # rclone_flags = "--transfers=40 --checkers=40 --verbose --tpslimit=10 --max-size=100M "
    # rclone_flags = "--transfers=40 --checkers=40 --verbose --tpslimit=10 "
    # rclone_flags = "--transfers=40 --checkers=40 --tpslimit=10 "
    rclone_flags = "--transfers=80 --checkers=80 --tpslimit=80 "

    if verbose:
        rclone_flags += "--verbose "


    # Adding max size filter
    rclone_flags += "--max-size 130M "

    local_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        path_rel_to_proj)

    rclone_comm_i = "rclone copy " + rclone_flags + local_path + " " + rclone_gdrive_stanford + ":" + gdrive_path_full + " " + exclude_str

    print(40 * "*")
    print(rclone_comm_i)
    print(path_rel_to_proj)
    print(40 * "*")

    # #########################################################
    rclone_comm_list_i = [i for i in rclone_comm_i.split(" ") if i != ""]

    result = subprocess.run(
        rclone_comm_list_i,
        stdout=subprocess.PIPE)

    #__|

#__|



def local_dir_matches_remote(
    path_i=None,
    gdrive_path_i=None,
    ):
    """
    """
    #| - local_dir_matches_remote
    # #########################################################
    # Getting rclone files list
    gdrive_root = "norskov_research_storage/00_projects/PROJ_irox_oer"

    rclone_name = os.environ["rclone_gdrive_stanford"]

    bash_comm = "rclone lsf " + rclone_name + ":"

    bash_comm_i = bash_comm + os.path.join(gdrive_root, gdrive_path_i)

    bash_out = subprocess.check_output(bash_comm_i.split(" "))
    files_dir_list = bash_out.decode('UTF-8').splitlines()

    files_dir_list_new = []
    for i in files_dir_list:
        if i[-1] == "/":
            i_new = i[0:-1]
        else:
            i_new = i
        files_dir_list_new.append(i_new)

    files_dir_list_new = np.sort(files_dir_list_new)

    # #########################################################
    # Local files  list
    files_list_local = os.listdir(path_i)
    files_list_local = np.sort(files_list_local)








    # #####################################################
    # files_list_local = ["A", "B", "C", "D", ]
    # local_matches_remote = ["A", "B", "C", "d", ]

    files_list_local_cpy = list(copy.deepcopy(files_list_local))
    #  local_matches_remote_cpy = copy.deepcopy(local_matches_remote)
    files_dir_list_new_cpy = list(copy.deepcopy(files_dir_list_new))

    shared_entries = list(
        set(files_list_local_cpy) & set(files_dir_list_new_cpy))

    # print("shared_entries:", shared_entries)

    for i in shared_entries:
        files_list_local_cpy.remove(i)
        files_dir_list_new_cpy.remove(i)

    # print("Files only on local:", files_list_local_cpy)
    # print("Files only in remote:", files_dir_list_new_cpy)

    # Files only on local: ['CHG', 'CHGCAR', 'WAVECAR']
    # Files only in remote: ['contcar_out.traj']

    # Files that are expected to be only on the remote
    expected_remote_files = ["CHG", "CHGCAR", "WAVECAR", ".dft_clean", "__misc__", "$SLURM_SUBMIT_DIR"]
    for i in expected_remote_files:
        if i in files_list_local_cpy:
            files_list_local_cpy.remove(i)

    # Files that are expected to exist only locally
    expected_local_files = ["contcar_out.traj", "dft-params.json"]
    for i in expected_local_files:
        if i in files_dir_list_new_cpy:
            files_dir_list_new_cpy.remove(i)

    # Remove .swp files from lists
    for i in files_dir_list_new_cpy:
        if ".swp" in i:
            files_dir_list_new_cpy.remove(i)
    for i in files_list_local_cpy:
        if ".swp" in i:
            files_list_local_cpy.remove(i)
    # #####################################################

    local_dir_matches_remote = False

    if len(files_list_local_cpy) == 0 and len(files_dir_list_new_cpy) == 0:
        local_dir_matches_remote = True

    # Sometimes the job.out gets synced to remote before job finishes and then locally it's tranformed into job.out.short
    if len(files_dir_list_new_cpy) == 1 and files_dir_list_new_cpy[0] == "job.out":
        # print(40 * "THIS IS NEW | ")
        local_dir_matches_remote = True


    if not local_dir_matches_remote:
        print("Files only on local:", files_list_local_cpy)
        print("Files only in remote:", files_dir_list_new_cpy)

    return(local_dir_matches_remote)
    #__|

def get_num_revs_for_group(
    group=None,
    ):
    """
    """
    #| - get_num_revs_for_group
    # group = grouped.get_group(name_i)

    rev_nums = group.rev_num.tolist()

    # The rev_nums should be unique anyways, but just checking
    unique_rev_nums = list(set(rev_nums))

    mess_i = "iksdjdfisdidfj45oigfdfghjk"
    assert len(unique_rev_nums) == len(rev_nums), mess_i

    unique_rev_nums = np.sort(unique_rev_nums)

    all_consecutive = all(np.diff(unique_rev_nums) == 1)
    mess_i = "Not all consecutive, all diffs should be 1"
    assert all_consecutive, mess_i

    num_revs = np.max(unique_rev_nums)

    return(num_revs)
    #__|
