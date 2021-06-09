# TEMP

#| - Import Modules
import os
# print(os.getcwd())
# import sys
#
from pathlib import Path
# from json import dump, load
# from shutil import copyfile
#
# import plotly.graph_objs as go
import pandas as pd

# #########################################################
from jupyter_modules.jupyter_methods import (
    # clean_ipynb,
    get_ipynb_notebook_paths,
    )
#__|



def get_notebooks_run_in__jobs_update():
    """
    """
    #| - get_notebooks_run_in__jobs_update

    #| - Read file lines
    # Jobs update method in bash_methods
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "scripts/bash_methods.sh")
    with open(path_i, "r") as f:
        lines__bash_methods = f.read().splitlines()

    # Read setup_new_jobs.sh
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/run_slabs/setup_new_jobs.sh")
    with open(path_i, "r") as f:
        lines__setup_new_jobs = f.read().splitlines()
    #__|

    #| - Parsing lines inbetween the parser markers
    parse_start = "08saduijf | Parser Start"
    parse_end = "08saduijf | Parser End"

    ind_start = None
    ind_end = None
    for i_cnt, line_i in enumerate(lines__bash_methods):
        if parse_start in line_i:
            ind_start = i_cnt
        if parse_end in line_i:
            ind_end = i_cnt
    lines__jobs_update = lines__bash_methods[ind_start:ind_end]

    parse_start = "5jfds7u8ik | Parser Start"
    parse_end = "5jfds7u8ik | Parser End"

    ind_start = None
    ind_end = None
    for i_cnt, line_i in enumerate(lines__bash_methods):
        if parse_start in line_i:
            ind_start = i_cnt
        if parse_end in line_i:
            ind_end = i_cnt
    lines__feat_update = lines__bash_methods[ind_start:ind_end]

    #__|

    lines_i = lines__jobs_update + lines__setup_new_jobs + lines__feat_update

    #| - Further processing/filtering
    # # #########################################################
    # # Taking only lines with `python` but not `echo` in them
    # python_lines = []
    # for line_i in lines_i:
    #     if "python" in line_i and "echo" not in line_i:
    #         python_lines.append(line_i)

    # # #########################################################
    # # Removing white space
    # python_lines_2 = []
    # for line_i in python_lines:
    #     line_new_i = []
    #     for i in line_i.split(" "):
    #         if i != "":
    #             line_new_i.append(i)

    #     line_i = " ".join(line_new_i)
    #     python_lines_2.append(line_i)

    # # #########################################################
    # # Dispose of `python command`
    # python_lines_3 = []
    # for line_i in python_lines_2:
    #     line_new_i = line_i.split(" ")[-1]
    #     python_lines_3.append(line_new_i)
    #__|



    python_lines = []

    data_dict_list = []
    for line_i in lines_i:
        data_dict_i = dict()

        source_file_i = None
        if line_i in lines__jobs_update:
            source_file_i = "jobs_update"
        elif line_i in lines__setup_new_jobs:
            source_file_i = "setup_new_jobs"
        elif line_i in lines__feat_update:
            source_file_i = "features_update"
        else:
            print("SIJFIJDS")

        if "python" in line_i and "echo" not in line_i:
            # python_lines.append(line_i)

            line_new_i = []
            for i in line_i.split(" "):
                if i != "":
                    line_new_i.append(i)
            line_new_1_i = " ".join(line_new_i)

            line_new_2_i = line_new_1_i.split(" ")[-1]

            # #####################################################
            data_dict_i["source_file"] = source_file_i
            data_dict_i["line_0"] = line_i
            data_dict_i["line_1"] = line_new_1_i
            data_dict_i["line_2"] = line_new_2_i
            # #####################################################
            data_dict_list.append(data_dict_i)
            # #####################################################

    df_0 = pd.DataFrame(
        data_dict_list
        )


    #| - Forming dataframe
    # #########################################################
    data_dict_list = []
    # #########################################################
    for ind_i, row_i in df_0.iterrows():
    # for line_i in python_lines_3:
        # #####################################################
        data_dict_i = dict()
        # #####################################################
        line_i = row_i["line_2"]
        source_file_i = row_i.source_file
        # #####################################################

        frag_i = "$PROJ_irox_oer"
        find_ind = line_i.find(frag_i)

        path_py_i = line_i[find_ind + len(frag_i) + 1:]

        path_ipynb_i = path_py_i.split(".")[0] + ".ipynb"

        path_no_ext_i = path_py_i.split(".")[0]



        # Checking if the files exist (both .py and .ipynb)
        root_dir = os.environ["PROJ_irox_oer"]

        path_py_full_i = os.path.join(root_dir, path_py_i)
        path_ipynb_full_i = os.path.join(root_dir, path_ipynb_i)

        file_exists_py = False
        my_file = Path(path_py_full_i)
        if my_file.is_file():
            file_exists_py = True

        file_exists_ipynb = False
        my_file = Path(path_ipynb_full_i)
        if my_file.is_file():
            file_exists_ipynb = True


        # #################################################
        data_dict_i["path_py"] = path_py_i
        data_dict_i["path_ipynb"] = path_ipynb_i
        data_dict_i["path_no_ext"] = path_no_ext_i
        data_dict_i["source_file"] = source_file_i
        data_dict_i["file_exists_py"] = file_exists_py
        data_dict_i["file_exists_ipynb"] = file_exists_ipynb
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

    df_ipynb_update = pd.DataFrame(data_dict_list)
    #__|


    df_ipynb_update = df_ipynb_update.sort_values("path_ipynb")
    df_ipynb_update = df_ipynb_update.drop_duplicates("path_ipynb")

    return(df_ipynb_update)
    #__|


def get_files_exec_in_run_all():
    """
    """
    #| - get_files_exec_in_run_al

    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        # "log_files/PROJ_irox_oer__comm_overnight_wrap.log",
        "log_files/PROJ_irox_oer__comm_overnight_wrap.log.old",
        )
    with open(path_i, "r") as f:
        lines = f.read().splitlines()

    frag = "PROJ_IrOx_OER/"

    # #########################################################
    data_dict_list = []
    # #########################################################
    for line_i in lines:
        # #####################################################
        data_dict_i = dict()
        # #####################################################

        if frag in line_i:

            line_split_i = line_i.split(" ")
            for seg_j in line_split_i:
                if frag in seg_j:

                    ends_in_file = False
                    if "." in seg_j.split("/")[-1]:
                        ends_in_file = True

                        file_ext_j = seg_j.split("/")[-1].split(".")[-1]

                        allowed_ext = ["py", "ipynb", "sh"]
                        if file_ext_j in allowed_ext:
                            # seg_short_j = seg_j[48 + len(frag):]
                            seg_short_j = seg_j[
                                seg_j.find(frag) + len(frag):
                                ]

                            # #################################
                            data_dict_i["path_full"] = seg_j
                            data_dict_i["path_short"] = seg_short_j
                            # #################################
                            data_dict_list.append(data_dict_i)
                            # #################################

    # #########################################################
    df_overnight = pd.DataFrame(data_dict_list)
    df_overnight = df_overnight.sort_values("path_short")
    df_overnight = df_overnight.drop_duplicates("path_short")
    # #########################################################

    return(df_overnight)
    #__|
