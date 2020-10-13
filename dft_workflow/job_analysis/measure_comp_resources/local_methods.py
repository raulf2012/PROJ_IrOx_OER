"""
"""

#| - Import Modules
import numpy as np
#__|

def calculate_loop_time_outcar(outcar_path):
    """
    """
    #| - calculate_loop_time_outcar
    out_dict = dict()

    # with open(outcar_path_i, "r") as f:
    with open(outcar_path, "r") as f:
        lines = f.read().splitlines()

    #| - Getting number of cores
    num_cores = None
    lines_tmp = []
    for line_i in lines:
        frag_i = "total cores"
        if frag_i in line_i:
            # print("r67uytgfguhgdyghujh", line_i)
            # print(line_i)
            lines_tmp.append(line_i)
    if len(lines_tmp) == 1:
        line_i = lines_tmp[0]
        # line_i = " running on   16 total cores"

        split_0 = line_i.split("  ")[-1]

        for i in split_0.split(" "):
            if i.isnumeric():
                num_cores = int(i)
                break
    #__|

    lines_2 = []
    for line_i in lines:
        frag_i = "LOOP+"
        if frag_i in line_i:
            lines_2.append(line_i)

    loop_time_list = []
    frag_j = "real time"
    for line_i in lines_2:
        tmp = 42

        # line_i = '     LOOP+:  cpu time18628.7941: real time18676.4463'

        ind_j = line_i.find(frag_j)

        line_frag_j = line_i[ind_j + len(frag_j):]

        try:
            loop_time_j = float(line_frag_j)
        except:
            loop_time_j = 0.
            print("Uh oh, no good")

        loop_time_list.append(loop_time_j)

    loop_time_sum__sec = np.sum(loop_time_list)
    loop_time_sum__min = loop_time_sum__sec / 60
    loop_time_sum__hr = loop_time_sum__min / 60

    loop_time_dict = dict()
    loop_time_dict["sec"] = loop_time_sum__sec
    loop_time_dict["min"] = loop_time_sum__min
    loop_time_dict["hr"] = loop_time_sum__hr

    # #####################################################
    out_dict["total_loop_time"] = loop_time_dict
    out_dict["loop_time_list"] = loop_time_list
    out_dict["num_cores"] = num_cores
    # #####################################################

    return(out_dict)
    #__|
