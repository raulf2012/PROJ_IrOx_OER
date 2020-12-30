# TEMP

#| - Import Modules
import os
import sys

import shutil
from pathlib import Path
#__|

def remove_conflicted_path_dir(path=None, remove_files_folders=False):
    """
    """
    #| - remove_conflicted_path_dir
    path_full_i = path

    if " (" in path_full_i:

        # #################################################
        found_wrong_level = False
        path_level_list = []
        for i in path_full_i.split("/"):
            if not found_wrong_level:
                path_level_list.append(i)
            if " (" in i:
                found_wrong_level = True
        path_upto_error = "/".join(path_level_list)

        my_file = Path(path_full_i)
        if my_file.is_dir():
            size_i = os.path.getsize(path_full_i)


            # If it's a small file size then it probably just has the init files and we're good to delete the dir
            # Seems that all files are 512 bytes in size (I think it's bytes)
            if size_i < 550:
                my_file = Path(path_upto_error)
                if my_file.is_dir():
                    if remove_files_folders:
                        print("Removing dir:", path_upto_error)
                        shutil.rmtree(path_upto_error)
            else:
                print(100 * "Issue | ")
                print(path_full_i)
                print(path_full_i)
                print(path_full_i)
                print(path_full_i)
                print(path_full_i)

            print("")
    #__|
