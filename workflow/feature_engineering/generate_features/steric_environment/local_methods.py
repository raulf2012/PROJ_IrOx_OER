"""
"""

# | - Import Modules
import os
import pickle
import pandas as pd
# __|




# atoms=atoms
# octahedra_atoms=octahedra_atoms_i
# active_site=active_site_i
# metal_active_site=metal_active_site_i
# name=name_i
# active_site_original=active_site_orig_i
# df_coord=df_coord_i
# df_octa_info=df_octa_info
# init_or_final=init_or_final

def get_df_octa_env(
    atoms=None,
    octahedra_atoms=None,
    active_site=None,
    metal_active_site=None,
    name=None,
    active_site_original=None,
    df_coord=None,
    df_octa_info=None,
    init_or_final=None,
    ):
    """
    """
    # | - get_df_octa_env
    output = None


    run_method = True
    if octahedra_atoms is None:
        run_method = False
    elif type(octahedra_atoms) == list and len(octahedra_atoms) == 0:
        run_method = False


    # print(run_method)

    if run_method:
        output = get_df_octa_env__wrap(
            atoms=atoms,
            octahedra_atoms=octahedra_atoms,
            active_site=active_site,
            metal_active_site=metal_active_site,
            name=name,
            active_site_original=active_site_original,
            df_coord=df_coord,
            df_octa_info=df_octa_info,
            init_or_final=init_or_final,
            )

    return(output)
    # __|


# atoms=atoms
# octahedra_atoms=octahedra_atoms_i
# active_site=active_site_i
# metal_active_site=metal_active_site_i
# name=name_i
# active_site_original=active_site_orig_i
# df_coord=df_coord_i
# df_octa_info=df_octa_info
# init_or_final=init_or_final

def get_df_octa_env__wrap(
    atoms=None,
    octahedra_atoms=None,
    active_site=None,
    metal_active_site=None,
    name=None,
    active_site_original=None,
    df_coord=None,
    df_octa_info=None,
    init_or_final=None,
    ):
    """
    """
    # | - get_df_octa_env
    from pymatgen.io.ase import AseAtomsAdaptor
    structure = AseAtomsAdaptor.get_structure(atoms)


    TEST = False
    if TEST:
        print("TEST TEST TEST ")
        atoms.write("__temp__/script_dev/atoms_init.traj")
        atoms.write("__temp__/script_dev/atoms_init.cif")


    data_dict_list = []
    for atom_i in atoms:

    # if True:
    #     atom_i = atoms[66]

        # print(atom_i.index)


        if not int(atom_i.index) == int(active_site):

            if atom_i.index not in octahedra_atoms:

                dist_i, image_i = structure[
                    int(active_site)].distance_and_image(
                        structure[atom_i.index]
                        )
            else:
                tmp = 42

                dist_i, image_i = get_dist_of_atom_in_octahedra(
                    structure=structure,
                    atom_A=active_site,
                    atom_B=atom_i.index,
                    atom_M=metal_active_site,
                    df_coord=df_coord,
                    df_octa_info=df_octa_info,
                    name=name,
                    active_site_original=active_site_original,
                    init_or_final=init_or_final,
                    )


            data_dict = dict()
            data_dict["atom_index"] = atom_i.index
            data_dict["atom_symbol"] = atom_i.symbol
            data_dict["image"] = image_i
            data_dict["dist"] = dist_i
            data_dict_list.append(data_dict)

    df = pd.DataFrame(data_dict_list)
    df = df.sort_values("dist")

    return(df)
    # __|
