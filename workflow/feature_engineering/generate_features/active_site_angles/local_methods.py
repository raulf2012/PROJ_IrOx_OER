"""
"""

# | - Import Modules

import math

from methods import unit_vector, angle_between

# from methods import get_df_coord, get_df_coord_wrap

# __|


from methods_features import get_angle_between_surf_normal_and_O_Ir


# def get_angle_between_surf_normal_and_O_Ir(
#     atoms=None,
#     df_coord=None,
#     active_site=None,
#     ):
#     """
#     """
#     # | - get_angle_between_surf_normal_and_O_Ir
#     atoms_i = atoms
#     df_coord_i = df_coord
#     active_site_i = active_site
#
#
#     row_coord_i = df_coord_i.loc[active_site_i]
#
#     nn_Ir = None
#     for nn_j in row_coord_i.nn_info:
#         if nn_j["site"].specie.name == "Ir":
#             nn_Ir = nn_j
#
#     if nn_Ir == None:
#         return(None)
#
#     # assert nn_Ir != None, "IJDFIDJSIFDS"
#
#     Ir_coord = nn_Ir["site"].coords
#
#
#     active_O_atom = atoms_i[int(active_site_i)]
#     O_coord = active_O_atom.position
#
#     Ir_O_vec = O_coord - Ir_coord
#
#     # Ir_O_vec = [0, 1, -1]
#
#     angle_rad = angle_between(
#         unit_vector(Ir_O_vec),
#         [0, 0, 1],
#         )
#     angle_deg = math.degrees(angle_rad)
#
#
#     return(angle_deg)
#     # __|
