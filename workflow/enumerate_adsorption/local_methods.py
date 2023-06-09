"""
"""

#| - Import Modules
import os
import sys

# print("import pickle")
import pickle
import copy
# import copy

import numpy as np
import pandas as pd

# import plotly.graph_objects as go
import plotly.graph_objs as go
import plotly.express as px

import scipy.integrate as integrate

from pymatgen_diffusion.aimd.van_hove import RadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor

# #########################################################
from plotting.my_plotly import my_plotly_plot
# from plotting.my_plotly import my_plotly_plot

# #########################################################
from methods import (
    get_structure_coord_df,
    get_df_coord,
    # get_df_dft,
    # symmetrize_atoms,
    # remove_atoms,
    )
#__|


def mean_O_metal_coord(df_coord=None):
    """
    """
    #| - mean_O_metal_coord
    df_coord_bulk_i = df_coord
    df_i = df_coord_bulk_i[df_coord_bulk_i.element == "O"]

    def method(row_i, metal_elem=None):
        neighbor_count = row_i.neighbor_count
        elem_num = neighbor_count.get(metal_elem, None)

        return(elem_num)

    df_i["num_metal"] = df_i.apply(
        method, axis=1,
        metal_elem="Ir")

    mean_O_metal_coord = df_i.num_metal.mean()

    return(mean_O_metal_coord)
    #__|

def get_indices_of_neigh(
    active_oxy_ind=None,
    df_coord_slab_i=None,
    metal_atom_symbol=None,
    ):
    """
    Given an index of an active oxygen, use the neighbor list to find the indices of the active metal atom and then the oxygens bound to the active metal atom.

    In the future, I will expand this to go to further shells.
    """
    #| - get_indices_of_neigh
    out_dict = dict()

    neighbor_metal_indices = []
    neighbor_oxy_indices = []

    df_i = df_coord_slab_i[df_coord_slab_i.structure_index == active_oxy_ind]
    row_i = df_i.iloc[0]
    for nn_i in row_i.nn_info:
        elem_i = nn_i["site"].specie.name

        if elem_i == metal_atom_symbol:
            metal_active_site = nn_i["site_index"]
            neighbor_metal_indices.append(metal_active_site)

            df_j = df_coord_slab_i[df_coord_slab_i.structure_index == metal_active_site]
            row_j = df_j.iloc[0]

            for nn_j in row_j.nn_info:
                elem_j = nn_j["site"].specie.name
                if elem_j == "O":
                    neighbor_oxy_indices.append(nn_j["site_index"])

    # #####################################################
    shell_2_metal_atoms = []
    for nn_j in row_j.nn_info:
        df_k = df_coord_slab_i[df_coord_slab_i.structure_index == nn_j["site_index"]]
        row_k = df_k.iloc[0]

        for nn_k in row_k.nn_info:
            if nn_k["site"].specie.name == "Ir":
                shell_2_metal_atoms.append(nn_k["site_index"])
    shell_2_metal_atoms = np.sort(list(set(shell_2_metal_atoms)))

    shell_2_metal_atoms_2 = []
    for i in shell_2_metal_atoms:
        if i in neighbor_metal_indices:
            pass
        else:
            shell_2_metal_atoms_2.append(i)

    out_dict["neighbor_oxy_indices"] = neighbor_oxy_indices
    out_dict["neighbor_metal_indices"] = neighbor_metal_indices
    out_dict["shell_2_metal_atoms"] = shell_2_metal_atoms_2

    return(out_dict)
    #__|

def process_rdf(
    atoms=None,
    active_site_i=None,
    df_coord_slab_i=None,
    metal_atom_symbol=None,
    custom_name=None,
    TEST_MODE=False,
    ):
    """
    """
    #| - process_rdf
    if custom_name is None:
        custom_name = ""

    #| - Create out folders
    import os

    # directory = "out_data"
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/enumerate_adsorption",
        "out_data")

    if not os.path.exists(directory):
        os.makedirs(directory)

    # assert False, "Fix os.makedirs"

    directory = "out_plot/rdf_figures"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = "out_data/rdf_data"
    if not os.path.exists(directory):
        os.makedirs(directory)
    #__|

    AAA = AseAtomsAdaptor()
    slab_structure = AAA.get_structure(atoms)


    # Pickling data ###########################################
    out_dict = dict()
    out_dict["active_site_i"] = active_site_i
    out_dict["df_coord_slab_i"] = df_coord_slab_i
    out_dict["metal_atom_symbol"] = metal_atom_symbol

    import os; import pickle
    path_i = os.path.join(
        os.environ["HOME"],
        "__temp__",
        "temp.pickle")
    with open(path_i, "wb") as fle:
        pickle.dump(out_dict, fle)
    # #########################################################

    neigh_dict = get_indices_of_neigh(
        active_oxy_ind=active_site_i,
        df_coord_slab_i=df_coord_slab_i,
        metal_atom_symbol=metal_atom_symbol)

    neighbor_oxy_indices = neigh_dict["neighbor_oxy_indices"]
    neighbor_metal_indices = neigh_dict["neighbor_metal_indices"]
    shell_2_metal_atoms = neigh_dict["shell_2_metal_atoms"]

    neighbor_indices = neighbor_oxy_indices + neighbor_metal_indices + shell_2_metal_atoms
    # neighbor_indices = neighbor_indices[0:1]
    # print("neighbor_indices:", neighbor_indices)

    #| - Get RDF
    RDF = RadialDistributionFunction(
        [slab_structure, ],
        [active_site_i, ],
        neighbor_indices,
        # ngrid=1801,
        ngrid=4801,
        rmax=8.0,
        cell_range=2,

        # sigma=0.2,
        # sigma=0.08,
        sigma=0.015,
        # sigma=0.008,
        # sigma=0.0005,
        )

    # data_file = "out_data/rdf_data/rdf_out.csv"

    data_file = os.path.join(
        "out_data/rdf_data",
        custom_name + "_" + str(active_site_i).zfill(4) + "_" + "rdf_out.csv")
    RDF.export_rdf(data_file)
    df_rdf = pd.read_csv(data_file)

    df_rdf = df_rdf.rename(columns={" g(r)": "g"})
    #__|

    #| - Plotting
    import plotly.graph_objs as go

    x_array = df_rdf.r
    # y_array = df_rdf[" g(r)"]
    y_array = df_rdf["g"]

    trace = go.Scatter(
        x=x_array,
        y=y_array,
        )
    data = [trace]

    fig = go.Figure(data=data)
    # fig.show()

    from plotting.my_plotly import my_plotly_plot

    if TEST_MODE:
        plot_dir = "__temp__"
    else:
        plot_dir = "rdf_figures"

    out_plot_file = os.path.join(
        plot_dir,
        custom_name + "_" + str(active_site_i).zfill(4) + "_rdf")
    my_plotly_plot(
        figure=fig,
        # plot_name=str(active_site_i).zfill(4) + "_rdf",
        plot_name=out_plot_file,
        write_html=True,
        write_png=False,
        png_scale=6.0,
        write_pdf=False,
        write_svg=False,
        try_orca_write=False,
        )
    #__|

    return(df_rdf)
    #__|

def compare_rdf_ij(
    df_rdf_i=None,
    df_rdf_j=None,
    ):
    """
    """
    #| - compare_rdf_ij
    df_0 = df_rdf_i
    df_1 = df_rdf_j

    # df_0 = pd.read_csv(
    #     "out_data/rdf_data/vtc49sbtxg__011__honorupo_58_0112_rdf_out.csv")
    # # df_1 = pd.read_csv("out_data/rdf_data/vtc49sbtxg__011__honorupo_58_0114_rdf_out.csv")
    # df_1 = pd.read_csv(
    #     "out_data/rdf_data/vtc49sbtxg__011__honorupo_58_0105_rdf_out.csv")

    df_0 = df_0.rename(
        columns={df_0.columns[1]: "g0", })

    df_1 = df_1.rename(
        columns={df_1.columns[1]: "g1", })

    df_0 = df_0.set_index("r")
    df_1 = df_1.set_index("r")

    norm_0 = integrate.trapz(df_0.g0, x=df_0.index)
    df_0["g0_norm"] = df_0.g0 / norm_0

    norm_1 = integrate.trapz(df_1.g1, x=df_1.index)
    df_1["g1_norm"] = df_1.g1 / norm_1

    df_comb = pd.concat([df_0, df_1], axis=1)
    df_comb["g_diff"] = df_comb.g1_norm - df_comb.g0_norm
    df_comb["g_diff_abs"] = np.abs(df_comb["g_diff"])

    #| - Plotting
    df_i = df_comb
    x_array = df_i.index.tolist()
    y_array = df_i.g_diff_abs

    trace = go.Scatter(
        x=x_array,
        y=y_array,
        )
    data = [trace]

    fig = go.Figure(data=data)
    # fig.show()
    #__|

    df_comb_i = df_comb[df_comb.index < 10]
    integrated_diff = integrate.trapz(df_comb_i.g_diff_abs, x=df_comb_i.index)

    return(integrated_diff)
    #__|


def get_all_active_sites(
    slab=None,
    slab_id=None,
    bulk_id=None,
    df_coord_slab_i=None,
    ):
    """
    """
    #| - get_all_active_sites
    data_dict_i = dict()

    #| - Collecting df_coord objects
    if df_coord_slab_i is None:
        # #####################################################
        df_coord_slab_i = get_df_coord(slab_id=slab_id, mode="slab")

    df_coord_bulk_i = get_df_coord(bulk_id=bulk_id, mode="bulk")
    #__|

    # #########################################################
    def method(row_i, metal_elem=None):
        neighbor_count = row_i.neighbor_count
        elem_num = neighbor_count.get(metal_elem, None)
        return(elem_num)

    df_i = df_coord_bulk_i
    df_i["num_metal"] = df_i.apply(
        method, axis=1,
        metal_elem="Ir")

    df_i = df_coord_slab_i
    df_i["num_metal"] = df_i.apply(
        method, axis=1,
        metal_elem="Ir")

    # #########################################################
    # mean_O_metal_coord = mean_O_metal_coord(df_coord=df_coord_bulk_i)

    dz = 4
    positions = slab.positions

    z_min = np.min(positions[:,2])
    z_max = np.max(positions[:,2])

    # #########################################################
    active_sites = []
    for atom in slab:
        if atom.symbol == "O":
            if atom.position[2] > z_max - dz:
                df_row_i = df_coord_slab_i[
                    df_coord_slab_i.structure_index == atom.index]
                df_row_i = df_row_i.iloc[0]
                num_metal = df_row_i.num_metal

                if num_metal == 1:
                    active_sites.append(atom.index)

    data_dict_i["active_sites"] = active_sites
    data_dict_i["num_active_sites"] = len(active_sites)


    return(active_sites)
    #__|


def get_unique_active_sites(
    slab=None,
    active_sites=None,
    bulk_id=None,
    facet=None,
    slab_id=None,
    metal_atom_symbol=None,
    ):
    """
    """
    #| - get_unique_active_sites

    df_coord_slab_i = get_df_coord(slab_id=slab_id, mode="slab")
    df_coord_bulk_i = get_df_coord(bulk_id=bulk_id, mode="bulk")

    # #########################################################
    # active_sites_i = df_active_sites[df_active_sites.slab_id == slab_id]
    # active_sites_i = active_sites_i.iloc[0]
    #
    # active_sites = active_sites_i.active_sites


    # #########################################################
    custom_name_pre = bulk_id + "__" + facet + "__" + slab_id

    df_rdf_dict = dict()
    for i in active_sites:
        # print(i)

        df_rdf_i = process_rdf(
            atoms=slab,
            active_site_i=i,
            df_coord_slab_i=df_coord_slab_i,
            metal_atom_symbol=metal_atom_symbol,
            custom_name=custom_name_pre,
            )
        df_rdf_dict[i] = df_rdf_i


    # #########################################################
    diff_rdf_matrix = np.empty((len(active_sites), len(active_sites), ))
    diff_rdf_matrix[:] = np.nan
    for i_cnt, active_site_i in enumerate(active_sites):
        df_rdf_i = df_rdf_dict[active_site_i]

        for j_cnt, active_site_j in enumerate(active_sites):
            df_rdf_j = df_rdf_dict[active_site_j]

            diff_i = compare_rdf_ij(
                df_rdf_i=df_rdf_i,
                df_rdf_j=df_rdf_j,
                )

            diff_rdf_matrix[i_cnt, j_cnt] = diff_i

    # #########################################################
    df_rdf_ij = pd.DataFrame(diff_rdf_matrix, columns=active_sites)
    df_rdf_ij.index = active_sites


    # #########################################################

    active_sites_cpy = copy.deepcopy(active_sites)


    diff_threshold = 0.3
    duplicate_active_sites = []
    for active_site_i in active_sites:

        if active_site_i in duplicate_active_sites:
            continue

        for active_site_j in active_sites:
            if active_site_i == active_site_j:
                continue

            diff_ij = df_rdf_ij.loc[active_site_i, active_site_j]
            if diff_ij < diff_threshold:
                try:
                    active_sites_cpy.remove(active_site_j)
                    duplicate_active_sites.append(active_site_j)
                except:
                    pass

    active_sites_unique = active_sites_cpy

    # #########################################################
    #| - Plotting heat map
    active_sites_str = [str(i) for i in active_sites]
    fig = go.Figure(data=go.Heatmap(
        z=df_rdf_ij.to_numpy(),
        x=active_sites_str,
        y=active_sites_str,
        # type="category",
        ))

    fig["layout"]["xaxis"]["type"] = "category"
    fig["layout"]["yaxis"]["type"] = "category"

    directory = "out_plot/rdf_heat_maps_1"
    assert False, "Fix os.makedirs"
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = "rdf_heat_maps/" + custom_name_pre + "_rdf_diff_heat_map"
    my_plotly_plot(
        figure=fig,
        plot_name=file_name,
        write_html=True,
        write_png=False,
        png_scale=6.0,
        write_pdf=False,
        write_svg=False,
        try_orca_write=False,
        )
    #__|

    return(active_sites_unique)
    #__|

def return_modified_rdf(
    df_rdf=None,
    chunks_to_edit=None,
    dx=None,
    ):
    """
    """
    #| - TEMP
    df_rdf_j = df_rdf

    if type(chunks_to_edit) is not list:
        chunks_to_edit = [chunks_to_edit]
    # df_rdf_j = df_rdf_j.rename(columns={" g(r)": "g"})

    # x-axis spacing of data
    dr = df_rdf_j.r.tolist()[1] - df_rdf_j.r.tolist()[0]

    df_i = df_rdf_j[df_rdf_j.g > 1e-5]

    trace = go.Scatter(
        x=df_i.r, y=df_i.g,
        mode="markers")
    data = [trace]

    fig = go.Figure(data=data)
    my_plotly_plot(
        figure=fig,
        plot_name="temp_rds_distr",
        write_html=True)
    # fig.show()

    # chunk_coord_list = []

    chunk_start_coords = []
    chunk_end_coords = []

    row_i = df_i.iloc[0]
    chunk_start_coords.append(row_i.r)

    for i in range(1, df_i.shape[0] - 1):
        # #####################################################
        row_i = df_i.iloc[i]
        row_ip1 = df_i.iloc[i + 1]
        row_im1 = df_i.iloc[i - 1]
        # #####################################################
        r_i = row_i.r
        r_ip1 = row_ip1.r
        r_im1 = row_im1.r
        # #####################################################

        # if i == 0:
        #     chunk_coord_list.append(r_i)

        if r_i - r_im1 > 3 * dr:
            chunk_start_coords.append(r_i)

        if r_ip1 - r_i > 3 * dr:
            chunk_end_coords.append(r_i)

    # #########################################################
    row_i = df_i.iloc[-1]
    chunk_end_coords.append(row_i.r)

    chunk_coord_list = []
    for i in range(len(chunk_end_coords)):

        start_i = chunk_start_coords[i]
        end_i = chunk_end_coords[i]

        # print(
        #     str(np.round(start_i, 2)).zfill(5),
        #     str(np.round(end_i, 2)).zfill(5),
        #     )

        chunk_coord_list.append([
            start_i, end_i
            ])


    df_chunks_list = []
    for i_cnt, chunk_i in enumerate(chunk_coord_list):

        # if i_cnt == chunk_to_edit:
        if i_cnt in chunks_to_edit:
            if type(dx) == list:
                dx_tmp = dx[i_cnt]
            else:
                dx_tmp = dx
            # dx_tmp = dx
        else:
            dx_tmp = 0

        df_j = df_rdf_j[(df_rdf_j.r >= chunk_i[0]) & (df_rdf_j.r <= chunk_i[1])]
        df_j.r += dx_tmp

        df_chunks_list.append(df_j)

    import pandas as pd

    df_i = pd.concat(df_chunks_list)

    # trace = go.Scatter(
    #     x=df_i.r, y=df_i.g,
    #     mode="markers")
    # data = [trace]

    # fig = go.Figure(data=data)
    # # my_plotly_plot(
    # #     figure=fig,
    # #     plot_name="temp_rds_distr",
    # #     write_html=True)
    # fig.show()

    return(df_i)
    #__|

def create_interp_df(df_i, x_combined):
    """
    """
    #| - create_interp_df
    r_combined = x_combined

    # df_i = df_rdf_j

    tmp_list = []
    data_dict_list = []
    for r_i in r_combined:
        # print("r_i:", r_i)
        data_dict_i = dict()

        # #################################################
        min_r = df_i.r.min()
        max_r = df_i.r.max()
        # #################################################

        if r_i in df_i.r.tolist():
            row_i = df_i[df_i.r == r_i].iloc[0]
            g_new = row_i.g

        else:
            # print(r_i)
            # tmp_list.append(r_i)

            if (r_i < min_r) or (r_i > max_r):
                g_new = 0.
            else:
                # break

                from scipy.interpolate import interp1d

                inter_fun = interp1d(
                    df_i.r, df_i.g,
                    kind='linear',
                    axis=-1,
                    copy=True,
                    bounds_error=None,
                    # fill_value=None,
                    assume_sorted=False,
                    )


                g_new = inter_fun(r_i)

        data_dict_i["r"] = r_i
        data_dict_i["g"] = g_new
        data_dict_list.append(data_dict_i)

    df_tmp = pd.DataFrame(data_dict_list)

    return(df_tmp)
    #__|


























def get_unique_active_sites_temp(
    slab=None,
    active_sites=None,
    bulk_id=None,
    facet=None,
    slab_id=None,
    metal_atom_symbol=None,
    df_coord_slab_i=None,
    create_heatmap_plot=False,
    ):
    """
    """
    #| - get_unique_active_sites

    #| - __temp__
    import os
    import pickle
    #__|


    if df_coord_slab_i is None:
        df_coord_slab_i = get_df_coord(slab_id=slab_id, mode="slab")

    df_coord_bulk_i = get_df_coord(bulk_id=bulk_id, mode="bulk")


    # #########################################################
    # active_sites_i = df_active_sites[df_active_sites.slab_id == slab_id]
    # active_sites_i = active_sites_i.iloc[0]
    #
    # active_sites = active_sites_i.active_sites


    # #########################################################
    custom_name_pre = bulk_id + "__" + facet + "__" + slab_id

    df_rdf_dict = dict()
    for i in active_sites:
        # print(i)

        df_rdf_i = process_rdf(
            atoms=slab,
            active_site_i=i,
            df_coord_slab_i=df_coord_slab_i,
            metal_atom_symbol=metal_atom_symbol,
            custom_name=custom_name_pre,
            )
        df_rdf_dict[i] = df_rdf_i

    # Saving df_rdf_dict
    # Pickling data ###########################################
    # directory = "out_data/df_rdf_dict"

    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/enumerate_adsorption",
        "out_data/df_rdf_dict",
        )

    # assert False, "Fix os.makedirs"
    if not os.path.exists(directory): os.makedirs(directory)
    with open(os.path.join(directory, custom_name_pre + ".pickle"), "wb") as fle:
        pickle.dump(df_rdf_dict, fle)
    # #########################################################


    # #########################################################
    diff_rdf_matrix = np.empty((len(active_sites), len(active_sites), ))
    diff_rdf_matrix[:] = np.nan
    for i_cnt, active_site_i in enumerate(active_sites):
        df_rdf_i = df_rdf_dict[active_site_i]

        for j_cnt, active_site_j in enumerate(active_sites):
            df_rdf_j = df_rdf_dict[active_site_j]

            diff_i = compare_rdf_ij(
                df_rdf_i=df_rdf_i,
                df_rdf_j=df_rdf_j,
                )

            diff_rdf_matrix[i_cnt, j_cnt] = diff_i

    # #########################################################
    df_rdf_ij = pd.DataFrame(diff_rdf_matrix, columns=active_sites)
    df_rdf_ij.index = active_sites

    # Pickling data ###########################################
    # directory = "out_data/df_rdf_ij"

    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/enumerate_adsorption",
        "out_data/df_rdf_ij",
        )

    # assert False, "Fix os.makedirs"
    if not os.path.exists(directory): os.makedirs(directory)
    with open(os.path.join(directory, custom_name_pre + ".pickle"), "wb") as fle:
        pickle.dump(df_rdf_ij, fle)
    # #########################################################

    # #########################################################
    active_sites_cpy = copy.deepcopy(active_sites)

    diff_threshold = 0.2
    duplicate_active_sites = []
    for active_site_i in active_sites:

        if active_site_i in duplicate_active_sites:
            continue

        for active_site_j in active_sites:
            if active_site_i == active_site_j:
                continue

            diff_ij = df_rdf_ij.loc[active_site_i, active_site_j]
            if diff_ij < diff_threshold:
                try:
                    active_sites_cpy.remove(active_site_j)
                    duplicate_active_sites.append(active_site_j)
                except:
                    pass

    active_sites_unique = active_sites_cpy

    # #########################################################
    #| - Creating Figure
    # print("TEMP isjdfjsd8sfs8d")
    # print("create_heatmap_plot:", create_heatmap_plot)
    if create_heatmap_plot:
        # print("SIDJFIDISJFIDSIFI")

        import plotly.express as px
        import plotly.graph_objects as go

        active_sites_str = [str(i) for i in active_sites]
        fig = go.Figure(data=go.Heatmap(
            z=df_rdf_ij.to_numpy(),
            x=active_sites_str,
            y=active_sites_str,
            xgap=3,
            ygap=3,
            # type="category",
            ))

        fig["layout"]["xaxis"]["type"] = "category"
        fig["layout"]["yaxis"]["type"] = "category"

        # fig.show()

        # directory = "out_plot/rdf_heat_maps_1"

        directory = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/enumerate_adsorption",
            "out_data/rdf_heat_maps_1",
            )

        # assert False, "Fix os.makedirs"
        if not os.path.exists(directory):
            os.makedirs(directory)


        # from plotting.my_plotly import my_plotly_plot

        # file_name = "rdf_heat_maps_1/" + custom_name_pre + "_rdf_diff_heat_map"

        # file_name = os.path.join(
        #     "/".join(directory.split("/")[1:]),
        #     custom_name_pre + "_rdf_diff_heat_map",
        #     )

        save_dir = os.path.join(
            "/".join(directory.split("/")[1:]),
            # custom_name_pre + "_rdf_diff_heat_map",
            )

        file_name = custom_name_pre + "_rdf_diff_heat_map"

        print(file_name)
        my_plotly_plot(
            figure=fig,
            save_dir=save_dir,
            place_in_out_plot=False,
            # plot_name="rdf_heat_maps/rdf_diff_heat_map",
            plot_name=file_name,

            write_html=True,
            write_png=False,
            png_scale=6.0,
            write_pdf=False,
            write_svg=False,
            try_orca_write=False,
            )
    #__|

    return(active_sites_unique)
    #__|
