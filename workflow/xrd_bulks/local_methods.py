# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
This module implements an XRD pattern calculator.

Raul A. Flores | 20200901 | I stole this from pymatgen
"""

#| - Import Modules
import os
import json
from math import sin, cos, asin, pi, degrees, radians

import numpy as np
import pandas as pd

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# from .core import DiffractionPattern, AbstractDiffractionPatternCalculator, \
#     get_unique_families

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.core import (
    DiffractionPattern,
    AbstractDiffractionPatternCalculator,
    get_unique_families,
    )
#__|


#| - Info
__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "ongsp@ucsd.edu"
__date__ = "5/22/14"
#__|

#| - XRD wavelengths in angstroms
WAVELENGTHS = {
    "CuKa": 1.54184,
    "CuKa2": 1.54439,
    "CuKa1": 1.54056,
    "CuKb1": 1.39222,
    "MoKa": 0.71073,
    "MoKa2": 0.71359,
    "MoKa1": 0.70930,
    "MoKb1": 0.63229,
    "CrKa": 2.29100,
    "CrKa2": 2.29361,
    "CrKa1": 2.28970,
    "CrKb1": 2.08487,
    "FeKa": 1.93735,
    "FeKa2": 1.93998,
    "FeKa1": 1.93604,
    "FeKb1": 1.75661,
    "CoKa": 1.79026,
    "CoKa2": 1.79285,
    "CoKa1": 1.78896,
    "CoKb1": 1.63079,
    "AgKa": 0.560885,
    "AgKa2": 0.563813,
    "AgKa1": 0.559421,
    "AgKb1": 0.497082,
}
#__|


with open(os.path.join(os.path.dirname(__file__),
                       "in_data/atomic_scattering_params.json")) as f:
                       #  "atomic_scattering_params.json")) as f:
    ATOMIC_SCATTERING_PARAMS = json.load(f)


class XRDCalculator(AbstractDiffractionPatternCalculator):
    #| - Header
    r"""
    Computes the XRD pattern of a crystal structure.

    This code is implemented by Shyue Ping Ong as part of UCSD's NANO106 -
    Crystallography of Materials. The formalism for this code is based on
    that given in Chapters 11 and 12 of Structure of Materials by Marc De
    Graef and Michael E. McHenry. This takes into account the atomic
    scattering factors and the Lorentz polarization factor, but not
    the Debye-Waller (temperature) factor (for which data is typically not
    available). Note that the multiplicity correction is not needed since
    this code simply goes through all reciprocal points within the limiting
    sphere, which includes all symmetrically equivalent facets. The algorithm
    is as follows

    1. Calculate reciprocal lattice of structure. Find all reciprocal points
       within the limiting sphere given by :math:`\\frac{2}{\\lambda}`.

    2. For each reciprocal point :math:`\\mathbf{g_{hkl}}` corresponding to
       lattice plane :math:`(hkl)`, compute the Bragg condition
       :math:`\\sin(\\theta) = \\frac{\\lambda}{2d_{hkl}}`

    3. Compute the structure factor as the sum of the atomic scattering
       factors. The atomic scattering factors are given by

       .. math::

           f(s) = Z - 41.78214 \\times s^2 \\times \\sum\\limits_{i=1}^n a_i \
           \\exp(-b_is^2)

       where :math:`s = \\frac{\\sin(\\theta)}{\\lambda}` and :math:`a_i`
       and :math:`b_i` are the fitted parameters for each element. The
       structure factor is then given by

       .. math::

           F_{hkl} = \\sum\\limits_{j=1}^N f_j \\exp(2\\pi i \\mathbf{g_{hkl}}
           \\cdot \\mathbf{r})

    4. The intensity is then given by the modulus square of the structure
       factor.

       .. math::

           I_{hkl} = F_{hkl}F_{hkl}^*

    5. Finally, the Lorentz polarization correction factor is applied. This
       factor is given by:

       .. math::

           P(\\theta) = \\frac{1 + \\cos^2(2\\theta)}
           {\\sin^2(\\theta)\\cos(\\theta)}
    """
    #__|

    #| - XRDCalculator

    # Tuple of available radiation keywords.
    AVAILABLE_RADIATION = tuple(WAVELENGTHS.keys())

    def __init__(self, wavelength="CuKa", symprec=0, debye_waller_factors=None):
        """
        Initializes the XRD calculator with a given radiation.

        Args:
            wavelength (str/float): The wavelength can be specified as either a
                float or a string. If it is a string, it must be one of the
                supported definitions in the AVAILABLE_RADIATION class
                variable, which provides useful commonly used wavelengths.
                If it is a float, it is interpreted as a wavelength in
                angstroms. Defaults to "CuKa", i.e, Cu K_alpha radiation.
            symprec (float): Symmetry precision for structure refinement. If
                set to 0, no refinement is done. Otherwise, refinement is
                performed using spglib with provided precision.
            debye_waller_factors ({element symbol: float}): Allows the
                specification of Debye-Waller factors. Note that these
                factors are temperature dependent.
        """
        #| - __init__
        if isinstance(wavelength, float):
            self.wavelength = wavelength
        else:
            self.radiation = wavelength
            self.wavelength = WAVELENGTHS[wavelength]
        self.symprec = symprec
        self.debye_waller_factors = debye_waller_factors or {}
        #__|

    def get_pattern(self, structure, scaled=True, two_theta_range=(0, 90)):
        """
        Calculates the diffraction pattern for a structure.

        Args:
            structure (Structure): Input structure
            scaled (bool): Whether to return scaled intensities. The maximum
                peak is set to a value of 100. Defaults to True. Use False if
                you need the absolute values to combine XRD plots.
            two_theta_range ([float of length 2]): Tuple for range of
                two_thetas to calculate in degrees. Defaults to (0, 90). Set to
                None if you want all diffracted beams within the limiting
                sphere of radius 2 / wavelength.

        Returns:
            (XRDPattern)
        """
        #| - get_pattern
        if self.symprec:
            finder = SpacegroupAnalyzer(structure, symprec=self.symprec)
            structure = finder.get_refined_structure()

        wavelength = self.wavelength
        latt = structure.lattice
        is_hex = latt.is_hexagonal()

        # Obtained from Bragg condition. Note that reciprocal lattice
        # vector length is 1 / d_hkl.
        min_r, max_r = (0, 2 / wavelength) if two_theta_range is None else \
            [2 * sin(radians(t / 2)) / wavelength for t in two_theta_range]

        # Obtain crystallographic reciprocal lattice points within range
        recip_latt = latt.reciprocal_lattice_crystallographic
        recip_pts = recip_latt.get_points_in_sphere(
            [[0, 0, 0]], [0, 0, 0], max_r)
        if min_r:
            recip_pts = [pt for pt in recip_pts if pt[1] >= min_r]

        # Create a flattened array of zs, coeffs, fcoords and occus. This is
        # used to perform vectorized computation of atomic scattering factors
        # later. Note that these are not necessarily the same size as the
        # structure as each partially occupied specie occupies its own
        # position in the flattened array.
        zs = []
        coeffs = []
        fcoords = []
        occus = []
        dwfactors = []

        for site in structure:
            for sp, occu in site.species.items():
                zs.append(sp.Z)
                try:
                    c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
                except KeyError:
                    raise ValueError("Unable to calculate XRD pattern as "
                                     "there is no scattering coefficients for"
                                     " %s." % sp.symbol)
                coeffs.append(c)
                dwfactors.append(self.debye_waller_factors.get(sp.symbol, 0))
                fcoords.append(site.frac_coords)
                occus.append(occu)

        zs = np.array(zs)
        coeffs = np.array(coeffs)
        fcoords = np.array(fcoords)
        occus = np.array(occus)
        dwfactors = np.array(dwfactors)
        peaks = {}
        two_thetas = []

        for hkl, g_hkl, ind, _ in sorted(
                recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])):
            # Force miller indices to be integers.
            hkl = [int(round(i)) for i in hkl]
            if g_hkl != 0:

                d_hkl = 1 / g_hkl

                # Bragg condition
                theta = asin(wavelength * g_hkl / 2)

                # s = sin(theta) / wavelength = 1 / 2d = |ghkl| / 2 (d =
                # 1/|ghkl|)
                s = g_hkl / 2

                # Store s^2 since we are using it a few times.
                s2 = s ** 2

                # Vectorized computation of g.r for all fractional coords and
                # hkl.
                g_dot_r = np.dot(fcoords, np.transpose([hkl])).T[0]

                # Highly vectorized computation of atomic scattering factors.
                # Equivalent non-vectorized code is::
                #
                #   for site in structure:
                #      el = site.specie
                #      coeff = ATOMIC_SCATTERING_PARAMS[el.symbol]
                #      fs = el.Z - 41.78214 * s2 * sum(
                #          [d[0] * exp(-d[1] * s2) for d in coeff])
                fs = zs - 41.78214 * s2 * np.sum(
                    coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1)

                dw_correction = np.exp(-dwfactors * s2)

                # Structure factor = sum of atomic scattering factors (with
                # position factor exp(2j * pi * g.r and occupancies).
                # Vectorized computation.
                f_hkl = np.sum(fs * occus * np.exp(2j * pi * g_dot_r)
                               * dw_correction)

                # Lorentz polarization correction for hkl
                lorentz_factor = (1 + cos(2 * theta) ** 2) / \
                    (sin(theta) ** 2 * cos(theta))

                # Intensity for hkl is modulus square of structure factor.
                i_hkl = (f_hkl * f_hkl.conjugate()).real

                two_theta = degrees(2 * theta)

                if is_hex:
                    # Use Miller-Bravais indices for hexagonal lattices.
                    hkl = (hkl[0], hkl[1], - hkl[0] - hkl[1], hkl[2])
                # Deal with floating point precision issues.
                ind = np.where(np.abs(np.subtract(two_thetas, two_theta)) <
                               AbstractDiffractionPatternCalculator.TWO_THETA_TOL)
                if len(ind[0]) > 0:
                    peaks[two_thetas[ind[0][0]]][0] += i_hkl * lorentz_factor
                    peaks[two_thetas[ind[0][0]]][1].append(tuple(hkl))
                else:
                    peaks[two_theta] = [i_hkl * lorentz_factor, [tuple(hkl)],
                                        d_hkl]
                    two_thetas.append(two_theta)

        # Scale intensities so that the max intensity is 100.
        max_intensity = max([v[0] for v in peaks.values()])
        x = []
        y = []
        hkls = []
        d_hkls = []
        for k in sorted(peaks.keys()):
            v = peaks[k]
            fam = get_unique_families(v[1])
            if v[0] / max_intensity * 100 > AbstractDiffractionPatternCalculator.SCALED_INTENSITY_TOL:
                x.append(k)
                y.append(v[0])
                hkls.append([{"hkl": hkl, "multiplicity": mult}
                             for hkl, mult in fam.items()])
                d_hkls.append(v[2])




        #| -  Saving variables for testing
        # tmp = 42
        # print("OISJIFjsidfifsjdifjsdds87ha678987husd")
        self.tmp = 42
        self.x = x
        self.y = y
        self.hkls = hkls
        self.d_hkls = d_hkls
        #__|


        xrd = DiffractionPattern(x, y, hkls, d_hkls)
        if scaled:
            xrd.normalize(mode="max", value=100)
        return xrd
        #__|

    #__|


def get_top_xrd_facets(
    atoms=None,
    # structure=None,
    ):
    """
    """
    #| -  get_top_xrd_facets
    # COMBAK Delete later
    # struct_i = structure
    # atoms_i = atoms

    AAA = AseAtomsAdaptor()
    struct_i = AAA.get_structure(atoms)

    XRDCalc = XRDCalculator(
        wavelength='CuKa',
        symprec=0,
        debye_waller_factors=None,
        )

    tmp = XRDCalc.get_pattern(
        struct_i,
        scaled=True,
        two_theta_range=(0, 90),
        )

    x = XRDCalc.x
    y = XRDCalc.y
    hkls = XRDCalc.hkls
    d_hkls = XRDCalc.d_hkls

    data_dict_list = []
    zipped = zip(x, y, hkls, d_hkls)
    for x_i, y_i, hkls_i, d_hkls_i in zipped:
        data_dict_i = dict()

        # print(hkls_i)

        facets_list = []
        multiplicities_list = []
        for j in hkls_i:
            facet_j = j["hkl"]
            multiplicity_j = j["multiplicity"]
            facets_list.append(facet_j)
            multiplicities_list.append(multiplicity_j)

        # #####################################################
        data_dict_i["x"] = x_i
        data_dict_i["y"] = y_i
        data_dict_i["d_hkls"] = d_hkls_i
        data_dict_i["facets"] = facets_list
        data_dict_i["multiplicities"] = multiplicities_list
        # #####################################################
        data_dict_list.append(data_dict_i)
        # #####################################################

    df_xrd_i = pd.DataFrame(data_dict_list)

    df_xrd_j = df_xrd_i.sort_values("y", ascending=False)

    df_xrd_i = df_xrd_i.sort_values("x", ascending=True)
    df_xrd_i["y_norm"] = 100 * (df_xrd_i.y / df_xrd_i.y.max())




    # return(df_xrd_is)
    # print("0000")

    #| - Get unique peaks
    # #########################################################
    duplicate_peaks = []
    # #########################################################
    for ind_i, row_i in df_xrd_i.iterrows():

        if ind_i in duplicate_peaks:
            # print("IJDIFJIDSIFJ")
            continue

        if ind_i in duplicate_peaks:
            print("drt67nmjkio")

        # #####################################################
        x_i = row_i.x
        y_norm_i = row_i.y_norm
        # #####################################################

        df_xrd_local = df_xrd_i[
            (df_xrd_i.x < x_i + 0.1) & \
            (df_xrd_i.x > x_i - 0.1) & \
            [True for i in range(len(df_xrd_i))]
            ]

        df_xrd_local = df_xrd_local.drop(ind_i)
        for ind_j, row_j in df_xrd_local.iterrows():
            y_norm_j = row_j.y_norm

            diff_y_ij = np.abs(y_norm_i - y_norm_j)
            if diff_y_ij < 5:
                duplicate_peaks.append(ind_j)

    # #########################################################
    df_xrd_unique = df_xrd_i.drop(duplicate_peaks)
    df_xrd_unique = df_xrd_unique.sort_values("y_norm", ascending=False)

    #__|

    # print("1111")

    #| - Eliminating facets that are equivalent

    # #####################################################
    indices_to_drop = []
    # #####################################################
    for ind_i, row_i in df_xrd_unique.iterrows():

        # #################################################
        facets_i = row_i.facets
        # #################################################

        for facet_j in facets_i:

            for ind_k, row_k in df_xrd_unique.iterrows():

                # #########################################
                facets_k = row_k.facets
                # #########################################

                for facet_l in facets_k:

                    if facet_j == facet_l:
                        continue
                    else:
                        duplicate_facet_found = \
                            compare_facets_for_being_the_same(facet_j, facet_l)

                        if duplicate_facet_found:
                            # print(duplicate_facet_found, facet_j, facet_l)

                            if np.sum(np.abs(facet_j)) > np.sum(np.abs(facet_l)):
                                indices_to_drop.append(ind_i)
                                # print(ind_i)
                            else:
                                indices_to_drop.append(ind_k)
                                # print(ind_k)

    # #####################################################
    indices_to_drop = list(set(indices_to_drop))
    # #####################################################

    df_xrd_unique_1 = df_xrd_unique.drop(index=indices_to_drop)

    #__|

    # print("2222")








    # Grabbing top five most intense peaks
    top_rows = df_xrd_j.iloc[0:5]

    facets_flattened = []
    for facets_i in top_rows.facets.tolist():
        for facet_j in facets_i:
            facets_flattened.append(facet_j)

    # return(facets_flattened)
    out_dict = dict(
        df_xrd=df_xrd_i,
        # df_xrd_unique=df_xrd_unique,
        df_xrd_unique=df_xrd_unique_1,
        )
    return(out_dict)
    # return(df_xrd_unique)
    #__|



# facet_0 = (1, 0, 1)
# facet_1 = (3, 0, 1)

from methods import compare_facets_for_being_the_same











































# print("I copied a bunch of these  methods for testing make sure to remove")
# print("I copied a bunch of these  methods for testing make sure to remove")
# print("I copied a bunch of these  methods for testing make sure to remove")
# print("I copied a bunch of these  methods for testing make sure to remove")
# print("I copied a bunch of these  methods for testing make sure to remove")
# print("I copied a bunch of these  methods for testing make sure to remove")
# print("I copied a bunch of these  methods for testing make sure to remove")
# print("I copied a bunch of these  methods for testing make sure to remove")
# print("I copied a bunch of these  methods for testing make sure to remove")
# print("I copied a bunch of these  methods for testing make sure to remove")


#| - __old__

# def compare_facets_for_being_the_same(
#     facet_0,
#     facet_1,
#     ):
#     """
#     Checks whether facet_0 and facet_1 differ only by an integer multiplicative.
#     """
#     #| - compare_facets_for_being_the_same
#     # #########################################################
#     facet_j = facet_0
#     facet_k = facet_1
#
#     # #########################################################
#     facet_j_abs = [np.abs(i) for i in facet_j]
#     facet_j_sum = np.sum(facet_j_abs)
#
#     # #########################################################
#     facet_k_abs = [np.abs(i) for i in facet_k]
#     facet_k_sum = np.sum(facet_k_abs)
#
#     # #########################################################
#     if facet_j_sum > facet_k_sum:
#         # facet_j_abs / facet_k_abs
#
#         facet_larger = facet_j_abs
#         facet_small = facet_k_abs
#     else:
#         facet_larger = facet_k_abs
#         facet_small = facet_j_abs
#
#     # #########################################################
#     facet_frac = np.array(facet_larger) / np.array(facet_small)
#
#     # #####################################################
#     something_wrong = False
#     all_terms_are_whole_nums = True
#     # #####################################################
#     div_ints =  []
#     # #####################################################
#     for i_cnt, i in enumerate(facet_frac):
#         # print(i.is_integer())
#         if np.isnan(i):
#             if facet_j_abs[i_cnt] != 0 or facet_k_abs[i_cnt] != 0:
#                 something_wrong = True
#                 print("Not good, these should both be zero")
#
#         elif not i.is_integer() or i == 0:
#             all_terms_are_whole_nums = False
#             # print("Not a whole number here")
#
#         elif i.is_integer():
#             div_ints.append(int(i))
#
#     all_int_factors_are_same = False
#     if len(list(set(div_ints))) == 1:
#         all_int_factors_are_same = True
#
#     duplicate_found = False
#     if all_terms_are_whole_nums and not something_wrong and all_int_factors_are_same:
#         duplicate_found = True
#         # print("Found a duplicate facet here")
#
#     # #####################################################
#     # out_data = dict()
#     # #####################################################
#     # out_data["duplicate_found"] = duplicate_found
#     # out_data["facet_larger"] =
#     # out_data[""] =
#     # #####################################################
#
#     return(duplicate_found)
#     #__|


# print("duplicate_found:", duplicate_found)


































































# """
# """
#
# #| - Import Modules
# import os
# import sys
#
# import copy
# from pathlib import Path
#
# import json
#
# import numpy as np
# import pandas as pd
#
# # #############################################################################
# from ase import io
#
# from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
# from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
#     SimplestChemenvStrategy,
#     MultiWeightsChemenvStrategy)
# from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
# from pymatgen.io.ase import AseAtomsAdaptor
#
# from catkit.gen.surface import SlabGenerator
#
# # #############################################################################
# from methods import (
#     get_df_dft, symmetrize_atoms,
#     get_structure_coord_df, remove_atoms)
# from proj_data import metal_atom_symbol
# #__|
#
# from methods import get_slab_thickness
#
# def analyse_local_coord_env(
#     atoms=None,
#     ):
#     """
#     """
#     #| - analyse_local_coord_env
#     out_dict = dict()
#
#     Ir_indices = []
#     for i, s in enumerate(atoms.get_chemical_symbols()):
#         if s == "Ir":
#             Ir_indices.append(i)
#
#     struct = AseAtomsAdaptor.get_structure(atoms)
#
#
#     lgf = LocalGeometryFinder()
#     lgf.setup_structure(structure=struct)
#
#     se = lgf.compute_structure_environments(
#         maximum_distance_factor=1.41,
#         only_cations=False)
#
#
#     strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
#
#     lse = LightStructureEnvironments.from_structure_environments(
#         strategy=strategy, structure_environments=se)
#
#     isite = 0
#     cor_env = []
#     coor_env_dict = dict()
#     for isite in Ir_indices:
#         c_env = lse.coordination_environments[isite]
#         coor_env_dict[isite] = c_env
#         cor_env += [c_env[0]['ce_symbol']]
#
#     out_dict["coor_env_dict"] = coor_env_dict
#
#     return(out_dict)
#     #__|
#
# def check_if_sys_processed(
#     bulk_id_i=None,
#     facet_str=None,
#     df_slab_old=None,
#     ):
#     """
#     """
#     #| - check_if_sys_processed
#     sys_processed = False
#
#     num_rows_tot = df_slab_old.shape[0]
#     if num_rows_tot > 0:
#         df = df_slab_old
#         df = df[
#             (df["bulk_id"] == bulk_id_i) &
#             (df["facet"] == facet_str) &
#             [True for i in range(len(df))]
#             ]
#
#         num_rows = df.shape[0]
#         if num_rows > 0:
#             # print("There is a row that already exists")
#
#             sys_processed = True
#             if num_rows > 1:
#                 print("There is more than 1 row for this bulk+facet combination, what to do?")
#
#             row_i = df.iloc[0]
#
#     return(sys_processed)
#     #__|
#
# def remove_nonsaturated_surface_metal_atoms(
#     atoms=None,
#     dz=None,
#     ):
#     """
#     """
#     #| - remove_nonsaturated_surface_metal_atoms
#
#     # #################################################
#     # #################################################
#     z_positions = atoms.positions[:,2]
#
#     z_max = np.max(z_positions)
#     z_min = np.min(z_positions)
#
#     # #################################################
#     # #################################################
#     df_coord_slab_i = get_structure_coord_df(atoms)
#
#     # #################################################
#     metal_atoms_to_remove = []
#     for atom in atoms:
#         if atom.symbol == metal_atom_symbol:
#             z_pos_i = atom.position[2]
#             if z_pos_i >= z_max - dz or z_pos_i <= z_min + dz:
#                 row_coord = df_coord_slab_i[
#                     df_coord_slab_i.structure_index == atom.index].iloc[0]
#                 num_o_neighbors = row_coord.neighbor_count.get("O", 0)
#
#                 if num_o_neighbors < 6:
#                     metal_atoms_to_remove.append(atom.index)
#
#     slab_new = remove_atoms(atoms=atoms, atoms_to_remove=metal_atoms_to_remove)
#
#     return(slab_new)
#     # __|
#
# def remove_noncoord_oxygens(
#     atoms=None,
#     ):
#     """ """
#     #| - remove_noncoord_oxygens
#     df_coord_slab_i = get_structure_coord_df(atoms)
#
#     # #########################################################
#     df_i = df_coord_slab_i[df_coord_slab_i.element == "O"]
#     df_i = df_i[df_i.num_neighbors == 0]
#
#     o_atoms_to_remove = df_i.structure_index.tolist()
#
#     # #########################################################
#     o_atoms_to_remove_1 = []
#     df_j = df_coord_slab_i[df_coord_slab_i.element == "O"]
#     for j_cnt, row_j in  df_j.iterrows():
#         neighbor_count = row_j.neighbor_count
#
#         if neighbor_count.get("Ir", 0) == 0:
#             if neighbor_count.get("O", 0) == 1:
#                 o_atoms_to_remove_1.append(row_j.structure_index)
#
#
#     o_atoms_to_remove = list(set(o_atoms_to_remove + o_atoms_to_remove_1))
#
#     slab_new = remove_atoms(atoms, atoms_to_remove=o_atoms_to_remove)
#
#     return(slab_new)
#     # __|
#
# def create_slab_from_bulk(atoms=None, facet=None, layers=20):
#     """Create slab from bulk atoms object and facet.
#
#     Args:
#         atoms (ASE atoms object): ASE atoms object of bulk structure.
#         facet (str): Facet cut
#         layers (int): Number of layers
#     """
#     #| - create_slab_from_bulk
#
#     standardize_bulk = False
#
#     print(5 * "TEMP | cfgyhji | ")
#     # print("Turning off standardization of bulk")
#     print("standardize_bulk:", standardize_bulk)
#     print("facet:", facet)
#     print("Hello checking in here")
#     print(5 * "TEMP | cfgyhji | ")
#
#     SG = SlabGenerator(
#         atoms, facet, layers, vacuum=15,
#         # atoms, facet, 20, vacuum=15,
#         fixed=None, layer_type='ang',
#         attach_graph=True,
#
#         standardize_bulk=standardize_bulk,
#         # standardize_bulk=True,
#         # standardize_bulk=False,
#
#         primitive=False,
#
#         tol=1e-03)
#
#     # print("TEMP", "SlabGenerator:", SlabGenerator)
#
#     slab_i = SG.get_slab()
#     slab_i.set_pbc([True, True, True])
#
#
#     # #############################################
#     # Write slab to file ##########################
#     directory = "out_data"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     path_i = os.path.join("out_data", "temp.cif")
#     slab_i.write(path_i)
#
#     # Rereading the structure file to get it back into ase format
#     slab_i = io.read(path_i)
#     # data_dict_i["slab_0"] = slab_i
#
#     return(slab_i)
#     #__|
#
# def remove_highest_metal_atoms(
#     atoms=None,
#     num_atoms_to_remove=None,
#     metal_atomic_number=77,
#     ):
#     """ """
#     #| - remove_highest_metal_atom
#     slab_m = atoms[atoms.numbers == metal_atomic_number]
#
#     positions_cpy = copy.deepcopy(slab_m.positions)
#     positions_cpy_sorted = positions_cpy[positions_cpy[:,2].argsort()]
#
#     indices_to_remove = []
#     for coord_i in positions_cpy_sorted[-2:]:
#         for i_cnt, atom in enumerate(atoms):
#             if all(atom.position == coord_i):
#                 indices_to_remove.append(i_cnt)
#
#     slab_new = remove_atoms(
#         atoms=atoms,
#         atoms_to_remove=indices_to_remove,
#         )
#
#     return(slab_new)
#     #__|
#
# def calc_surface_area(atoms=None):
#     """ """
#     #| - calc_surface_area
#     cell = atoms.cell
#
#     cross_prod = np.cross(cell[0], cell[1])
#     area = np.linalg.norm(cross_prod)
#
#     return(area)
#     #__|
#
# def remove_all_atoms_above_cutoff(
#     atoms=None,
#     cutoff_thickness=17,
#     ):
#     """
#     """
#     #| - remove_all_atoms_above_cutoff
#     positions = atoms.positions
#
#     z_positions = positions[:,2]
#
#     z_max = np.max(z_positions)
#     z_min = np.min(z_positions)
#
#     atoms_new = atoms[z_positions < z_min + cutoff_thickness]
#
#     return(atoms_new)
#     #__|
#
# def create_final_slab_master(
#     atoms=None,
#     ):
#     """Master method to create final IrOx slab.
#     """
#     #| - create_final_slab_master
#     TEMP = True
#     slab_0 = atoms
#
#     ###########################################################
#     slab_thickness_i = get_slab_thickness(atoms=slab_0)
#     # print("slab_thickness_i:", slab_thickness_i)
#     slab_thickness_out = slab_thickness_i
#
#
#     cutoff_thickness_i = 14
#     break_loop = False
#     while not break_loop:
#         #| - Getting slab pre-ready
#         slab = remove_all_atoms_above_cutoff(
#             atoms=slab_0,
#             cutoff_thickness=cutoff_thickness_i)
#
#         ###########################################################
#         slab = remove_nonsaturated_surface_metal_atoms(atoms=slab, dz=4)
#         slab = remove_noncoord_oxygens(atoms=slab)
#
#         slab_thickness_i = get_slab_thickness(atoms=slab)
#         # print("slab_thickness_i:", slab_thickness_i)
#
#         if slab_thickness_i < 15:
#             cutoff_thickness_i = cutoff_thickness_i + 1.
#         else:
#             break_loop = True
#         #__|
#
#
#     #| - Main loop, chipping off surface atoms
#     # print("SIDJFISDIFJISDJFIJSDIJF")
#     # ###########################################################
#     # i_cnt = 2
#     # while slab_thickness_i > 15:
#     #     print("slab_thickness_i:", slab_thickness_i)
#     #
#     #     i_cnt += 1
#     #     # print(i_cnt)
#     #
#     #     # #####################################################
#     #     # Figuring out how many surface atoms to remove at one time
#     #     # Taken from R-IrO2 (100), which has 8 surface Ir atoms and a surface area of 58 A^2
#     #     surf_area_per_surface_metal = 58 / 8
#     #     surface_area_i = calc_surface_area(atoms=slab)
#     #     ideal_num_surface_atoms = surface_area_i / surf_area_per_surface_metal
#     #     num_atoms_to_remove = ideal_num_surface_atoms / 3
#     #     num_atoms_to_remove = int(np.round(num_atoms_to_remove))
#     #     # #####################################################
#     #
#     #     slab_new_0 = remove_highest_metal_atoms(
#     #         atoms=slab,
#     #         num_atoms_to_remove=num_atoms_to_remove,
#     #         metal_atomic_number=77)
#     #     if TEMP:
#     #         slab_new_0.write("out_data/temp_out/slab_1_" + str(i_cnt) + "_0" + ".cif")
#     #
#     #     slab_new_1 = remove_nonsaturated_surface_metal_atoms(
#     #         atoms=slab_new_0,
#     #         dz=4)
#     #     if TEMP:
#     #         slab_new_1.write("out_data/temp_out/slab_1_" + str(i_cnt) + "_1" + ".cif")
#     #
#     #     slab_new_2 = remove_noncoord_oxygens(atoms=slab_new_1)
#     #     if TEMP:
#     #         slab_new_2.write("out_data/temp_out/slab_1_" + str(i_cnt) + "_2" + ".cif")
#     #
#     #
#     #     slab_thickness_i = get_slab_thickness(atoms=slab_new_2)
#     #     # print("slab_thickness_i:", slab_thickness_i)
#     #
#     #     if TEMP:
#     #         slab_new_2.write("out_data/temp_out/slab_1_" + str(i_cnt) + ".cif")
#     #
#     #     slab = slab_new_2
#     #__|
#
#     return(slab)
#     #__|
#
# def create_save_dataframe(data_dict_list=None, df_slab_old=None):
#     """
#     """
#     #| - create_save_dataframe
#     # #####################################################
#     # Create dataframe
#     df_slab = pd.DataFrame(data_dict_list)
#     num_new_rows = len(data_dict_list)
#
#     if num_new_rows > 0:
#         df_slab = df_slab.set_index("slab_id")
#     elif num_new_rows == 0:
#         print("There aren't any new rows to process")
#         assert False
#
#     # #####################################################
#     df_slab = pd.concat([
#         df_slab_old,
#         df_slab,
#         ])
#
#     # Pickling data #######################################
#     import os; import pickle
#     directory = "out_data"
#     if not os.path.exists(directory): os.makedirs(directory)
#     with open(os.path.join(directory, "df_slab.pickle"), "wb") as fle:
#         pickle.dump(df_slab, fle)
#     # #####################################################
#
#     df_slab_old = df_slab
#
#     return(df_slab_old)
#     #__|
#
# def constrain_slab(
#     atoms=None,
#     ):
#     """Constrain lower portion of slab geometry.
#
#     Has a little bit of built in logic which should provide better slabs.
#     """
#     #| - constrain_slab
#     slab = atoms
#
#     positions = slab.positions
#
#     z_pos = positions[:,2]
#
#     z_max = np.max(z_pos)
#     z_min = np.min(z_pos)
#
#     # ang_of_slab_to_constrain = (2 / 4) * (z_max - z_min)
#     ang_of_slab_to_constrain = (z_max - z_min) - 6
#
#
#     # #########################################################
#     indices_to_constrain = []
#     for atom in slab:
#         if atom.symbol == metal_atom_symbol:
#             if atom.position[2] < (z_min + ang_of_slab_to_constrain):
#                 indices_to_constrain.append(atom.index)
#
#     for atom in slab:
#         if atom.position[2] < (z_min + ang_of_slab_to_constrain - 2):
#             indices_to_constrain.append(atom.index)
#
#     df_coord_slab_i = get_structure_coord_df(slab)
#
#     # #########################################################
#     other_atoms_to_constrain = []
#     for ind_i in indices_to_constrain:
#         row_i = df_coord_slab_i[df_coord_slab_i.structure_index == ind_i]
#         row_i = row_i.iloc[0]
#
#         nn_info = row_i.nn_info
#
#         for nn_i in nn_info:
#             ind_j = nn_i["site_index"]
#             other_atoms_to_constrain.append(ind_j)
#
#     # print(len(indices_to_constrain))
#
#     indices_to_constrain.extend(other_atoms_to_constrain)
#
#     # print(len(indices_to_constrain))
#
#     # #########################################################
#     constrain_bool_mask = []
#     for atom in slab:
#         if atom.index in indices_to_constrain:
#             constrain_bool_mask.append(True)
#         else:
#             constrain_bool_mask.append(False)
#
#     # #########################################################
#     slab_cpy = copy.deepcopy(slab)
#
#     from ase.constraints import FixAtoms
#     c = FixAtoms(mask=constrain_bool_mask)
#     slab_cpy.set_constraint(c)
#
#     # slab_cpy.constraints
#
#     return(slab_cpy)
#     #__|
#
# def read_data_json():
#     """
#     """
#     #| - read_data_json
#     path_i = os.path.join(
#         "out_data", "data.json")
#     my_file = Path(path_i)
#     if my_file.is_file():
#         data_path = os.path.join(
#             "out_data/data.json")
#         with open(data_path, "r") as fle:
#             data = json.load(fle)
#     else:
#         data = dict()
#
#     return(data)
#     #__|
#
# def resize_z_slab(atoms=None, vacuum=15):
#     """
#     """
#     #| - resize_z_slab
#     z_pos = atoms.positions[:,2]
#     z_max = np.max(z_pos)
#     z_min = np.min(z_pos)
#
#     new_z_height = z_max - z_min + vacuum
#
#     cell = atoms.cell
#
#     cell_cpy = cell.copy()
#     cell_cpy = cell_cpy.tolist()
#
#     cell_cpy[2][2] = new_z_height
#
#     atoms_cpy = copy.deepcopy(atoms)
#     atoms_cpy.set_cell(cell_cpy)
#
#     return(atoms_cpy)
#     #__|
#
# def repeat_xy(atoms=None, min_len_x=6, min_len_y=6):
#     """
#     """
#     #| - repeat_xy
#     print("min_len_x:", min_len_x, "min_len_y:", min_len_y)
#
#     cell = atoms.cell.array
#
#     x_mag = np.linalg.norm(cell[0])
#     y_mag = np.linalg.norm(cell[1])
#
#     import math
#     mult_x = 1
#     if x_mag < min_len_x:
#         mult_x = math.ceil(min_len_x / x_mag)
#         #  mult_x = round(min_len_x / x_mag)
#         mult_x = int(mult_x)
#         # print(mult_x)
#     mult_y = 1
#     if y_mag < min_len_y:
#         mult_y = math.ceil(min_len_y / y_mag)
#         #  mult_y = round(min_len_y / y_mag)
#         mult_y = int(mult_y)
#
#     repeat_list = (mult_x, mult_y, 1)
#     atoms_repeated = atoms.repeat(repeat_list)
#
#     # Check if the atoms were repeated or not
#     if mult_x > 1 or mult_y > 1:
#         is_repeated = True
#     else:
#         is_repeated = False
#
#     # Construct final out dict
#     out_dict = dict(
#         atoms_repeated=atoms_repeated,
#         is_repeated=is_repeated,
#         repeat_list=repeat_list,
#         )
#     return(out_dict)
#     #__|

#__|
