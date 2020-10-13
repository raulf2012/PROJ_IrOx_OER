"""
"""

#| - Import Moduels
from sys import argv
import numpy as np
import pylab as p
import ase
from ase.io import read
from ase.visualize import view
from catkit.gen.utils.connectivity import get_cutoff_neighbors, get_voronoi_neighbors
from mendeleev import element
#__|

def get_connectivity(atoms):
    """
    """
    #| - get_connectivity
    matrix = get_cutoff_neighbors(atoms, scale_cov_radii=1.1)
    #matrix = get_voronoi_neighbors(atoms, cutoff=3)
    return matrix
    #__|

def set_formal_oxidation_state(atoms, charge_O=-2, charge_H=1):
    """
    """
    #| - set_formal_oxidation_state
    O_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'O'])
    H_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'H'])
    M_indices = np.array([i for i, a in enumerate(atoms)
                          if not a.symbol in ['O', 'H']])
    non_O_indices = np.array([i for i, a in enumerate(atoms)
                              if not a.symbol in ['O']])
    # Connectivity matrix from CatKit Voronoi
    con_matrix = get_connectivity(atoms)
    oxi_states = np.ones([len(atoms)])
    oxi_states[O_indices] = charge_O
    if H_indices:  # First correct O charge due to H
        oxi_states[H_indices] = charge_H
        for H_i in H_indices:
            H_O_connectivity = con_matrix[H_i][O_indices]
            norm = np.sum(H_O_connectivity)
            O_indices_H = O_indices[np.where(H_O_connectivity)[0]]
            oxi_states[O_indices_H] += charge_H / norm
    for metal_i in M_indices:  # Substract O connectivity
        M_O_connectivity = con_matrix[metal_i][O_indices]
        norm = np.sum(con_matrix[O_indices][:, M_indices], axis=-1)
        oxi_states[metal_i] = sum(
            M_O_connectivity * -oxi_states[O_indices] / norm)
    atoms.set_initial_charges(np.round(oxi_states, 4))
    return atoms
    #__|

def set_renormalized_oxidation_state(atoms, moments=None, charge_O=-2, charge_H=1):
    O_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'O'])
    H_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'H'])
    M_indices = np.array([i for i, a in enumerate(atoms)
                          if not a.symbol in ['O', 'H']])
    non_O_indices = np.array([i for i, a in enumerate(atoms)
                              if not a.symbol in ['O']])
    # Connectivity matrix from CatKit Voronoi
    con_matrix = get_connectivity(atoms)
    oxi_states = np.ones([len(atoms)])
    oxi_states[O_indices] = charge_O + \
        np.abs(atoms.get_initial_magnetic_moments()[O_indices])
    if H_indices:  # First correct O charge due to H
        oxi_states[H_indices] = charge_H - \
            np.abs(atoms.get_initial_magnetic_moments()[H_indices])
        for H_i in H_indices:
            H_O_connectivity = con_matrix[H_i][O_indices]
            norm = np.sum(H_O_connectivity)
            O_indices_H = O_indices[np.where(H_O_connectivity)[0]]
            oxi_states[O_indices_H] += oxi_states[H_i] / norm
    for metal_i in M_indices:  # Substract O connectivity
        M_O_connectivity = con_matrix[metal_i][O_indices]
        norm = np.sum(con_matrix[O_indices][:, M_indices], axis=-1)
        oxi_states[metal_i] = sum(
            M_O_connectivity * -oxi_states[O_indices] / norm)
    atoms.set_initial_charges(np.round(oxi_states, 4))
    return atoms
def get_n_outer_electrons(atoms):
    nds = []
    for i, s in enumerate(atoms.symbols):
        group = element(s).group_id
        if not s in ['O', 'H']:
            nd = group - atoms.get_initial_charges()[i]
            nd = min(max(nd, 0), 10)
        else:
            nd = 0
        nds += [nd]
    atoms.set_tags(nds)
    return(atoms)
if __name__ == '__main__':
    atoms = read(argv[1])
    #atoms = set_oxidation_state(atoms, charge_O=-2, charge_H=1)
    # ASE gui only displays initial moments for some reson.
    atoms.set_initial_magnetic_moments(atoms.get_magnetic_moments())
    # print(atoms.results)
    atoms = set_renormalized_oxidation_state(atoms)
    # get_n_outer_electrons(atoms)
    #atoms.constraints = None
    view(atoms)
