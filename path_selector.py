from scipy.spatial import Voronoi
from structure import *


def get_multiplying_factors(paths, a, b, c, min_dist=3):

    vs = np.abs([p1 - p2 for p1, p2 in paths])
    ka = int(np.ceil(max(vs[:, 0]) + min_dist / a))
    kb = int(np.ceil(max(vs[:, 1]) + min_dist / b))
    kc = int(np.ceil(max(vs[:, 2]) + min_dist / c))
    return ka, kb, kc


def find_rneighbors(points, central_points, min_dist=0.0, max_dist=4):

    if min_dist >= max_dist:
        raise ValueError("min_dist should be less than max_dist!")
    neighbors = {i: None for i in central_points}
    for i in central_points:
        local_neighbors = []
        for j, p in enumerate(points):
            if i != j and min_dist <= np.linalg.norm(p - points[i]) <= max_dist:
                local_neighbors.append(j)
        neighbors[i] = local_neighbors
    return neighbors


def find_vneighbors(points, central_points, key=1):

    """
     Parameter key can takes values 1, 2, or 3 that correspond to the
    search for atomic domains common by vertices, edges or faces.
    """

    neighbors = {i: None for i in central_points}
    vor = Voronoi(points)
    for i in central_points:
        region = vor.regions[vor.point_region[i]]
        if -1 in region:
            raise ValueError("The domain for \"" + str(i) + "\" point is not closed!")
        local_neighbors = []
        for j in range(len(points)):
            numb_common_vertices = len(np.intersect1d(region, vor.regions[vor.point_region[j]]))
            if i != j and numb_common_vertices >= key:
                local_neighbors.append(j)
        neighbors[i] = local_neighbors
    return neighbors


def find_neighbor_positions(structure, sublattice=[], mod=1,
                            min_dist=-float('inf'), max_dist=float('inf')):


    neighbors = []
    non_equiv_neighbors = []
    structure.multiply_cell()
    structure.nonequiv_bonds = {}
    atoms = [a for a in structure.atoms.values() if a.element.number in sublattice]
    points = np.array([a.cart_coord for a in atoms])
    non_equiv = [i for i, a in enumerate(atoms) if a.symmetry == 1 and (a.translation == [0, 0, 0]).all()]
    if mod == 0:
        neighbor_positions = find_rneighbors(points, non_equiv, min_dist=min_dist, max_dist=max_dist)
    else:
        neighbor_positions = find_vneighbors(points, central_points=non_equiv, key=mod)
    for i, indexes in neighbor_positions.items():
        for j in indexes:
            if min_dist <= np.linalg.norm(points[i] - points[j]) <= max_dist:
                neighbors.append((atoms[i], atoms[j]))
    structure_copy = structure.copy()
    structure_copy.reamove_connectivity()
    for a1, a2 in neighbors:
        structure_copy.add_bond(a1, a2, type='S')
    for bond in structure_copy.nonequiv_bonds.values():
        non_equiv_neighbors.append((structure.get_atom(bond.atom_1.name), structure.get_atom(bond.atom_2.name)))
    non_equiv_neighbors.sort(key=lambda x: np.linalg.norm(x[0].cart_coord - x[1].cart_coord))
    return non_equiv_neighbors
