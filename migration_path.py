import numpy as np


def find_pref_paths(structure, ccp_1, ccp2, length_limit=5):

    def go_deep(path, traversed):

        if len(traversed) < length_limit:
            for atom in structure.get_neighbors(path[-1], bonding_types='S'):
                if atom.name == ccp2.name:
                    new_path = [p for p in path]
                    new_path.append(ccp2)
                    paths.append(new_path)
                elif not traversed.get(atom.name[0]):
                    new_path = [p for p in path]
                    new_traversed = {k: v for k, v in traversed.items()}
                    new_path.append(atom)
                    new_traversed[atom.name[0]] = atom
                    go_deep(new_path, new_traversed)

    paths = []
    structure.calc_adjacency_list()
    go_deep([ccp_1], {ccp_1.name[0]: ccp_1})
    return paths


def find_migration_path(structure, start, end, tol=1e-4, length_limit=5):

    structure = structure.copy()
    structure.multiply_cell()
    # Searching ends of the path
    ccp1, dist1 = structure.get_closest_atom(structure.cell.get_cart_coord(start),
                                             condition=lambda a: a.element.number == 2)
    ccp2, dist2 = structure.get_closest_atom(structure.cell.get_cart_coord(end),
                                             condition=lambda a: a.element.number == 2)
    print(ccp1.name, ccp2.name, ccp1.properties, ccp2.properties, dist1, dist2)
    if ccp1.name == ccp2.name:
        return [ccp1]
    # Reamoving ends of path from cps list
    atoms = np.array(list(structure.nonequiv_atoms.values()))
    cps = atoms[np.where(list(map((lambda x: x.element.number == -1 or x.element.number == 2), atoms)))]
    cps = np.delete(cps, np.where(list(map((lambda x: x.name[0] == ccp1.name[0] or x.name[0] == ccp2.name[0]), cps))))
    # Sorting cps by electron density
    cps = np.array(sorted(cps, key=lambda x: -x.properties))
    structure.calc_adjacency_list()
    # Removing of critical points
    copy = structure.copy()
    while structure.is_connected(ccp1, ccp2, bonding_types='S'):
        copy = structure.copy()
        density = cps[0].properties
        while len(cps) != 0 and abs(density - cps[0].properties) < tol:
            structure.remove_atoms([cps[0]])
            cps = np.delete(cps, 0)
        structure.calc_adjacency_list()
    structure = copy
    # Searching of migration paths
    pref_paths = []
    for p in find_pref_paths(structure, ccp1, ccp2, length_limit):
        eldens_ = [cp.properties for cp in p]
        pref_paths += [[max(eldens_) - min(eldens_), sum(eldens_), p]]
    pref_paths.sort(key=lambda x: (x[0], x[1]))
    if len(pref_paths) == 0:
        return []
    migration_path = pref_paths[0][2]
    return migration_path