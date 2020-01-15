from symmetry import*
from geometry import*
from constants import*
import numpy as np


class Atom:

    def __init__(
            self,
            atomic_number,
            index,
            fract_coord,
            cart_coord,
            symmetry=1,
            site=(0, 0, 0),
            occupancy=1.0,
            multiplicity=None,
            properties=None
    ):
        self.name = None
        self.element = PERIODIC_TABLE[atomic_number]
        self.index = index
        self.fract_coord = np.array(fract_coord)
        self.cart_coord = np.array(cart_coord)
        self.symmetry = symmetry
        self.site = np.array(site, dtype=np.int8)
        self.translation = np.array([0, 0, 0], dtype=np.int8)
        self.occupancy = occupancy
        self.multiplicity = multiplicity
        self.properties = properties
        self.equal = self
        self.assign_name()

    def copy(self):

        atom_copy = Atom(
            self.element.number,
            self.index,
            self.fract_coord,
            self.cart_coord,
            self.symmetry,
            self.site,
            self.occupancy,
            self.multiplicity,
            self.properties
        )
        atom_copy.translation = np.array(self.translation)
        atom_copy.equal = self.equal
        return atom_copy

    @staticmethod
    def give_name(atomic_number, index, symmetry, site):

        name = ((atomic_number, index, symmetry), tuple(site))
        return name

    def assign_name(self):

        self.name = self.give_name(self.element.number, self.index, self.symmetry, self.site)
        return self.name

    def __str__(self):

        return str(self.name)

class Bond:

    def __init__(self, atom_1, atom_2, type='V', length=None, multiplicity=None, order=None, weight=None):

        self.name = None
        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.type = type
        self.length = length
        self.multiplicity = multiplicity
        self.order = order
        self.weight = weight
        self.equal = self
        self.assign_name()

    def copy(self):

        bond_copy = Bond(
            self.atom_1,
            self.atom_2,
            self.type,
            self.length,
            self.multiplicity,
            self.order,
            self.weight
        )
        bond_copy.equal = self.equal
        return bond_copy

    def cal_length(self):

        return np.linalg.norm(np.array(self.atom_1.cart_coord) - self.atom_2.cart_coord)

    def assign_name(self):

        self.name = (self.atom_1.name, self.atom_2.name, self.type)
        return self.name

    def __str__(self):

        text = str(self.name)
        return text


class Cell:

    def __init__(self, a, b, c, alpha, beta, gamma):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.vectors = None
        self.inv_vectors = None
        self.volume = None
        self.calc_cell_vectors()

    def calc_cell_vectors(self, prec=0.001):

        if abs(self.alpha - 90) < prec and abs(self.beta - 90) < prec and abs(self.gamma - 90) < prec:
            self.vectors = np.array(
                [[self.a, 0, 0],
                 [0, self.b, 0],
                 [0, 0, self.c]]
            )
        else:
            self.vectors = calc_vectors(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)
            self.inv_vectors = np.linalg.inv(self.vectors)
        return self.vectors

    def get_cart_coord(self, fract_coord, prec=0.001):

        if abs(self.alpha - 90) < prec and abs(self.beta - 90) < prec and abs(self.gamma - 90) < prec:
            cart_coord = fract_coord * [self.a, self.b, self.c]
        else:
            cart_coord = (
                    self.vectors[0] * fract_coord[0]
                    + self.vectors[1] * fract_coord[1]
                    + self.vectors[2] * fract_coord[2]
            )
        return cart_coord

    def get_fract_coord(self, cart_coord, prec=0.001):

        if abs(self.alpha - 90) < prec and abs(self.beta - 90) < prec and abs(self.gamma - 90) < prec:
            fract_coord = cart_coord / [self.a, self.b, self.c]
        else:
            fract_coord = (
                    self.inv_vectors[0] * cart_coord[0]
                    + self.inv_vectors[1] * cart_coord[1]
                    + self.inv_vectors[2] * cart_coord[2]
            )
        return fract_coord

    def cal_volume(self):

        self.volume = np.dot(self.vectors[0], np.cross(self.vectors[1], self.vectors[2]))
        return self.volume

    def copy(self):

        copy = Cell(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)
        return copy

    def extend(self, ka, kb, kc):

        self.a = ka * self.a
        self.b = kb * self.b
        self.c = kc * self.c
        self.calc_cell_vectors()
        return None


class StructuralGroup:

    def __init__(self, atoms, bonds, translations, index=1, symmetry=1, site=np.array([0, 0, 0]), multiplicity=1):

        self.name = None
        self.index = index
        self.symmetry = symmetry
        self.site = site
        self.translations = np.array(translations)
        self.atoms = atoms
        self.bonds = bonds
        self.periodicity = None
        self.multiplicity = multiplicity
        self.composition = None
        self.formula = None
        self.equal = self

    def assign_name(self):

        self.name = ((self.formula, self.index, self.symmetry), tuple(self.site))
        return self.name

    def copy(self):

        group = StructuralGroup(self.atoms,
                                self.bonds,
                                self.translations,
                                self.index,
                                self.symmetry,
                                self.site,
                                self.multiplicity)
        group.equal = self.equal
        return group

    def calc_nonequiv_translations(self):

        if len(self.translations) != 0:
            nonequiv_translations = [self.translations[0]]
            for t in self.translations:
                new = True
                for nt in nonequiv_translations:
                    if (
                            (t == nt).all() or
                            (t == -nt).all() or
                            (t[0] == nt[0] == t[1] == nt[1] == 0) or
                            (t[0] == nt[0] == t[2] == nt[2] == 0) or
                            (t[1] == nt[1] == t[2] == nt[2] == 0)
                    ):
                        new = False
                        break
                if new:
                    nonequiv_translations.append(t)
            self.translations = nonequiv_translations
        return self.translations

    def calc_periodicity(self):

        translations = self.calc_nonequiv_translations()
        if len(translations) == 0:
            self.periodicity = 0
        else:
            if len(translations) == 1:
                self.periodicity = 1
            elif len(translations) == 2:
                self.periodicity = 2
            elif are_coplanar(translations, prec=0):
                self.periodicity = 2
            else:
                self.periodicity = 3
        return self.periodicity

    def draw(self):

        import pylab as pl
        import mpl_toolkits.mplot3d as a3
        import matplotlib.pyplot as plt
        ax = a3.Axes3D(pl.figure())
        atoms = list(self.atoms.values())
        # Plot atoms
        for a in atoms:
            c = a.cart_coord
            ax.scatter(c[0], c[1], c[2], color=a.element.color, s=a.element.wdv_radius * 200, zorder=50)
            ax.text(c[0], c[1], c[2], a.element.symbol + str(a.index), fontsize=10, zorder=100)
        # Plot bonds
        if len(self.bonds) != 0:
            n1 = np.array([b.atom_1.cart_coord for b in list(self.bonds.values())])
            n2 = np.array([b.atom_2.cart_coord for b in list(self.bonds.values())])
            x = [[n1[i][0], n2[i][0]] for i in range(len(n1))]
            y = [[n1[i][1], n2[i][1]] for i in range(len(n1))]
            z = [[n1[i][2], n2[i][2]] for i in range(len(n1))]
            for i in range(len(x)):
                ax.plot(x[i], y[i], z[i], linewidth=3, color='c', zorder=1)
        #plt.title(self.name)
        plt.tight_layout()
        plt.axis('off')
        plt.show()
        return None


class Structure:

    def __init__(self):

        self.structure_data = None
        self.cell = None
        self.symmetry = None
        self.equiv_positions = {}
        self.elements_count = {}
        self.nonequiv_atoms = {}
        self.equiv_atoms = {}
        self.atoms = {}
        self.nonequiv_bonds = {}
        self.equiv_bonds = {}
        self.bonds = {}
        self.adjacency_list = {}
        self.structural_groups = {}

    def build_structure(self, structure_data):

        self.structure_data = structure_data
        # Unit cell
        self.cell = Cell(
            structure_data.cell_length_a,
            structure_data.cell_length_b,
            structure_data.cell_length_c,
            structure_data.cell_angle_alpha,
            structure_data.cell_angle_beta,
            structure_data.cell_angle_gamma
        )
        # Symmetry
        self.symmetry = Symmetry(structure_data.symmetry_rotations, structure_data.symmetry_translations)
        # Atoms
        atoms_index = []
        for i, symbol in enumerate(structure_data.atom_site_type_symbol):
            atomic_number = get_number(symbol)
            self.add_atom(atomic_number, structure_data.atom_site_fract[i])
            atoms_index += [self.elements_count[symbol]]
        # Bonds
        for i in range(len(structure_data.symmetry_1)):
            atom_1 = self.get_equal_atom(
                get_number(structure_data.atom_site_type_symbol[structure_data.indexes_1[i]]),
                atoms_index[structure_data.indexes_1[i]],
                structure_data.symmetry_1[i],
                structure_data.translation_1[i]
            )
            atom_2 = self.get_equal_atom(
                get_number(structure_data.atom_site_type_symbol[structure_data.indexes_2[i]]),
                atoms_index[structure_data.indexes_2[i]],
                structure_data.symmetry_2[i],
                structure_data.translation_2[i]
            )

            bond_type = structure_data.bond_type[i]
            self.add_bond(atom_1, atom_2, bond_type)
        self.calc_adjacency_list()
        return self

    def copy(self):

        copy = Structure()
        copy.cell = self.cell.copy()
        copy.symmetry = self.symmetry.copy()
        copy.structure_data = self.structure_data
        for atom in self.nonequiv_atoms.values():
            atom_copy = atom.copy()
            atom_copy.equal = atom_copy
            copy.nonequiv_atoms[atom_copy.name] = atom_copy
            copy.multiply_atom(atom_copy)
        for bond in self.nonequiv_bonds.values():
            atom_1 = copy.get_atom(bond.atom_1.name)
            atom_2 = copy.get_atom(bond.atom_2.name)
            copy.add_bond(atom_1, atom_2, type=bond.type, length=bond.length, weight=np.array(bond.weight))
        copy.calc_adjacency_list()
        return copy

    def increment_element_index(self, symbol):

        index = self.elements_count.get(symbol)
        if index is None:
            index = 1
            self.elements_count[symbol] = index
        else:
            index += 1
            self.elements_count[symbol] = index
        return index

    def multiply_atom(self, atom, prec=0.01):

        equiv_positions = {1: (1, np.array([0, 0, 0]))}
        equiv_atoms = [atom]
        multiplicity = 1
        for i, symop in enumerate(self.symmetry.symm_operations[1:]):
            new_fract_coord = self.symmetry.apply_symmetry(atom.fract_coord, symop)
            new_fract_coord, translation = self.symmetry.calc_coord_and_translation(new_fract_coord)
            new_cart_coord = self.cell.get_cart_coord(new_fract_coord)
            unique = True
            for j, equiv_atom in enumerate(equiv_atoms):
                if np.linalg.norm(new_cart_coord - equiv_atom.cart_coord) < prec:
                    if equiv_atom.symmetry != i + 2:
                        equiv_positions[i + 2] = (equiv_atoms[j].symmetry, equiv_atoms[j].translation - translation)
                    unique = False
                    break
            if unique:
                equiv_atom = atom.copy()
                equiv_atom.fract_coord = new_fract_coord
                equiv_atom.cart_coord = new_cart_coord
                equiv_atom.symmetry = i + 2
                equiv_atom.translation = translation
                equiv_atom.assign_name()
                equiv_atoms += [equiv_atom]
                equiv_positions[i + 2] = (equiv_atom.symmetry, np.array([0, 0, 0]))
                multiplicity += 1
        atom.multiplicity = multiplicity
        self.equiv_atoms.update([(a.name, a) for a in equiv_atoms])
        self.atoms.update(self.equiv_atoms)
        self.equiv_positions[(atom.element.number, atom.index)] = equiv_positions
        return None

    def add_atom(self, atomic_number, fract_coord, symmetry=1, properties=None):

        #fract_coord = fract_coord % 1
        fract_coord = Symmetry.calc_coord_and_translation(fract_coord)[0]
        index = self.increment_element_index(PERIODIC_TABLE[atomic_number].symbol)
        cart_coord = self.cell.get_cart_coord(fract_coord)
        atom = Atom(atomic_number, index, fract_coord, cart_coord, symmetry=symmetry, properties=properties)
        self.nonequiv_atoms[atom.name] = atom
        self.multiply_atom(atom)
        return atom

    def get_atom(self, atom_name):

        atom = self.atoms.get(atom_name)
        if atom is None:
            translation = atom_name[1]
            atom_name = (atom_name[0], (0, 0, 0))
            atom = self.atoms.get(atom_name)
            if atom is not None:
                atom = atom.copy()
                atom.cart_coord += (
                        translation[0] * self.cell.vectors[0]
                        + translation[1] * self.cell.vectors[1]
                        + translation[2] * self.cell.vectors[2]
                )
                atom.site += translation
                atom.translation += translation
                atom.assign_name()
                self.atoms[atom.name] = atom
        return atom

    def remove_atoms(self, atoms):

        for atom in atoms:
            if self.get_atom(atom.equal.name):
                bonds = {}
                for bond in self.nonequiv_bonds.values():
                    if bond.atom_1.equal == atom.equal or bond.atom_2.equal == atom.equal:
                        bonds[bond.name] = bond
                self.remove_bonds(list(bonds.values()))
                equiv_atoms = list(self.equiv_atoms.values())
                all_atoms = list(self.atoms.values())
                for a in all_atoms:
                    if a.equal == atom.equal:
                        self.atoms.pop(a.name)
                for a in equiv_atoms:
                    if a.equal == atom.equal:
                        self.equiv_atoms.pop(a.name)
                self.nonequiv_atoms.pop(atom.equal.name)

        return None

    def get_equal_atom(self, atomic_number, index, symmetry, translation):

        eq_symmetry, additional_translation = self.equiv_positions[(atomic_number, index)][symmetry]
        name = Atom.give_name(atomic_number, index, eq_symmetry, (0, 0, 0))
        site = translation + additional_translation - self.get_atom(name).translation
        atom = self.get_atom(Atom.give_name(atomic_number, index, eq_symmetry, site))
        return atom

    def get_closest_atom(self, point, condition=(lambda x: True)):

        dist = float('inf')
        closest_atom = None
        for atom in self.atoms.values():
            if condition(atom):
                d = np.linalg.norm(atom.cart_coord - point)
                if dist > d:
                    dist = d
                    closest_atom = atom
        return closest_atom, dist

    def is_connected(self, atom_1, atom_2, bonding_types='V S W H'):

        traversed = {}
        traversed2 = {}
        atoms = [atom_1]
        for atom in atoms:
            if not traversed.get(atom.name):
                traversed[atom.name] = atom
                traversed2[atom.name[0]] = atom
                for neighbor in self.get_neighbors(atom, bonding_types):
                    if neighbor.name == atom_2.name:
                        return True
                    if not (traversed2.get(neighbor.name[0])):
                        atoms.append(neighbor)
        return False

    @staticmethod
    def calc_compositions(atoms, cond=lambda x: True):

        s_dict = {}
        symbols = [a.element.symbol for a in atoms]
        for s in symbols:
            if cond(s):
                if s_dict.get(s) is None:
                    s_dict[s] = 1
                else:
                    s_dict[s] += 1
        composition = list(s_dict.items())
        composition.sort(key=lambda x: get_number(x[0]))
        return composition

    @staticmethod
    def sort_atoms(atoms):

        atoms.sort(key=lambda x: (x.element.number, x.index, x.symmetry, x.site[0], x.site[1], x.site[2]))
        return atoms

    def add_bond(self, atom_1, atom_2, type='V', length=None, multiplicity=None, order=None, weight=None):

        atom_1, atom_2 = self.sort_atoms([atom_1, atom_2])
        atom_2 = self.get_atom((atom_2.name[0], tuple(atom_2.site - atom_1.site)))
        atom_1 = self.get_atom((atom_1.name[0], (0, 0, 0)))
        if self.bonds.get((atom_1.name, atom_2.name, type)) is not None:
            return None
        bond = Bond(atom_1, atom_2, type, length, multiplicity, order, weight)
        self.nonequiv_bonds[bond.name] = bond
        self.multiply_bond(bond)
        return None

    def multiply_bond(self, bond):

        s = self.symmetry
        equiv_bonds = {}
        self.equiv_bonds[bond.name] = bond
        self.bonds[bond.name] = bond
        for i, symop in enumerate(s.symm_operations[1:]):
            symmetry_1 = s.symm_operations[bond.atom_1.symmetry - 1]
            symmetry_1 = (symmetry_1[0], symmetry_1[1] + bond.atom_1.translation)
            new_symmetry_1, translations_1 = s.get_equiv_symop(s.summarize_operations(symmetry_1, symop))
            atom_1 = self.get_equal_atom(bond.atom_1.element.number, bond.atom_1.index, new_symmetry_1, translations_1)
            symmetry_2 = s.symm_operations[bond.atom_2.symmetry - 1]
            symmetry_2 = (symmetry_2[0],  symmetry_2[1] + bond.atom_2.translation)
            new_symmetry_2, translations_2 = s.get_equiv_symop(s.summarize_operations(symmetry_2, symop))
            atom_2 = self.get_equal_atom(bond.atom_2.element.number, bond.atom_2.index, new_symmetry_2, translations_2)
            atom_1, atom_2 = self.sort_atoms([atom_1, atom_2])
            atom_2 = self.get_atom((atom_2.name[0], tuple(atom_2.site - atom_1.site)))
            atom_1 = self.get_atom((atom_1.name[0], (0, 0, 0)))
            equiv_bond = Bond(atom_1, atom_2, bond.type)
            equiv_bond.weight = bond.weight
            equiv_bond.equal = bond.equal
            equiv_bonds[equiv_bond.name] = equiv_bond
            self.equiv_bonds[equiv_bond.name] = equiv_bond
            self.bonds[equiv_bond.name] = equiv_bond
            if abs(bond.cal_length() - equiv_bond.cal_length()) > 0.01:
                print(self.structure_data.block_name)
                print(bond.cal_length(), equiv_bond.cal_length())
                raise RuntimeError("!!!!!!!!")
        return None

    def get_bond(self, name):

        atom_1 = self.get_atom(name[0])
        atom_2 = self.get_atom(name[1])
        type = name[2]
        atom_1, atom_2 = self.sort_atoms([atom_1, atom_2])
        bond = self.equiv_bonds[
            ((atom_1.name[0], (0, 0, 0)), (atom_2.name[0], tuple(atom_2.site - atom_1.site)), type)
        ]
        if atom_1.name == name[0]:
            translation = name[0][1]
        else:
            translation = name[1][1]
        if translation != (0, 0, 0):
            bond = bond.copy()
            bond.atom_1 = self.get_atom((bond.atom_1.name[0], tuple(bond.atom_1.site + translation)))
            bond.atom_2 = self.get_atom((bond.atom_2.name[0], tuple(bond.atom_2.site + translation)))
            bond.assign_name()
            self.bonds[bond.name] = bond
        return bond

    def remove_bonds(self, bonds):

        for bond in bonds:
            if self.get_bond(bond.equal.name):
                all_bonds = list(self.bonds.values())
                for b in all_bonds:
                    if b.equal == bond.equal:
                        self.bonds.pop(b.name)
                equiv_bonds = list(self.equiv_bonds.values())
                for b in equiv_bonds:
                    if b.equal == bond.equal:
                        self.equiv_bonds.pop(b.name)
                self.nonequiv_bonds.pop(bond.equal.name)
        return None

    @staticmethod
    def sort_bonds(bonds):

        bonds.sort(key=lambda x: (
            x.atom_1.element.number, x.atom_1.index,
            x.atom_2.element.number, x.atom_2.index,
            x.atom_2.site[0],
            x.atom_2.site[1],
            x.atom_2.site[2]
        ))
        return bonds

    def calc_adjacency_list(self):

        adjacency = {name: {} for name in self.equiv_atoms.keys()}
        for bond in self.equiv_bonds.values():
            adjacency[bond.atom_1.name][bond.atom_2.name] = [bond.atom_2, bond.type]
            adjacency[(bond.atom_2.name[0], (0, 0, 0))][(bond.atom_1.name[0], tuple(- bond.atom_2.site))] = [
                self.get_atom((bond.atom_1.name[0], tuple(- bond.atom_2.site))), bond.type
                ]
        self.adjacency_list = adjacency
        return adjacency

    def reamove_connectivity(self):

        self.nonequiv_bonds = {}
        self.equiv_bonds = {}
        self.bonds = {}
        self.adjacency_list = None
        self.symmetry.equivalent_atoms = {}
        self.symmetry.equivalent_bonds = {}
        self.symmetry.equivalent_positions = {}
        return None

    def get_neighbors(self, atom, bonding_types="V Sp vdW Hb"):

        neighbors = []
        if atom.name[1] == (0, 0, 0):
            for neighbor in self.adjacency_list[atom.name].values():
                if neighbor[1] in bonding_types:
                    neighbors += [neighbor[0]]
        else:
            neighbors_ = list(self.adjacency_list[(atom.name[0], (0, 0, 0))].values())
            for neighbor in neighbors_:
                if neighbor[1] in bonding_types:
                    site = tuple(neighbor[0].site + atom.site)
                    neighbors += [self.get_atom((neighbor[0].name[0], site))]
        return neighbors

    def multiply_cell(self, a=(-1, 2), b=(-1, 2), c=(-1, 2)):

        atoms = []
        bonds = []
        translations = [(t1, t2, t3)
                        for t1 in range(a[0], a[1])
                        for t2 in range(b[0], b[1])
                        for t3 in range(c[0], c[1])]
        for atom in self.equiv_atoms.values():
            for t in translations:
                name = (atom.name[0], t)
                atoms.append(self.get_atom(name))
        for bond in self.equiv_bonds.values():
            for t in translations:
                name = ((bond.name[0][0], t), (bond.name[1][0], tuple(bond.atom_2.site + t)),  bond.name[2])
                bonds.append(self.get_bond(name))
        return atoms, bonds

    def extend_unit_cell(self, ka, kb, kc):

        self.cell.extend(ka, kb, kc)
        atoms, bonds = self.multiply_cell(a=(0, ka), b=(0, kb), c=(0, kc))
        self.remove_symmetry()
        atoms_dict = {}
        for atom in atoms:
            if (atom.site != (0, 0, 0)).any():
                atoms_dict[atom.name] = (self.increment_element_index(atom.element.symbol), atom)
            else:
                atom.equel = atom
                atoms_dict[atom.name] = (atom.index, atom)
        for atom in self.atoms.values():
            c, t = Symmetry.calc_coord_and_translation((atom.fract_coord + atom.site) / [ka, kb, kc])
            atom.index, atom.equel = atoms_dict[(atom.name[0], tuple(atom.site + t * [ka, kb, kc]))]
            atom.fract_coord = c
            atom.site = - t
            atom.translation = np.array(atom.site)
        for atom in self.atoms.values():
            atom.assign_name()
        for bond in self.bonds.values():
            bond.assign_name()
        self.nonequiv_atoms = {atom.name: atom for atom in atoms}
        self.equiv_atoms = {atom.name: atom for atom in atoms}
        self.atoms = {atom.name: atom for atom in self.atoms.values()}
        self.nonequiv_bonds = {bond.name: bond for bond in bonds}
        self.equiv_bonds = {bond.name: bond for bond in bonds}
        self.bonds = {bond.name: bond for bond in self.bonds.values()}
        return

    def remove_symmetry(self):

        atoms_dict = {}
        self.symmetry = Symmetry(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]), np.array([[0., 0., 0.]]))
        self.structure_data.symmetry_equiv_pos_as_xyz = ["x,y,z"]
        self.equiv_positions = {}
        self.elements_count = {}
        atoms = self.sort_atoms(list(self.equiv_atoms.values()))
        for atom in atoms:
            atoms_dict[atom.name] = atom
            atom.index = self.increment_element_index(atom.element.symbol)
            atom.translation = np.array([0, 0, 0])
            atom.symmetry = 1
            atom.equal = atom
            atom.assign_name()
        for atom in self.atoms.values():
            if np.any(atom.site != [0, 0, 0]):
                atoms_dict[atom.name] = atom
                atom.index = atoms_dict[(atom.name[0], (0, 0, 0))].index
                atom.symmetry = 1
                atom.translation = atom.site
                atom.assign_name()
        self.atoms = {atom.name: atom for atom in self.atoms.values()}
        self.equiv_atoms = {atom.name: atom for atom in atoms}
        self.nonequiv_atoms = {atom.name: atom for atom in atoms}
        bonds_dict = {bond.name: bond for bond in self.bonds.values()}
        self.bonds = {bond.assign_name(): bond for bond in self.bonds.values()}
        self.equiv_bonds = {bond.assign_name(): bond for bond in self.equiv_bonds.values()}
        self.nonequiv_bonds = {bond.assign_name(): bond for bond in self.equiv_bonds.values()}
        self.calc_adjacency_list()
        return atoms_dict, bonds_dict

    def find_struct_groups(self, bonding_types='V', nonequiv_group=True):

        indexes = {}
        traversed = {}
        if nonequiv_group:
            atoms_list = list(self.nonequiv_atoms.values())
        else:
            atoms_list = list(self.equiv_atoms.values())
        atoms_list = self.sort_atoms(atoms_list)

        for atom in atoms_list:
            if not traversed.get(atom.name[0]):

                atoms = {atom.name: atom}
                bonds = {}
                translations = []
                sgroup_atoms = list(atoms.values())
                for a in sgroup_atoms:
                    if not traversed.get(a.name[0]):
                        traversed[a.name[0]] = a
                        for neighbor in self.get_neighbors(a, bonding_types):
                            if not (traversed.get(neighbor.name[0])):
                                atoms[neighbor.name] = neighbor
                                sgroup_atoms.append(neighbor)
                                bond = self.get_bond((a.name, neighbor.name, 'V'))
                                bonds[bond.name] = bond
                            elif (traversed.get(neighbor.name[0]).site != neighbor.site).any():
                                trans_equiv_atom = traversed.get(neighbor.name[0])
                                translations.append(trans_equiv_atom.site - neighbor.site)
                                bond = self.get_bond((a.name, neighbor.name, 'V'))
                                bonds[bond.name] = bond

                sgroup_atoms = self.sort_atoms(sgroup_atoms)
                label = sgroup_atoms[0].name[0][:2]
                group = StructuralGroup(atoms, bonds, translations, 1, sgroup_atoms[0].symmetry)
                if indexes.get(group.formula):
                    if not indexes[group.formula].get(label):
                        indexes[group.formula].update([(label, len(indexes[group.formula]) + 1)])
                else:
                    indexes[group.formula] = {label: 1}
                group.index = indexes[group.formula][label]
                group.assign_name()
                self.structural_groups[group.name] = group
        return self.structural_groups


    def __str__(self):

        data = "data_" + self.structure_data.block_name + '\n'
        data += "_cell_length_a                      " + str(self.cell.a) + '\n'
        data += "_cell_length_b                      " + str(self.cell.b) + '\n'
        data += "_cell_length_c                      " + str(self.cell.c) + '\n'
        data += "_cell_angle_alpha                   " + str(self.cell.alpha) + '\n'
        data += "_cell_angle_beta                    " + str(self.cell.beta) + '\n'
        data += "_cell_angle_gamma                   " + str(self.cell.gamma) + '\n'
        data += ("loop_\n"
                 + "_space_group_symop_id\n"
                 + "_space_group_symop_operation_xyz\n")
        for j in range(len(self.structure_data.symmetry_equiv_pos_as_xyz)):
            data += (str(j + 1) + ' ' + str(self.structure_data.symmetry_equiv_pos_as_xyz[j]).replace(' ', '') + '\n')
        data += ("loop_\n"
                 + "_atom_site_label\n"
                 + "_atom_site_type_symbol\n"
                 + "_atom_site_fract_x\n"
                 + "_atom_site_fract_y\n"
                 + "_atom_site_fract_z\n")
        atoms = list(self.nonequiv_atoms.values())
        atoms = self.sort_atoms(atoms)
        for atom in atoms:
            data += (atom.element.symbol + str(atom.index) + ' '
                     + atom.element.symbol + ' '
                     + '%6.5f' % (atom.fract_coord[0]) + ' '
                     + '%6.5f' % (atom.fract_coord[1]) + ' '
                     + '%6.5f' % (atom.fract_coord[2]) + '\n')
        bonds = list(self.nonequiv_bonds.values())
        if len(bonds) != 0:
            data += ("loop_\n"
                     + "_topol_link.node_label_1\n"
                     + "_topol_link.node_label_2\n"
                     + "_topol_link.site_symmetry_symop_1\n"
                     + "_topol_link.site_symmetry_translation_1_x\n"
                     + "_topol_link.site_symmetry_translation_1_y\n"
                     + "_topol_link.site_symmetry_translation_1_z\n"
                     + "_topol_link.site_symmetry_symop_2\n"
                     + "_topol_link.site_symmetry_translation_2_x\n"
                     + "_topol_link.site_symmetry_translation_2_y\n"
                     + "_topol_link.site_symmetry_translation_2_z\n"
                     + "_topol_link.type\n")
            bonds.sort(key=(lambda x: ((x.atom_1.element.number, x.atom_1.index),
                                       (x.atom_2.element.number, x.atom_2.index))))
            for bond in bonds:
                data += (bond.atom_1.element.symbol + str(bond.atom_1.index) + ' '
                         + bond.atom_2.element.symbol + str(bond.atom_2.index) + ' '
                         + str(bond.atom_1.symmetry) + ' '
                         + str(bond.atom_1.translation[0]) + ' '
                         + str(bond.atom_1.translation[1]) + ' '
                         + str(bond.atom_1.translation[2]) + ' '
                         + str(bond.atom_2.symmetry) + ' '
                         + str(bond.atom_2.translation[0]) + ' '
                         + str(bond.atom_2.translation[1]) + ' '
                         + str(bond.atom_2.translation[2]) + ' '
                         + bond.type + '\n')
        data += "#End of " + self.structure_data.block_name + '\n'
        return data

    def to_poscar(self, fract=True, symbols=True, scale=1.0, title="Generated by IonExplorer 1.0"):

        poscar = title + '\n'
        poscar += "%2.1f" % scale + '\n'
        poscar += ("{:20.10f}{:20.10f}{:20.10f}\n".format(*self.cell.vectors[0]) +
                   "{:20.10f}{:20.10f}{:20.10f}\n".format(*self.cell.vectors[1]) +
                   "{:20.10f}{:20.10f}{:20.10f}\n".format(*self.cell.vectors[2]))
        atoms = list(self.equiv_atoms.values())
        atoms.sort(key=lambda a: (a.element.number, a.element.symbol))
        composition = self.calc_compositions(atoms)
        if symbols:
            poscar += ''.join(["{:>5s}".format(s) for s, n in composition]) + '\n'
        poscar += ''.join(["{:>5d}".format(n) for s, n in composition]) + '\n'
        if fract:
            poscar += "Direct\n"
            poscar += ''.join(["{:16.10f}{:16.10f}{:16.10f}\n".format(*a.fract_coord) for a in atoms])
        else:
            poscar += "Cartesian\n"
            poscar += ''.join(["{:16.10f}{:16.10f}{:16.10f}\n".format(*a.cart_coord) for a in atoms])
        return poscar




