import re
import numpy as np


class Symmetry:

    def __init__(self, rotations, translations):

        self.rotations = rotations
        self.translations = translations
        self.symm_operations = [[rotations[i], translations[i]] for i in range(len(rotations))]
        self.order = len(self.rotations)
        self.equivalent_positions = {}
        self.equivalent_atoms = {}
        self.equivalent_bonds = {}

    def copy(self):

        copy = Symmetry([], [])
        copy.rotations = np.array(self.rotations)
        copy.translations = np.array(self.translations)
        copy.symm_operations = [[copy.rotations[i], copy.translations[i]] for i in range(len(copy.rotations))]
        copy.order = self.order
        copy.equivalent_positions = {k: np.array(v) for k, v in self.equivalent_positions.items()}
        copy.equivalent_atoms = {k: np.array(v) for k, v in self.equivalent_atoms.items()}
        copy.equivalent_bonds = {k: np.array(v) for k, v in self.equivalent_bonds.items()}
        return copy

    @staticmethod
    def pars_sym_code(sym):

        # print(sym)
        r = re.compile(
            ("(\-)?\+?(\d\/\d|\d+\.\d+|[x,y,z])"
             + "(\+|\-)?(\d\/\d|\d+\.\d+|[x,y,z])?"
             + "(\+|\-)?(\d\/\d|\d+\.\d+|[x,y,z])?"
             + "(\+|\-)?(\d\/\d|\d+\.\d+|[x,y,z])?")
        )
        rot = np.zeros((3, 3), dtype='int')
        trans = np.zeros(3)
        for i, s in enumerate(sym.replace(' ', '').lower().split(',')):
            m = r.match(s)
            if m:
                for c in ((m.group(1), m.group(2)),
                          (m.group(3), m.group(4)),
                          (m.group(5), m.group(6)),
                          (m.group(7), m.group(8))):
                    sing = 1
                    if c[0] == '-':
                        sing = -1
                    if c[1] is not None and '/' in c[1]:
                        trans[i] += sing * float(c[1][0]) / float(c[1][2])
                    elif c[1] is not None and '.' in c[1]:
                        trans[i] += sing * float(c[1][0])
                    elif c[1] is not None:
                        rot[i][ord(c[1]) - ord('x')] = sing
            else:
                raise Exception("Unrecognised symmetry code : " + sym)
        # print(rot)
        # print(trans)
        return rot, trans

    @staticmethod
    def summarize_operations(op_1, op_2):

        rotation = np.dot(op_2[0], op_1[0])
        translations = [
            (op_2[1][0] + (op_2[0][0] * op_1[1]).sum()),
            (op_2[1][1] + (op_2[0][1] * op_1[1]).sum()),
            (op_2[1][2] + (op_2[0][2] * op_1[1]).sum())
        ]
        symop = [rotation, translations]
        return symop

    def get_equiv_symop(self, symop):

        rot = symop[0]
        transl, additional_transl = self.calc_coord_and_translation(symop[1])
        symmetry_index = self.get_index([rot, transl])
        return symmetry_index, -additional_transl

    @staticmethod
    def apply_symmetry(point, symop, prec=1e-5):

        new_coord = np.dot(symop[0], point)
        new_coord = new_coord + symop[1]
        new_coord = np.array([round(c) if abs(c) < prec or abs(abs(c % 1) - 1) < prec else c for c in new_coord])
        #new_coord = np.array([1. if abs(x - 1) < prec else x for x in new_coord])
        #new_coord = np.array([0. if abs(x) < prec else x for x in new_coord])
        #new_coord = np.array([-1. if abs(x + 1) < prec else x for x in new_coord])
        return new_coord

    @staticmethod
    def calc_coord_and_translation(fract_coord, prec=1e-5):

        fract_coord = [round(c) if abs(c) < prec or abs(abs(c % 1) - 1) < prec else c for c in fract_coord]
        #fract_coord = np.array([1. if abs(x - 1) < prec else x for x in fract_coord])
        #fract_coord = np.array([0. if abs(x) < prec else x for x in fract_coord])
        #fract_coord = np.array([-1. if abs(x + 1) < prec else x for x in fract_coord])
        translation = np.array([0, 0, 0])
        new_coord = np.array([0., 0., 0.])
        for i in range(len(new_coord)):
            new_coord[i], translation[i] = (lambda x: (x % 1, - int(x - (x % 1))))(fract_coord[i])
        return new_coord, translation

    def get_index(self, symm_operation, prec=0.001):

        for i, symop in enumerate(self.symm_operations):
            if (symm_operation[0] == symop[0]).all() and np.linalg.norm(symm_operation[1] - symop[1]) < prec:
                return i + 1
        return None
