from CifFile import ReadCif
from structure import *
from structure_data import StructureData
import os


def read_file(file_name, out_list=True):

    with open(file_name, 'r') as open_file:
        if out_list:
            text = open_file.readlines()
        else:
            text = open_file.read()
    return text


def write_file(file_name, text, key='w'):

    with open(file_name, key) as new_file:
        if text is list:
            new_file.writelines(text)
        else:
            new_file.write(text)
    return None


def read_structure(path_to_file):

    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(path_to_file))
        file_name = os.path.split(path_to_file)[-1]
        cif_data = ReadCif(file_name)
        os.chdir(cwd)
    except:
        os.chdir(cwd)
        raise ValueError("The reading or parsing of the " + file_name + " is failed!")
    try:
        name, data = list(cif_data.items())[0]
        structure_data = StructureData(name, data)
        structure = Structure().build_structure(structure_data)
        structure.reamove_connectivity()
        return structure
    except:
        print("Reading structure is failed!")


def read_structures(path_to_file):

    cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(path_to_file))
        file_name = os.path.split(path_to_file)[-1]
        cif_data = ReadCif(file_name)
        os.chdir(cwd)
    except:
        os.chdir(cwd)
        raise ValueError("The reading or parsing of the " + file_name + " is failed!")
    for i, (name, data) in enumerate(cif_data.items()):
        try:
            structure_data = StructureData(name, data)
            structure = Structure().build_structure(structure_data)
            structure.reamove_connectivity()
            yield structure
        except:
            os.chdir(cwd)
            print("Reading " + str(i) + " structure is failed!")


def write_path_data(file_name, start, end, length, shift, mult_fact):

    path_info = ("start: {:11.10f} {:11.10f} {:11.10f}\n".format(*start) +
                 "end: {:11.10f} {:11.10f} {:11.10f}\n".format(*end) +
                 "length: {:7.4f}\n".format(length) +
                 "shift: {} {} {}\n".format(*shift) +
                 "supercell: {} {} {}".format(*mult_fact))
    write_file(file_name, path_info)
    return None


def write_cps_data(file_name, gradien_map):

    cps_data = "{}\t{}\t{}\t{}\t{}\n".format("CPs", 'X', "Y", "Z", "FIELD_VALUE")
    for a in gradien_map.nonequiv_atoms.values():
        if a.properties is not None:
            cps_data += ("{}\t".format(a.element.symbol + str(a.index)) +
                         "{:11.10f}\t{:11.10f}\t{:11.10f}\t{:11.10f}\n".format(*a.fract_coord, a.properties))
    write_file(file_name, cps_data)
    return None


def write_trajectory(file_name, trajectory, fvals):

    data = "{}\t{}\t{}\t{}\n".format("X", "Y", "Z", "FIELD_VALUE")
    for j in range(len(trajectory)):
        data += "{:11.10f}\t{:11.10f}\t{:11.10f}\t{:11.10f}\n".format(*trajectory[j], fvals[j])
    write_file(file_name, data)
    return None