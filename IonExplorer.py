import os
import argparse
from structure import *
import read_write as rw
from migration_path import find_migration_path
from path_selector import find_neighbor_positions, get_multiplying_factors
import critic2_api as cr2
from multiprocessing.dummy import Pool as ThreadPool


def run_multiproc(func, args, n_proc=1):

    pool = ThreadPool(n_proc)
    pool.map(func, args)
    pool.close()
    pool.join()
    return None


def mkdir(dir_name):

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    return None


def gen_inp_configs(structure, working_ion, min_length=0.0, max_length=float('inf'), min_dist=3):

    configs = []
    paths_info = []
    structure_copy = structure.copy()
    neighbors = find_neighbor_positions(structure_copy, [working_ion], 1, min_length, max_length)
    paths = [(a1.fract_coord + a1.site, a2.fract_coord + a2.site) for a1, a2 in neighbors]
    mult_fact = get_multiplying_factors(paths, structure.cell.a, structure.cell.b, structure.cell.c, min_dist)
    structure_copy.extend_unit_cell(*mult_fact)
    for a1, a2 in neighbors:
        config = structure_copy.copy()
        shift = [0.5, 0.5, 0.5] - (a1.fract_coord + a2.fract_coord + a2.site) / 2.
        for atom in config.nonequiv_atoms.values():
            atom.fract_coord = Symmetry.calc_coord_and_translation(atom.fract_coord + shift)[0]
        start, end = config.get_atom(a1.name), config.get_atom(a2.name)
        config.remove_atoms([start, end])
        paths_info += [
            (start.fract_coord, end.fract_coord, np.linalg.norm(start.cart_coord - end.cart_coord), shift, mult_fact)
        ]
        configs += [config]
    return configs, paths_info


def get_profile(gradient_map, migration_path):

    trajectory = []
    fvals = []
    if len(migration_path) == 1:
        trajectory += [migration_path[0].fract_coord]
    else:
        for i in range(len(migration_path) - 1):
            p = gradient_map.get_bond((migration_path[i].name, migration_path[i + 1].name, 'S')).weight.copy()
            if migration_path[i].name[0] != p.cp1.name[0]:
                p = p.flip(gradient_map.get_atom)
            trajectory += [p.cp1.fract_coord]
            trajectory.extend(p.fract_coords)
            trajectory += [p.cp2.fract_coord]
            fvals += [p.cp1.properties]
            fvals.extend(p.eldens)
            fvals += [p.cp2.properties]
    return trajectory, fvals


def find_trajectories(work_dir, structure, mig_ion, min_length=0,
                      max_length=float('inf'), pdist=3, max_wind=7, n_poc=1):

    args = []
    max_cps = 2 * max_wind + 1
    configs, paths_info = gen_inp_configs(structure, mig_ion, min_length, max_length, pdist)
    for i, config in enumerate(configs):
        path_info = paths_info[i]
        c1, c2 = path_info[0], path_info[1]
        args += [(work_dir, i, c1, c2, max_cps)]
        rw.write_file(os.path.join(work_dir, str(i) + ".cif"), str(config))
        rw.write_path_data(os.path.join(work_dir, str(i) + "_path_data.txt"), *path_info)
    run_multiproc(find_trajectory, args, n_poc)
    return None

def find_trajectory(args):

    try:
        work_dir, path_ind, start, end, max_cps = args
        os.chdir(work_dir)
        path_ind = str(path_ind)
        cr2.write_input(path_ind, start, end)
        cr2.run_critic2(str(path_ind))
        structure = rw.read_structure(os.path.join(work_dir, path_ind + ".cif"))
        critic2_out = os.path.join(work_dir, path_ind + "_critic2_out.txt")
        flux_out = os.path.join(work_dir, path_ind + "_flux.txt")
        gradient_map = cr2.extract_gradient_map(structure, critic2_out, flux_out)
        rw.write_file(os.path.join(work_dir, path_ind + "_gradient_map.cif"), str(gradient_map))
        start_fval, end_fval = cr2.extract_eldens(os.path.join(work_dir, path_ind + "_critic2_out.txt"))
        rw.write_cps_data(os.path.join(work_dir, path_ind +"_cps_data.txt"), gradient_map)
        path = find_migration_path(gradient_map, start, end, tol=0.1, length_limit=max_cps)
        if path is not None:
            ps, fvals = get_profile(gradient_map, path)
            trajectory = [start, *ps, end]
            fvals = [start_fval, *fvals, end_fval]
            for p in trajectory:
                structure.add_atom(0, p)
            rw.write_file(os.path.join(work_dir, path_ind + "_path.cif"), str(structure))
            rw.write_trajectory(os.path.join(work_dir, path_ind + "_trajectory.txt"), trajectory, fvals)
    except:
        print("The path " + path_ind + " is failed!")
    return None


def write_neb_vasp_inp(work_dir, structure, init_traject, element_numb):

    path = structure.copy()
    for p in init_traject:
        path.add_atom(element_numb, p)
    rw.write_file(os.path.join(work_dir, "POSCAR"), path.to_poscar())
    for j, p in enumerate(init_traject):
        img_dir = os.path.join(work_dir, "0" + str(j))
        mkdir(img_dir)
        img = structure.copy()
        img.add_atom(element_numb, p)
        rw.write_file(os.path.join(img_dir, "POSCAR"), img.to_poscar())
    return None


def gen_neb_vasp_inp(work_dir, structure, mig_ion, min_length=0.0,
                     max_length=float('inf'), pdist=3, n_imgs=None, d_imgs=0.5):

    configs, paths_info = gen_inp_configs(structure, mig_ion, min_length, max_length, pdist)
    for i, config in enumerate(configs):
        path_info = paths_info[i]
        start, end, length = path_info[0], path_info[1], path_info[2]
        path_dir = os.path.join(work_dir, "path_" + str(i))
        mkdir(path_dir)
        rw.write_path_data(os.path.join(path_dir, str(i) + "_path_data.txt"), *path_info)
        if n_imgs is None:
            n_imgs = int(np.ceil(length) / d_imgs)
        init_traject = [(end - start) * (j / float(n_imgs - 1)) + start for j in range(n_imgs)]
        write_neb_vasp_inp(path_dir, config, init_traject, mig_ion)
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="IonExplorer a tool for the analysis of ion migration pathways.")

    parser.add_argument("-i", "--input_file", nargs="?", type=str, default=None,
                       help="--input_file: path to input cif file.")

    parser.add_argument("-m", "--migration_ion", nargs="?", type=str, default=None,
                        help="migration_ion: Atomic symbol of the migration ion.")

    parser.add_argument("-j", "--job_type", nargs="?", type=int, default=1,
                        help=("job_type: 0-generation of input files for VASP NEB calculation," +
                              "1-search for the ion migration trajectories by analysis of the " +
                              "electron density distribution."))

    parser.add_argument("-l", "--length_interval", nargs=2, type=float, default=(0., float('inf')),
                        help=("length_interval: The minimal and maximal length between " +
                              "the initial and final position of the migration ion."))

    parser.add_argument("-d", "--min_dist", nargs="?", type=int, default=3.,
                        help="min_dist: The minimal distances between the translation equivalent migration paths")

    parser.add_argument("-n", "--num_img", nargs="?", type=int, default=None,
                        help="num_img: The number of images.")

    parser.add_argument("-b", "--img_dist", nargs="?", type=float, default=0.5,
                        help="img_dist: The distance between images. Valid if num_img is not specified.")

    parser.add_argument("-p", "--num_proc", nargs="?", type=int, default=1,
                        help="num_proc: The number of processes.")

    parser.add_argument("-w", "--max_wind", nargs="?", type=int, default=3,
                        help="max_wind: The maximal number of windows that a migration path can cross. " +
                             "The criteria apply in analysis of the gradient paths")

    args = parser.parse_args()
    file_name = args.input_file
    if file_name is None:
        raise ValueError("The path to the input file is not specified!")
    mig_ion = get_number(args.migration_ion)
    if mig_ion is None:
        raise ValueError("The working ions are not specified!")
    min_length, max_length = args.length_interval
    min_dist = args.min_dist
    num_img = args.num_img
    img_dist = args.img_dist
    num_proc = args.num_proc
    max_wind = args.max_wind
    job_type = args.job_type
    cwd = os.getcwd()
    for i, structure in enumerate(rw.read_structures(os.path.join(cwd, file_name))):
        structure_dir = os.path.join(cwd, "structure_" + str(i))
        mkdir(structure_dir)
        if job_type == 0:
            work_dir = os.path.join(structure_dir, "vasp_neb")
            mkdir(work_dir)
            gen_neb_vasp_inp(work_dir, structure, mig_ion, min_length, max_length, min_dist, num_img, img_dist)
        elif job_type == 1:
            work_dir = os.path.join(structure_dir, "paths")
            mkdir(work_dir)
            find_trajectories(work_dir, structure, mig_ion, min_length, max_length, min_dist, max_wind, num_proc)
        else:
            raise ValueError("Unrecognized job type!")

