import numpy as np
from functools import reduce
from scipy.spatial.distance import cdist

def calc_dist(ps1, ps2):

    return np.sqrt(np.sum((ps1-ps2)**2, axis=1))


def calc_angles(ps1, ps2, ps3):

    vs1 = np.array(ps1) - ps2
    vs2 = np.array(ps3) - ps2
    a = np.array(map(np.linalg.norm, vs1)) * map(np.linalg.norm, vs2)
    a[a == 0] = None
    with np.warnings.catch_warnings():
        cos = np.array([np.dot(v1, v2) / a for v1, v2, a in zip(vs1, vs2, a)])
        cos[cos > 1] = 1.0
        cos[cos < -1] = - 1.0
        angles = np.arccos(cos)
        angles[angles > np.pi] = np.pi
    return angles


def calc_dihedral(ps1, ps2, ps3, ps4):

    """ [0 : 180] """

    vs1 = -1 * (ps2 - ps1)
    vs2 = ps3 - ps2
    vs3 = ps4 - ps3
    v1 = [v1 - np.dot(v1, v2) / np.dot(v2, v2) * v2 for v1, v2 in zip(vs1, vs2)]
    v2 = [v3 - np.dot(v3, v2) / np.dot(v2, v2) * v2 for v3, v2 in zip(vs3, vs2)]
    a = np.array(map(np.linalg.norm, v1)) * map(np.linalg.norm, v2)
    a[a == 0] = None
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r".*encountered")
        cos = np.array([np.dot(v1, v2) / a for v1, v2, a in zip(v1, v2, a)])
        cos[cos > 1] = 0.0
        cos[cos < -1] = -1.0
        dihedral = np.arccos(cos)
        dihedral[dihedral > np.pi] = np.pi
        
    return dihedral


def calc_dihedral2(ps1, ps2, ps3, ps4):

    """ Clockwise [0 : 180] counter-clockwise (0 : -180) """

    b0 = -1.0 * (ps2 - ps1)
    b1 = ps3 - ps2
    b2 = ps4 - ps3
    b1 = np.array(map(lambda x: x / np.linalg.norm(x), b1))
    v = b0 - map(lambda x: np.dot(x[0], x[1]) * x[1], zip(b0, b1))
    w = b2 - map(lambda x: np.dot(x[0], x[1]) * x[1], zip(b2, b1))
    x = map(lambda x: np.dot(x[0], x[1]), zip(v, w))
    c = map(lambda x: np.cross(x[0], x[1]), zip(b1, v))
    y = map(lambda x: np.dot(x[0], x[1]), zip(c, w))
    dihedral = np.array(map(lambda x: np.arctan2(x[0], x[1]), zip(y, x)))
    return dihedral


def rotate(a, ps, axe=0):

    """ Turns coordinate system around axe """

    sin = np.sin(a)
    cos = np.cos(a)
    rm = np.array([[[1, 0, 0], [0, cos, -sin], [0, sin,  cos]],
                  [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]],
                  [[cos, -sin, 0], [sin, cos, 0], [0,  0, 1]]])
    m = np.full((len(ps), 3, 3), rm[axe])
    ps = map(lambda x, y: np.dot(x, y), m, ps)
    return ps


def calc_centroid(ps):
    
    centroid = reduce(lambda x, y: x + y, np.array(ps)) / len(ps)
    return centroid



def orient(ps, origin, v1, v2):

    """ Orients the coordinate system (v1 along Z axis, v2 along X axis) """
    
    ps = np.vstack((v1, v2, ps))
    ps -= origin
    if ps[0][1] == 0:
        a = 0
    else:
        a = np.arcsin(np.fabs(ps[0][1]) / np.sqrt(ps[0][1] ** 2 + ps[0][2] ** 2))
    if (ps[0][1] < 0 <= ps[0][2]) or (ps[0][1] > 0 > ps[0][2]):
        a = 2 * np.pi - a
    if (ps[0][1] * np.sin(a) + ps[0][2] * np.cos(a)) < 0:
        a = np.pi + a  
    ps = rotate(a, ps, 0)
    if ps[0][0] == 0:
        b = 0
    else:
        b = np.arcsin(np.fabs(ps[0][0]) / np.sqrt(ps[0][0] ** 2 + ps[0][2] ** 2))
    if (ps[0][0] < 0 and ps[0][2] < 0) or (ps[0][0] > 0 and ps[0][2] > 0):
        b = 2 * np.pi - b
    if (ps[0][2] * np.cos(b) - ps[0][0] * np.sin(b)) < 0:
        b = np.pi + b
    ps = rotate(b, ps, 1)
    if ps[1][1] == 0:
        c = 0
    else:
        c = np.arcsin(np.fabs(ps[1][1]) / np.sqrt(ps[1][0]**2 + ps[1][1]**2))
    if (ps[1][0] < 0 and ps[1][1] < 0) or (ps[1][0] > 0 and ps[1][1] > 0):
        c = 2 * np.pi - c
    if (ps[1][0] * np.cos(c) - ps[1][1] * np.sin(c)) < 0:
        c = np.pi + c
    ps = rotate(c, ps, 2)
    return ps[2:]


def are_coplanar(vs, prec=0.01):
    
    coplanar = True
    if len(vs) < 3:
        return coplanar
    else:
        for i in range(len(vs)-2):
            for j in range(i+1, len(vs)-1):
                ab = np.cross(vs[i], vs[j])
                for l in range(j+1, len(vs)):
                    if abs(np.dot(ab, vs[l])) > prec:
                        coplanar = False
                        return coplanar


def calc_vectors(a, b, c, alpha, betta, gamma):

    alpha = np.radians(alpha % 180)
    betta = np.radians(betta % 180)
    gamma = np.radians(gamma % 180)
    c1 = c * np.cos(betta)
    c2 = c * (np.cos(alpha) - np.cos(gamma) * np.cos(betta)) / np.sin(gamma)
    c3 = np.sqrt(c * c - c1 * c1 - c2 * c2)
    m = np.array([[a, 0., 0.],
                  [b * np.cos(gamma), b * np.sin(gamma), 0.],
                  [c1, c2, c3]])
    return m

def find_equal_ps(ps1, ps2, tol=0.1):

    dists = cdist(ps1, ps2)
    inds = np.argmin(dists, axis=1)
    inds = np.array([ind if dists[i][ind] < tol else None for i, ind in enumerate(inds)])
    return inds

