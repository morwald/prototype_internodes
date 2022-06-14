#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as sp
from scipy import spatial

def nodes_to_dofs(nodes):
    return np.column_stack([2*nodes, 2*nodes+1]).reshape([-1, 1])

def find_contact_nodes(nodes1i, nodes2i, coords1i, coords2i):
    radiuses1, nnzR21 = compute_radiuses(nodes1i, nodes2i, coords1i, coords2i)
    radiuses2, nnzR12 = compute_radiuses(nodes2i, nodes1i, coords2i, coords1i)

    nodes1i_mask = nnzR12 > 0
    nodes2i_mask = nnzR21 > 0

    while np.any(nodes1i_mask == False) or np.any(nodes2i_mask == False):
        nodes1i = nodes1i[nodes1i_mask]
        nodes2i = nodes2i[nodes2i_mask]

        coords1i = coords1i[nodes1i_mask]
        coords2i = coords2i[nodes2i_mask]

        dofs1i = nodes_to_dofs(nodes1i).ravel()
        dofs2i = nodes_to_dofs(nodes2i).ravel()

        radiuses1, nnzR21 = compute_radiuses(nodes1i, nodes2i, coords1i, coords2i)
        radiuses2, nnzR12 = compute_radiuses(nodes2i, nodes1i, coords2i, coords1i)

        nodes1i_mask = nnzR12 > 0
        nodes2i_mask = nnzR21 > 0

    return nodes1i, nodes2i, coords1i, coords2i, radiuses1, radiuses2

def compute_radiuses(nodes1i, nodes2i, coords1i, coords2i):
    c = 0.5 # conditition (2)
    C = 0.95 # condition (3)
    n = 1 # consider n nearest neighboors
    d = 0.05 # tolerance, for radius of "attack" estimation

    M = len(coords1i)
    N = len(coords2i)

    radiuses = np.zeros(M)
    nnzRMM = np.zeros(M)
    nnzRNM = np.zeros(N)
    nnzCMM = np.zeros(M)
    nnzCNM = np.zeros(M)

    maxS = np.inf
    f = 0
    niter = 0
    maxiter = 10

    while maxS > f and niter < maxiter-1:
        f = np.floor(1/(np.power(1-c, 4)*(1+4*c))) # maximum number of supports

        for k in range(M):
            point = coords1i[k, :].reshape(1, -1)
            neighbors = coords1i.copy()
            neighbors[k, :] = np.inf
            distMM = spatial.distance.cdist(neighbors, point).ravel()
            distMN = spatial.distance.cdist(coords2i, point).ravel()

            rMM = np.min(distMM)
            rNM = np.sqrt(d*d + 0.25*np.power(rMM, 2))
            radius = np.maximum(rMM, rNM)

            # rMM = distMM[np.argpartition(distMM, n)[:n]]
            # rNM = np.sqrt(d*d + 0.25*np.power(rMM, 2))
            # radius = np.maximum([rMM[-1], rNM])

            # if radius > rMM[0]/c:
            #     radius = rMM[0]/c

            if radius > rMM/c:
                radius = rMM/c

            s1 = distMM < radius
            s2 = distMN < C*radius

            nnzRMM[s1] = nnzRMM[s1] + 1
            nnzRNM[s2] = nnzRNM[s2] + 1
            nnzCMM[k] = np.sum(s1)
            nnzCNM[k] = np.sum(s2)

            radiuses[k] = radius

        maxS = np.max(nnzRMM)

        if maxS > f:
            c = 0.5 * (1 + c);
            nnzRMM = np.zeros(M)
            nnzRNM = np.zeros(N)
            nnzCMM = np.zeros(M)
            nnzCNM = np.zeros(M)
            niter = niter+1

    return radiuses, nnzRNM

def wendland(dists, radiuses):
    result = np.zeros(len(dists))

    mask = dists <= radiuses
    result[mask] = np.power(1-dists[mask]/radiuses[mask], 4) * (1+4*dists[mask]/radiuses[mask])
    return result

def phi_constructor(coords_i, coords_j, radiuses_j, rad_func):
    N = len(coords_i)
    M = len(coords_j)

    dists = spatial.distance.cdist(coords_i, coords_j)
    radiuses_j = np.tile(radiuses_j, N)
    phi = rad_func(dists.ravel(), radiuses_j.ravel())

    return phi.reshape([N, M])

def Rij_constructor(coords_i, coords_j, radiuses_j):
    phiMM = phi_constructor(coords_j, coords_j, radiuses_j, wendland)
    phiNM = phi_constructor(coords_i, coords_j, radiuses_j, wendland)

    Rij = phiNM.dot(np.linalg.inv(phiMM))
    g = Rij.dot(np.ones((Rij.shape[1], 1)))
    Rij_norm = Rij * (1/g)
    return Rij_norm

def assemble_Rijs(coords1i, coords2i, radiuses1, radiuses2):
    R12_normal = Rij_constructor(coords1i, coords2i, radiuses2)
    R12 = sp.sparse.csr_matrix(extend_to_2D(R12_normal))

    R21_normal = Rij_constructor(coords2i, coords1i, radiuses1)
    R21 = sp.sparse.csr_matrix(extend_to_2D(R21_normal))

    return R12_normal, R21_normal, R12, R21

def extend_to_2D(R):
    R_extended = np.repeat(np.repeat(R,2,axis=1), 2, axis=0)
    R_extended[1::2,::2] = 0
    R_extended[::2,1::2] = 0
    return R_extended

def remove_traction(coords_new, connectivity1b, connectivity2b, connectivity1b_body, connectivity2b_body, nodes1i, nodes2i, nodes1b, nodes2b, lambda1, R12, R21):
    normals1b = compute_normals(coords_new, nodes1b, connectivity1b, connectivity1b_body)
    normals2b = compute_normals(coords_new, nodes2b, connectivity2b, connectivity2b_body)

    normals1i = normals1b[np.in1d(nodes1b, nodes1i)]
    normals2i = normals2b[np.in1d(nodes2b, nodes2i)]

    lambda2 = -R21.dot(lambda1.reshape([-1, 1])).reshape([-1, 2])

    scalar1 = np.sum(lambda1*normals1i, axis=1)
    scalar2 = np.sum(lambda2*normals2i, axis=1)

    nodes1i_dump = nodes1i[scalar1>0]
    nodes2i_dump = nodes2i[scalar2>0]

    if len(nodes1i_dump) == 0 and len(nodes2i_dump) == 0:
        # gap verification
        nodes1i_add, nodes2i_add = detect_gaps(coords_new, nodes1i, nodes2i, normals1i, normals2i)

        nodes1i, diff_nb_nodes1i = update_interface(nodes1i_add, nodes1i, 'add')
        nodes2i, diff_nb_nodes2i = update_interface(nodes2i_add, nodes2i, 'add')
    else:
        nodes1i, diff_nb_nodes1i = update_interface(nodes1i_dump, nodes1i, 'dump')
        nodes2i, diff_nb_nodes2i = update_interface(nodes2i_dump, nodes2i, 'dump')

    print(diff_nb_nodes1i, ' nodes removed from interface 1')
    print(diff_nb_nodes2i, ' nodes removed from interface 2')

    return nodes1i, nodes2i, diff_nb_nodes1i, diff_nb_nodes2i

def update_interface(new_nodes, nodesi, case):
    if case == 'dump':
        nodesi_new = nodesi[~np.in1d(nodesi, new_nodes)]
    if case == 'add':
        nodesi_new = np.union1d(nodesi, new_nodes)

    diff_nb_nodes = len(nodesi) -len(nodesi_new)
    return nodesi_new, diff_nb_nodes

def compute_normals(coords_new, nodesb, connectivityb, connectivityb_body):
    n = len(nodesb)
    m = len(connectivityb)

    connectivityi_body = connectivityb_body[np.in1d(connectivityb_body, nodesb).reshape(connectivityb_body.shape).any(axis=1)]
    nodesb_body = np.unique(connectivityi_body[~np.isin(connectivityb_body, nodesb)])

    tangents = coords_new[connectivityb[:, 1]] - coords_new[connectivityb[:, 0]]
    lengths = np.linalg.norm(tangents, axis=1).reshape([-1,1])
    tangents = tangents/lengths

    normals = np.zeros((m, 2))
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    normals_avg = np.zeros((n, 2))
    gamma = 1e-3 # step size
    for j in range(n):
        node = nodesb[j]
        coord = coords_new[node, :]
        id = np.in1d(connectivityb, node).reshape(connectivityb.shape).any(axis=1)
        length = lengths[id]
        normal_avg = 1/np.sum(length)*np.sum(normals[id, :]*length, axis=0)

        tang_plus = (coord + gamma*normal_avg).reshape([-1, 2])
        tang_minus = (coord - gamma*normal_avg).reshape([-1, 2])

        min_plus = np.min(spatial.distance.cdist(coords_new[nodesb_body, :], tang_plus).ravel())
        min_minus = np.min(spatial.distance.cdist(coords_new[nodesb_body, :], tang_minus).ravel())

        if min_plus > min_minus:
            normals_avg[j, :] = normal_avg
        else:
            normals_avg[j, :] = -normal_avg

    norms = np.linalg.norm(normals_avg, axis=1).reshape([-1, 1])
    normals_avg = (normals_avg/norms).reshape([-1, 2])
    return normals_avg

def detect_gaps(coords_new, nodes1i, nodes2i, normals1i, normals2i):
    tol = 0.9 # tolerance for gap detection
    h = 0.05 # mesh size
    coords1i = coords_new[nodes1i, :]
    coords2i = coords_new[nodes2i, :]

    nodes1i, nodes2i, coords1i, coords2i, radiuses1, radiuses2 = find_contact_nodes(nodes1i, nodes2i, coords1i, coords2i)

    R21_normal = Rij_constructor(coords2i, coords1i, radiuses1)
    R21 = sp.sparse.csr_matrix(extend_to_2D(R21_normal))

    R12_normal = Rij_constructor(coords1i, coords2i, radiuses2)
    R12 = sp.sparse.csr_matrix(extend_to_2D(R12_normal))

    diffs1 = R12.dot(coords2i.reshape([-1, 1])) - coords1i.reshape([-1, 1])
    diffs1 = diffs1.reshape([-1, 2])
    diffs2 = R21.dot(coords1i.reshape([-1, 1])) - coords2i.reshape([-1, 1])
    diffs2 = diffs2.reshape([-1, 2])

    scalar1 = np.sum(diffs1*normals1i, axis=1)
    scalar2 = np.sum(diffs2*normals2i, axis=1)

    threshold = -tol*h

    nodes1i_add = nodes1i[scalar1<threshold]
    nodes2i_add = nodes2i[scalar2<threshold]

    return nodes1i_add, nodes2i_add
