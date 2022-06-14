#!/usr/bin/env python
# coding: utf-8

import akantu as aka
import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve

def assemble_interface_masses(model, dofs1i, dofs2i):
    """Assembles mass matrices for interfaces.

    :param model: akantu solid mechanics model
    :param dofs1i: dofs of body 1 interface (master)
    :param dofs2i: dofs of body 2 interface (slave)
    :returns: sparse matrices of interface mass
    """
    # assemble and store mass
    model.assembleMass()
    M_aka = model.getDOFManager().getMatrix('M')
    M = sp.sparse.lil_matrix(aka.AkantuSparseMatrix(M_aka))

    # select only the dofs of the interface
    M1i = M[np.ix_(dofs1i, dofs1i)]
    M2i = M[np.ix_(dofs2i, dofs2i)]

    M1i = sp.sparse.csc_matrix(M1i)
    M2i = sp.sparse.csc_matrix(M2i)

    return M1i, M2i

def assemble_Bs(M1i, M2i, R12, R21, indx1i, indx2i,
        nb_free_dofs, nb_constraint_dofs):
    """Assembles B and B_tilde for internodes matrix.

    :param M1i: sparse matrix interface mass 1 
    :param M2i: sparse matrix interface mass 2 
    :param R12: interpolation matrix master to slave
    :param R21: interpolation matrix slave to master
    :param indx1i: global indices of interface 1 (master)
    :param indx2i: global indices of interface 2 (slave)
    :param nb_free_dofs: number of non blocked dofs 
    :param nb_constraint_dofs: number of master dofs which act as constraint 
    :returns: sparse blocke matrices for internodes formulation 
    """

    B = sp.sparse.csr_matrix(np.zeros((nb_free_dofs, nb_constraint_dofs)))
    B_tilde = sp.sparse.csr_matrix(np.zeros((nb_constraint_dofs, nb_free_dofs)))
    C = sp.sparse.csr_matrix(np.zeros((nb_constraint_dofs, nb_constraint_dofs)))

    B[indx1i, :] = - M1i
    B[indx2i, :] = M2i * R21

    B_tilde[:, indx1i] = sp.sparse.eye(nb_constraint_dofs)
    B_tilde[:, indx2i] = - R12

    return B, B_tilde, C

def assemble_A_explicit(K_free, B, B_tilde, C):
    """Explicitly assembles internodes matrix.

    :param K_free: stiffness matrix with non blocked dofs
    :param B: B matrix of internodes formulation
    :param B_tilde: B_tilde matrix of internodes formulation
    :param C: all zero matrix of internodes formulation
    """
    A1 = sp.sparse.hstack([K_free, B])
    A2 = sp.sparse.hstack([B_tilde, C])

    A = sp.sparse.vstack([A1, A2])
    A = sp.sparse.csr_matrix(A)

    return A


def assemble_b(f_free, R12_normal, positions1i, positions2i,
        nb_free_dofs, nb_constraint_dofs, rescaling):
    """Assemble right hand side of Ax=b.
    :param f_free: force vector of free dofs 
    :param R12_normal: nodal interpolation matrix master to slave
    :param positions1i: initial positions of interface 1 nodes 
    :param positions2i: initial positions of interface 2 nodes 
    :param nb_free_dofs: number of non blocked dofs 
    :param nb_constraint_dofs: number of master dofs which act as constraint
    :param nb_dofs: total number of dofs
    :param rescaling: rescaling stiffness matrix
    """
    b = np.zeros(nb_free_dofs + nb_constraint_dofs)

    # Dirichlet displacements
    # TODO: implement

    b[0:nb_free_dofs] = 1/rescaling * f_free #- K[free_dofs, blocked_dofs]*positions[blocked_dofs]
    b[nb_free_dofs:] = (R12_normal.dot(positions2i) - positions1i).ravel()

    return b

def solve_direct(A, b, positions, free_dofs,
        nb_constraint_dofs, nb_dofs, rescaling):
    """Direct solve of internodes equation Ax=b.

    :param A: sparse matrix internodes
    :param b: right hand side vector
    :param positions: initial positions of all nodes
    :param scaling: scaling factor for stiffness matrix
    :param free_dofs: mask of free dofs
    :param nb_constraint_dofs: number of master dofs which act as constraint
    :param nb_dofs: total number of dofs
    :param rescaling: rescaling stiffness matrix
    :returns: new positions and lambdas resulting from constraints
    """
    nb_free_dofs = len(free_dofs)
    u = np.zeros(nb_dofs)

    x = spsolve(A, b)

    u[free_dofs] = x[:nb_free_dofs]

    lambda1 = rescaling * x[nb_free_dofs:nb_free_dofs+nb_constraint_dofs].reshape([-1, 2])

    positions_new = positions + u.reshape([-1, 2])

    return positions_new, lambda1
