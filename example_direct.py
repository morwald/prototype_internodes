#!/usr/bin/env python
# coding: utf-8

import akantu as aka
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from .functions_direct import *
from .functions_contact_probl import *

# example
def main():
    mesh_file = 'meshs/contact2_p1_0_05.msh'
    material_file = 'material.dat'

    aka.parseInput(material_file)
    spatial_dimension = 2

    mesh = aka.Mesh(spatial_dimension)
    mesh.read(mesh_file)

    # initialize model
    model = aka.SolidMechanicsModel(mesh)
    model.initFull(_analysis_method=aka._implicit_dynamic)

    # boundary conditions
    model.applyBC(aka.FixedValue(0., aka._x), 'lower_bottom')
    model.applyBC(aka.FixedValue(0., aka._y), 'lower_bottom')

    # Dirichlet boundary conditions
    # model.applyBC(aka.FixedValue(-0.1, aka._y), 'upper_top')

    # Neumann boundary conditions (K is not invertible)
    traction = np.zeros(spatial_dimension)
    traction[1] = -1e9
    model.applyBC(aka.FromTraction(traction), "upper_top")

    # init and solve
    model, data = init_direct(model, mesh, mesh_file, material_file, spatial_dimension,
            "upper_bottom", "lower_top")
    solve_step_direct(model, data, nb_max_iter=10, plot=True)

def init_direct(model, mesh, mesh_file, material_file, spatial_dimension,
        gmsh_interface1, gmsh_interface2):
    # dictonary to store mesh data
    data = {}

    ### initialize mesh data ###
    # data['positions] of nodes
    data['nb_nodes'] = mesh.getNbNodes()
    data['nodes'] = np.arange(data['nb_nodes'])
    data['positions'] = mesh.getNodes()
    data['dofs'] = nodes_to_dofs(data['nodes'])
    data['nb_dofs'] = data['nb_nodes'] * spatial_dimension

    # connectivity
    data['connectivity'] = mesh.getConnectivity(aka._triangle_3)
    data['connectivity_boundary'] = mesh.getConnectivity(aka._segment_2)

    # coordinates, nodes and dofs of body 1
    data['nodes1'] = mesh.getElementGroup("body_upper").getNodeGroup().getNodes().ravel()
    data['positions1'] = data['positions'][data['nodes1']]
    data['dofs1'] = nodes_to_dofs(data['nodes1']).ravel()

    # coordinates, nodes and dofs of body 1
    data['nodes2'] = mesh.getElementGroup("body_lower").getNodeGroup().getNodes().ravel()
    data['positions2'] = data['positions'][data['nodes2']]
    data['dofs2'] = nodes_to_dofs(data['nodes2']).ravel()


    ### initialize boundary ###
    # 1b denotes boundary 1, 2b denotes boundary 2

    # get nodes from selected surfaces
    data['nodes1b'] = mesh.getElementGroup(gmsh_interface1).getNodeGroup().getNodes().ravel()
    data['nodes2b'] = mesh.getElementGroup(gmsh_interface2).getNodeGroup().getNodes().ravel()

    # or get nodes via akantu boundary algorithm
    # _ = mesh.createBoundaryGroupFromGeometry()
    # data['nodes1b'] = mesh.getElementGroup("boundary_0").getNodeGroup().getNodes().ravel()
    # data['nodes2b'] = mesh.getElementGroup("boundary_1").getNodeGroup().getNodes().ravel()

    data['positions1b'] = data['positions'][data['nodes1b']]
    data['dofs1b'] = nodes_to_dofs(data['nodes1b']).ravel()

    data['positions2b'] = data['positions'][data['nodes2b']]
    data['dofs2b'] = nodes_to_dofs(data['nodes2b']).ravel()

    # connectivity of segements on the boundary
    data['connectivity1b'] = data['connectivity_boundary'][np.in1d(data['connectivity_boundary'],
            data['nodes1b']).reshape(data['connectivity_boundary'].shape).any(axis=1)]
    data['connectivity2b'] = data['connectivity_boundary'][np.in1d(data['connectivity_boundary'],
            data['nodes2b']).reshape(data['connectivity_boundary'].shape).any(axis=1)]

    # connectivity of elements belonging to boundary
    data['connectivity1b_body'] = data['connectivity'][np.in1d(data['connectivity'],
            data['nodes1b']).reshape(data['connectivity'].shape).any(axis=1)]
    data['connectivity2b_body'] = data['connectivity'][np.in1d(data['connectivity'],
            data['nodes2b']).reshape(data['connectivity'].shape).any(axis=1)]


    ### boundary conditions ###
    # get free dofs 
    blocked_dofs_mask = model.getBlockedDOFs().ravel()
    data['free_dofs'] = data['dofs'][~blocked_dofs_mask].ravel()
    data['nb_free_dofs'] = len(data['free_dofs'])

    # remove nodes wich are blocked (if akantu boundary algorithm)
    # data['dofs1b'] = data['dofs1b'][np.in1d(data['dofs1b'], data['free_dofs'])]
    # data['dofs2b'] = data['dofs2b'][np.in1d(data['dofs2b'], data['free_dofs'])]

    return model, data

def solve_step_direct(model, data, nb_max_iter=10, plot=False):
    ### assemble stiffness matrices: K, K_free ###
    model.assembleStiffnessMatrix()

    K_aka = model.dof_manager.getMatrix("K")
    K = sp.sparse.csc_matrix(aka.AkantuSparseMatrix(K_aka))

    # K for all non blocked dofs
    rescaling = 30e9 # with E
    K_free = K[np.ix_(data['free_dofs'], data['free_dofs'])] / rescaling

    ### assemble external forces: f, f_free ###
    f = model.getExternalForce().ravel()
 
    # f for all non blocked dofs
    f_free = f[data['free_dofs']]


    ### initialize interface ###
    # 1i denotes interface 1, 2i denotes inteface 2

    # take all boundary initialy as interface nodes
    # possible selection with radius calculation
    nodes1i = data['nodes1b']
    positions1i = data['positions1b']
    nodes2i = data['nodes2b'] 
    positions2i = data['positions2b']

    if plot:
        plt.figure()
        plt.triplot(data['positions'][:, 0], data['positions'][:, 1], data['connectivity'])
        plt.title('mesh')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.title('Initial')
        plt.ylim([0.9, 1.1])

        plt.show()

    # run until converged or nb_of_steps/max is attained
    for i in range(nb_max_iter):

        # select nodes belonging to interface
        nodes1i, nodes2i, positions1i, positions2i, radiuses1, radiuses2 = find_contact_nodes(nodes1i, nodes2i,
                positions1i, positions2i)

        dofs1i = nodes_to_dofs(nodes1i).ravel()
        dofs2i = nodes_to_dofs(nodes2i).ravel()
        nb_constraint_dofs = len(dofs1i)

        # global index of the interface among the boundary dofs
        sorter_free_dofs = np.argsort(data['free_dofs'])
        indx1i = sorter_free_dofs[np.searchsorted(data['free_dofs'], dofs1i, sorter=sorter_free_dofs)]
        indx2i = sorter_free_dofs[np.searchsorted(data['free_dofs'], dofs2i, sorter=sorter_free_dofs)]

        # subassemble matrices
        R12_normal, R21_normal, R12, R21 = assemble_Rijs(positions1i, positions2i, radiuses1, radiuses2)
        M1i, M2i = assemble_interface_masses(model, dofs1i, dofs2i)
        B, B_tilde, C = assemble_Bs(M1i, M2i, R12, R21, indx1i, indx2i, data['nb_free_dofs'], nb_constraint_dofs)
        A = assemble_A_explicit(K_free, B, B_tilde, C)

        b = assemble_b(f_free, R12_normal, positions1i, positions2i,
                data['nb_free_dofs'], nb_constraint_dofs, rescaling)

        # solve
        positions_new, lambda1 = solve_direct(A, b, data['positions'], data['free_dofs'],
                nb_constraint_dofs, data['nb_dofs'], rescaling)

        if plot:
            plt.figure()
            plt.triplot(positions_new[:, 0], positions_new[:, 1], data['connectivity'])
            plt.title('mesh')
            plt.xlabel('x')
            plt.ylabel('y')

            plt.title('Iteration ' + str(i+1))
            plt.ylim([0.9, 1.1])

            plt.show()

        # add or remove nodes
        nodes1i, nodes2i, diff_nb_nodes1i, diff_nb_nodes2i = remove_traction(positions_new,
                data['connectivity1b'], data['connectivity2b'], data['connectivity1b_body'], data['connectivity2b_body'],
                nodes1i, nodes2i, data['nodes1b'], data['nodes2b'], lambda1, R12, R21)

        positions1i = data['positions'][nodes1i, :]
        positions2i = data['positions'][nodes2i, :]

        if np.abs(diff_nb_nodes1i)+np.abs(diff_nb_nodes2i) == 0:
            print()
            print('successfully converged in ', i+1, ' iterations')
            break

        return positions_new


if __name__ == '__main__':
    main()
