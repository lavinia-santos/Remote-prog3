import numpy as np
import math
import gradients
import energies
import reading
import bond_angles


def optimize_bfgs (file_name):
    """
    This function is used for optimizing the geometry of a molecule using the BFGS algorithm.
    It reads the input file, calculates the gradient, and updates the coordinates of the atoms.
    """
    # Read the input file
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)
    
    # Get the initial gradient
    grad0 = gradients.gradient_full(file_name, atom_types, atom_coords, bonds, num_atoms)
    grad_r0 = gradients.calculate_bond_stretching_gradient(file_name, atom_types)
    #put grad_r0 values into a matrix
    grad_r0_values = np.array(list(grad_r0.values()))

    print("atom_coords: ", atom_coords) 

    #calculates initial energy
    E0 = energies.total_energy(file_name, atom_types)
    print("E0:",E0)

    # print("grad_r0_values",grad_r0_values)
    #flateen grad_r0_values
    grad_r0_values_flat = grad_r0_values.flatten()


    # r_0 = bond_angles.bond_length_all(file_name) #just to know the initial bond lengths
    # print(r_0)

    # Set the initial inverse Hessian approximation to the identity matrix
    M = np.identity(num_atoms)
    M = M * (1/300)  # Set the initial inverse Hessian approximation to a small value

    pk = -np.dot(M, grad_r0_values) #creates the search direction to update the coordinates due to bond stretching gradient
    print("pk:",pk)
    pk_flat = pk.flatten()
    # print(M)
    # print(-np.dot(M, grad_r0_values))

    alpha = 0.8

    # Update the coordinates of the atoms
    atom_coords_new = {}
    
    for i in range(1, num_atoms + 1):
        atom_coords_new[str(i)] = atom_coords[str(i)] + alpha * pk[i - 1]
    
    print("atom_coords_new:",atom_coords_new)

    # Calculate the new energy
    E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
    print("E_k:",E_k)

    #check wolfe condition
    c1 = 0.1

    while E_k >= E0 + (c1 * alpha * np.dot(pk_flat,grad_r0_values_flat)):
        print("Wolfe condition not satisfied")
        alpha = alpha * 0.8
        print("new alpha:",alpha)
        for i in range(1, num_atoms + 1):
            atom_coords_new[str(i)] = atom_coords[str(i)] + alpha * pk[i - 1]
        E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
    else:
        print("Wolfe condition satisfied")
        print("alpha:",alpha)
        print("E0:",E0)
        print("E_k:",E_k)



optimize_bfgs("ethane_dist")



    



    