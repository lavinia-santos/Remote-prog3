import numpy as np
import math
import gradients
import energies
import reading
import bond_angles


def calculate_B_and_G_matrices(file_name):
    
    
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)

    _, grad_r0_cartesian = gradients.calculate_bond_stretching_gradient(file_name, atom_types, read_coordinates_from_file=True, coordinates=None)
    grad_angle0_cartesian = gradients.calculate_angle_bending_gradient(file_name, atom_types, read_coordinates_from_file=True, coordinates=None)
    grad_dihedral0_cartesian = gradients.calculate_dihedral_angle_gradient(file_name,atom_coords, bonds,atom_types)

    # print(grad_r0_cartesian)

    grad0_cartesian = gradients.gradient_full(file_name, atom_types, atom_coords, bonds, num_atoms)

    nx = 3*num_atoms

    # print("bonds: ",bonds)

    #get number of bond lengths
    num_bonds = len(bonds)
    # print("num bonds: ",num_bonds)

    #get number of angles
    num_angles = len(bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types))
    # print("num angles: ",num_angles)

    #get number of dihedrals
    torsion_angles, chains = bond_angles.calculate_torsion_angle(atom_coords, bonds, atom_types)
    num_dihedrals = len(torsion_angles)/2
    num_dihedrals = int(num_dihedrals)
    # print("num_dihedrals: ",num_dihedrals)

    nq = num_bonds + num_angles + num_dihedrals
    nq = int(nq)
    # print(nq)

    B = np.zeros((nq,nx))
    
    count = -1
    old_bond = ""

    for element in grad_r0_cartesian.keys():
        print("element: ",element)
        print("grad_r0_cartesian[element]: ",grad_r0_cartesian[element])
        atom_number = int(element[10])
        print("atom_number: ",atom_number)
        bond = element[2:7]
        print("bond: ",bond)
        if bond != old_bond:
            count += 1        
        B[count][(3 * atom_number) - 3] = grad_r0_cartesian[element][0]
        B[count][(3 * atom_number) - 2] = grad_r0_cartesian[element][1]
        B[count][(3 * atom_number) - 1] = grad_r0_cartesian[element][2]

        old_bond = bond

        

    # for atom in grad_r0_cartesian.keys():
        # print("atom: ",atom)
        # atom_number = int(atom[1:])
        # # print("atom_number: ",atom_number)
        # # print(grad_r0_cartesian[atom])
        
        # B[0][atom_number - 1] = grad_r0_cartesian[atom][0]
        # B[0][atom_number] = grad_r0_cartesian[atom][1]
        # B[0][atom_number + 1] = grad_r0_cartesian[atom][2]

            
    print(B)





calculate_B_and_G_matrices("ethane")


