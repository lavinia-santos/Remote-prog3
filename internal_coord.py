import numpy as np
import math
import gradients
import energies
import reading
import bond_angles


def calculate_B_and_G_matrices(file_name, read_coordinates_from_file=True, coordinates=None):
    
    
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)

    if not read_coordinates_from_file and coordinates is not None:
        atom_coords = coordinates

    _, grad_r0_cartesian = gradients.calculate_bond_stretching_gradient(file_name, atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=atom_coords)
    _, grad_angle0_cartesian = gradients.calculate_angle_bending_gradient(file_name, atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=atom_coords)
    _, grad_dihedral0_cartesian = gradients.calculate_dihedral_angle_gradient(file_name,atom_coords, bonds,atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=atom_coords)


    grad0_cartesian = gradients.gradient_full(file_name, atom_types, atom_coords, bonds, num_atoms)

    nx = 3*num_atoms


    num_bonds = len(bonds)


    num_angles = len(bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types))


    #get number of dihedrals
    torsion_angles, chains = bond_angles.calculate_torsion_angle(atom_coords, bonds, atom_types)
    num_dihedrals = len(torsion_angles)/2
    num_dihedrals = int(num_dihedrals)

    nq = num_bonds + num_angles + num_dihedrals
    nq = int(nq)

    B = np.zeros((nq,nx))
    
    count_bond = -1
    old_bond = ""

    for element in grad_r0_cartesian.keys():

        parts = element.split("/")

        last_part = parts[-1]
        

        atom_number_str = ''.join(filter(str.isdigit, last_part))
        
        if atom_number_str.isdigit():
            atom_number = int(atom_number_str)
        else:
            continue
        
        bond = parts[0]
        
        if bond != old_bond:
            count_bond += 1
        
        B[count_bond][(3 * atom_number) - 3] = grad_r0_cartesian[element][0]
        B[count_bond][(3 * atom_number) - 2] = grad_r0_cartesian[element][1]
        B[count_bond][(3 * atom_number) - 1] = grad_r0_cartesian[element][2]
        
        
        old_bond = bond


        count_angle = num_bonds - 1
        old_angle = ""

        for element in grad_angle0_cartesian.keys():

            parts = element.split("/")

            last_part = parts[-1]

            atom_number_str = ''.join(filter(str.isdigit, last_part))
            
            if atom_number_str.isdigit():
                atom_number = int(atom_number_str)

            else:

                continue
            

            angle = parts[0]

            

            if angle != old_angle:
                count_angle += 1
            
            B[count_angle][(3 * atom_number) - 3] = grad_angle0_cartesian[element][0]
            B[count_angle][(3 * atom_number) - 2] = grad_angle0_cartesian[element][1]
            B[count_angle][(3 * atom_number) - 1] = grad_angle0_cartesian[element][2]
            

            old_angle = angle


    count_dihedral = num_bonds + num_angles - 1
    old_dihedral = ""

    for element in grad_dihedral0_cartesian.keys():

        parts = element.split("/")

        last_part = parts[-1]

        atom_number_str = ''.join(filter(str.isdigit, last_part))
        
        if atom_number_str.isdigit():
            atom_number = int(atom_number_str)
        else:
            continue
        
        dihedral = parts[0]

        
        if dihedral != old_dihedral:
            count_dihedral += 1
        
        B[count_dihedral][(3 * atom_number) - 3] = grad_dihedral0_cartesian[element][0]
        B[count_dihedral][(3 * atom_number) - 2] = grad_dihedral0_cartesian[element][1]
        B[count_dihedral][(3 * atom_number) - 1] = grad_dihedral0_cartesian[element][2]

        old_dihedral = dihedral



    B_transpose = np.transpose(B)
    G = np.dot(B,B_transpose)
    ############################ checar bem essa parte, entender o que fez, se deu sorted no eigenvalues, ou nao, etc####################   

    eigenvalues, eigenvectors = np.linalg.eig(G)
    eigenvalues_sorted = np.sort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:,eigenvalues.argsort()]

    

    #remove imaginary part of the eigenvalues
    eigenvalues_sorted = np.real(eigenvalues_sorted)
    eigenvalues = np.real(eigenvalues)


    #count number of eigenvalues that are close to zero
    count = 0
    for i in range(len(eigenvalues_sorted)):
        if eigenvalues_sorted[i] < 1e-8:
            count += 1
    #check if this part makes sense
    #if we are using this count for anything


    D = np.diag(eigenvalues)


    for i in range(len(D)):
        if D[i][i] >= 1e-8:
            D[i][i] = 1/D[i][i]


    eigenvectors = np.real(eigenvectors)
    eigenvectors_sorted = np.real(eigenvectors_sorted)

    V = eigenvectors

    V_transpose = np.transpose(V)

    G_inverse = np.dot(np.dot(V,D),V_transpose)



    return B, G_inverse



calculate_B_and_G_matrices("nbutane")

 


def cartesian_to_internal(file_name, read_coordinates_from_file=True, coordinates=None):
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)

    if not read_coordinates_from_file and coordinates is not None:
        atom_coords = coordinates

    bond_lengths, _ = bond_angles.bond_length_all(file_name, read_coordinates_from_file=read_coordinates_from_file, coordinates=atom_coords, print_bond_length=False, check_bonds=False, print_dict=False)


    
    angles = bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types)
    


    
    torsion_angles, chains = bond_angles.calculate_torsion_angle(atom_coords, bonds, atom_types)
    


    #build an array with internal coordinates
    internal_coords = []
    for bond in bond_lengths:
        internal_coords.append(bond_lengths[bond])
    for angle in angles:
        internal_coords.append(angle[3])
    for torsion in torsion_angles.values():
        torsion = np.deg2rad(torsion)

        if torsion not in internal_coords:
            internal_coords.append(torsion)


    internal_coords = np.array(internal_coords)

    return internal_coords






cartesian_to_internal("nbutane")


