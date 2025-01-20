import numpy as np
import math
import gradients
import energies
import reading
import bond_angles


def calculate_B_and_G_matrices(file_name, read_coordinates_from_file=True, coordinates=None):
    
    
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)

    if not read_coordinates_from_file:
        atom_coords = coordinates

    _, grad_r0_cartesian = gradients.calculate_bond_stretching_gradient(file_name, atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=coordinates)
    _, grad_angle0_cartesian = gradients.calculate_angle_bending_gradient(file_name, atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=coordinates)
    _, grad_dihedral0_cartesian = gradients.calculate_dihedral_angle_gradient(file_name,atom_coords, bonds,atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=coordinates)

    # print(grad_r0_cartesian)
    # print("grad angles:",grad_angle0_cartesian)

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
    
    count_bond = -1
    old_bond = ""

    for element in grad_r0_cartesian.keys():
        # print("element: ",element)
        length = len(element)
        # print("length: ",length)
        # print("grad_r0_cartesian[element]: ",grad_r0_cartesian[element])
        if length == 11:
            atom_number = int(element[10])
            # print("atom_number: ",atom_number)
            bond = element[2:7]
        elif length == 12:
            atom_number = int(element[11])
            # print("atom_number: ",atom_number)
            bond = element[2:8]
        elif length == 13:
            #concatenates elements 11 and 12
            atom_number = int(element[11:13])
            # print("atom_number: ",atom_number)
            bond = element[2:9]
        # print("bond: ",bond)
        if bond != old_bond:
            count_bond += 1        
        B[count_bond][(3 * atom_number) - 3] = grad_r0_cartesian[element][0]
        B[count_bond][(3 * atom_number) - 2] = grad_r0_cartesian[element][1]
        B[count_bond][(3 * atom_number) - 1] = grad_r0_cartesian[element][2]

        old_bond = bond

    count_angle = num_bonds -1
    old_angle = ""
    for element in grad_angle0_cartesian.keys():
        length = len(element)
        # print("element: ",element)
        # print("length: ",length)
        # print("grad_angle0_cartesian[element]: ",grad_angle0_cartesian[element])
        if length == 13:
            atom_number = int(element[12])
        # print("atom_number: ",atom_number)
            angle = element[1:9]
        # print("angle: ",angle)
        elif length == 14:
            atom_number = int(element[13])
            angle = element[1:10]
        elif length == 15:
            atom_number = int(element[13:15])
            angle = element[1:10]
        if angle != old_angle:
            count_angle += 1
        B[count_angle][(3 * atom_number) - 3] = grad_angle0_cartesian[element][0]
        B[count_angle][(3 * atom_number) - 2] = grad_angle0_cartesian[element][1]
        B[count_angle][(3 * atom_number) - 1] = grad_angle0_cartesian[element][2]
        # print("line B written: ",B[count_angle])

        old_angle = angle

    count_dihedral = num_bonds + num_angles -1
    old_dihedral = ""
    for element in grad_dihedral0_cartesian.keys():
        # print("element: ",element)
        length = len(element)
        # print("length: ",length)
        # print("grad_dihedral0_cartesian[element]: ",grad_dihedral0_cartesian[element])
        # if 
        atom_number = int(element[15])
        # print("atom_number: ",atom_number)
        dihedral = element[1:12]
        # print("dihedral: ",dihedral)
        if dihedral != old_dihedral:
            count_dihedral += 1
        B[count_dihedral][(3 * atom_number) - 3] = grad_dihedral0_cartesian[element][0]
        B[count_dihedral][(3 * atom_number) - 2] = grad_dihedral0_cartesian[element][1]
        B[count_dihedral][(3 * atom_number) - 1] = grad_dihedral0_cartesian[element][2]
        # print("line B written: ",B[count_dihedral])

        old_dihedral = dihedral

    
    #print each line of B separately and the respective line number 
    # for i in range(len(B)):
    #     print("line number: ",i+1)
    #     print(B[i])



    # print(B.shape)
    #get transpose of B
    B_transpose = np.transpose(B)
    #print B_transpose shape
    # print(B_transpose.shape)

    #get G matrix
    G = np.dot(B,B_transpose)
    # print(G)
    #print each line of G separately and the respective line number
    # for i in range(len(G)):
    #     print("line number: ",i+1)
    #     print(G[i])

    #diagonalize G matrix
    eigenvalues, eigenvectors = np.linalg.eig(G)
    eigenvalues_sorted = np.sort(eigenvalues)
    eigenvectors_sorted = eigenvectors[:,eigenvalues.argsort()]

    
    
    #filter out eigenvalues that are close to zero
    # for i in range(len(eigenvalues_sorted)):
    #     if eigenvalues_sorted[i] < 1e-8:
    #         eigenvalues_sorted[i] = 0
    

    #remove zero eigenvalues
    # eigenvalues_sorted = eigenvalues_sorted[eigenvalues_sorted != 0]

    #remove imaginary part of the eigenvalues
    eigenvalues_sorted = np.real(eigenvalues_sorted)
    eigenvalues = np.real(eigenvalues)
    #change order of eigenvalues, from largest to smallest
    # eigenvalues_sorted = eigenvalues_sorted[::-1]
    # print("eigenvalues: ",eigenvalues_sorted)
    # print("eigenvalues: ",eigenvalues)

    #count number of eigenvalues that are close to zero
    count = 0
    for i in range(len(eigenvalues_sorted)):
        if eigenvalues_sorted[i] < 1e-8:
            count += 1

    #remove values that are close to zero
    # eigenvalues_sorted = eigenvalues_sorted[eigenvalues_sorted > 1e-8]

    # #add zeros to the end of the array
    # for i in range(count):
    #     eigenvalues_sorted = np.append(eigenvalues_sorted,0)




    # print (eigenvalues_sorted)

    #put eigenvalues in a diagonal matrix
    D = np.diag(eigenvalues)
    # for i in range(len(D)):
    #     print("line number: ",i+1)
    #     print(D[i])
    
    # print(D)
    #get the inverse of D by replacing the diagonal elements with their reciprocals

    for i in range(len(D)):
        if D[i][i] >= 1e-8:
            D[i][i] = 1/D[i][i]

    #print each line of D separately and the respective line number
    # for i in range(len(D)):
    #     print("line number: ",i+1)
    #     print(D[i])


    # print(D) #lambda-1

    # print("eigenvectors: ",eigenvectors)
    #remove imaginary part of the eigenvectors
    eigenvectors = np.real(eigenvectors)
    eigenvectors_sorted = np.real(eigenvectors_sorted)
    # print("eigenvectors: ",eigenvectors)

    #build a matrix with the eigenvectors as columns
    V = eigenvectors
    # print(V.shape)

    V_transpose = np.transpose(V)
    # print(V_transpose.shape)

    G_inverse = np.dot(np.dot(V,D),V_transpose)

    # print(G_inverse)

    # for i in range(len(G_inverse)):
    #     print("line number: ",i+1)
    #     print(G_inverse[i])

    return B, G_inverse









    G_inv = np.linalg.inv(G)
    # print(G_inv)

    #print each line of G_inv separately and the respective line number
    # for i in range(len(G_inv)):
    #     print("line number: ",i+1)
    #     print(G_inv[i])

    #create dummy matrix


calculate_B_and_G_matrices("ethane")


def cartesian_to_internal(file_name, read_coordinates_from_file=True, coordinates=None):
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)

    if not read_coordinates_from_file:
        atom_coords = coordinates

    bond_lengths, _ = bond_angles.bond_length_all(file_name, read_coordinates_from_file=read_coordinates_from_file, coordinates=coordinates, print_bond_length=False, check_bonds=False, print_dict=False)
    # print("bonds",bond_lengths)

    
    angles = bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types)
    
    # print("angles",angles)

    
    torsion_angles, chains = bond_angles.calculate_torsion_angle(atom_coords, bonds, atom_types)
    
    # print("dihedrals:",torsion_angles.values())

    #build an array with internal coordinates
    internal_coords = []
    for bond in bond_lengths:
        internal_coords.append(bond_lengths[bond])
    for angle in angles:
        internal_coords.append(angle[3])
    for torsion in torsion_angles.values():
        torsion = np.deg2rad(torsion)
        # print(torsion)
        # print("internal_coords until now:",internal_coords)
        if torsion not in internal_coords:
            # print("appending torsion")
            internal_coords.append(torsion)
    # print(internal_coords)

    #print length of internal_coords
    # print(len(internal_coords))

    internal_coords = np.array(internal_coords)

    return internal_coords




    

    


        

            
    





# calculate_B_and_G_matrices("ethane_dist")

cartesian_to_internal("ethane_dist")


