import numpy as np
import math
from reading import read_input
from reading import read_parameters
import bond_angles




def calculate_bond_stretching_gradient(file, atom_types):
    """
    This function calculates the gradient of the bond stretching energy with respect to the Cartesian coordinates.
    """
    # Read parameters
    parameters = read_parameters()
    kb = parameters['kb']
    r0 = parameters['r0']
    
    # Get molecule information
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    
    # Calculate bond lengths
    bond_lengths = bond_angles.bond_length_all(file)
    
    # Initialize gradient dictionary
    gradients = {f"{atom_types[str(i)]}{i}": np.zeros(3) for i in range(1, num_atoms + 1)}
    
    # Calculate gradient for each bond
    for bond in bonds:
        atom1, atom2 = str(bond[0]), str(bond[1])
        print("atom1 bond:", atom1)
        bond_key_number = f"{atom1}-{atom2}"
        reverse_key_number = f"{atom2}-{atom1}"
        bond_key = f"{atom_types[atom1]}{atom_types[atom2]}"
        reverse_key = f"{atom_types[atom2]}{atom_types[atom1]}"
        
        # Get bond length
        r = bond_lengths.get(bond_key_number) or bond_lengths.get(reverse_key_number)
        
        # Get force parameters for the bond
        if bond_key in kb:
            k_bond = float(kb[bond_key])
            r_eq = float(r0[bond_key])
        elif reverse_key in kb:
            k_bond = float(kb[reverse_key])
            r_eq = float(r0[reverse_key])
        else:
            print(f"Warning: No parameters found for bond {bond_key}. Skipping.")
            continue
        
        # Calculate the gradient
        delta_r = np.array(atom_coords[atom1]) - np.array(atom_coords[atom2])
        norm_delta_r = np.linalg.norm(delta_r)
        if norm_delta_r == 0:
            continue  # Avoid division by zero
        
        force_magnitude = 2 * k_bond * (norm_delta_r - r_eq) / norm_delta_r
        force_vector = force_magnitude * delta_r
        
        gradients[f"{atom_types[atom1]}{atom1}"] += force_vector
        gradients[f"{atom_types[atom2]}{atom2}"] -= force_vector
    
    return gradients

def calculate_angle_bending_gradient(file, atom_types):
    """
    This function calculates the gradient of the angle bending energy with respect to the Cartesian coordinates.
    """
    # Read parameters
    parameters = read_parameters()
    ka = parameters['ka']
    theta0 = parameters['theta0']
    
    # Get molecule information
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    
    # Calculate angles
    angles = bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types)
    
    # Initialize gradient dictionary
    gradients = {f"{atom_types[str(i)]}{i}": np.zeros(3) for i in range(1, num_atoms + 1)}
    
    # Calculate gradient for each angle
    for angle in angles:
        atom1, atom2, atom3, angle_value = angle
        
        # Get atom symbols
        atom1_letter = str(atom1[0])
        atom2_letter = str(atom2[0])
        atom3_letter = str(atom3[0])
        # print("atom1 angle letter:", atom1_letter)

        atom1_number = str(atom1[1])
        atom2_number = str(atom2[1])
        atom3_number = str(atom3[1])
        # print("atom1 angle number:", atom1_number)
        
        # Extract possible keys for the angle
        angle_key_1 = f"{atom1_letter}{atom2_letter}{atom3_letter}"
        angle_key_2 = f"{atom3_letter}{atom2_letter}{atom1_letter}"
        angle_key_3 = f"{atom3_letter}{atom1_letter}{atom2_letter}"
        angle_key_4 = f"{atom1_letter}{atom3_letter}{atom2_letter}"
        angle_key_5 = f"{atom2_letter}{atom1_letter}{atom3_letter}"
        angle_key_6 = f"{atom2_letter}{atom3_letter}{atom1_letter}"
        
        # Get the value of k_a and θ_0 for the angle
        if angle_key_1 in ka:
            k_angle = float(ka[angle_key_1])
            theta_eq = float(theta0[angle_key_1])
            # print("angle_key_1:", angle_key_1) 
            # print("k_angle:", k_angle)
            # print("theta_eq:", theta_eq) 
        elif angle_key_2 in ka:
            k_angle = float(ka[angle_key_2])
            theta_eq = float(theta0[angle_key_2])
            # print("angle_key_2:", angle_key_2)
            # print("k_angle:", k_angle)
            # print("theta_eq:", theta_eq)
        elif angle_key_3 in ka:
            k_angle = float(ka[angle_key_3])
            theta_eq = float(theta0[angle_key_3])
            # print("angle_key_3:", angle_key_3)
            # print("k_angle:", k_angle)
            # print("theta_eq:", theta_eq)
        elif angle_key_4 in ka:
            k_angle = float(ka[angle_key_4])
            theta_eq = float(theta0[angle_key_4])
            # print("angle_key_4:", angle_key_4)
            # print("k_angle:", k_angle)
            # print("theta_eq:", theta_eq)
        elif angle_key_5 in ka:
            k_angle = float(ka[angle_key_5])
            theta_eq = float(theta0[angle_key_5])
            # print("angle_key_5:", angle_key_5)
            # print("k_angle:", k_angle)
            # print("theta_eq:", theta_eq)
        elif angle_key_6 in ka:
            k_angle = float(ka[angle_key_6])
            theta_eq = float(theta0[angle_key_6])
            # print("angle_key_6:", angle_key_6)
            # print("k_angle:", k_angle)
            # print("theta_eq:", theta_eq)
        else:
            print(f"Warning: No parameters found for angle {angle_key_1}. Skipping.")
            continue
        #calculate the gradient
        # Calculate the gradient
        
        coord1 = atom_coords[atom1_number]
        coord2 = atom_coords[atom2_number]
        coord3 = atom_coords[atom3_number]
        print("coord1:",atom1_letter,atom1_number, coord1)
        print("coord2:",atom2_letter,atom2_number, coord2)
        print("coord3:", atom3_letter,atom3_number,coord3)
        # if coord1 == 0 or coord2 == 0 or coord3 == 0:
        #     print("Warning: One of the atoms has no coordinates. Stopping.")
        #     break
        # angle_value = np.radians(angle_value)
        theta_eq = np.radians(theta_eq)
        # print("angle_value:", angle_value)
        # print("theta_eq:", theta_eq)
        #builde two vectors for the angle
        rba = np.array(coord1) - np.array(coord2)
        rbc = np.array(coord3) - np.array(coord2)
        # rab = np.array(coord2) - np.array(coord1)
        p = np.cross(rba, rbc)
        rab2= rba**2
        rbc2= rbc**2
        rab_norm = np.linalg.norm(rba)
        rab_norm2 = rab_norm**2
        p_norm = np.linalg.norm(p)
        rbc_norm = np.linalg.norm(rbc)
        rbc_norm2 = rbc_norm**2
        force = 2 * k_angle * (angle_value - theta_eq)
        #external product of the two vectors
        p = np.cross(rba, rbc)
        gradient2 = force*((np.cross(-rba,p)/(rab_norm2 * p_norm))+(np.cross(rbc,p)/(rbc_norm2 * p_norm)))
        gradient1 = force*((np.cross(rba,p))/(rab_norm2 * p_norm))
        gradient3 = force*((np.cross(-rbc,p))/(rbc_norm2 * p_norm))
        gradients[f"{atom_types[atom2_number]}{atom2_number}"] += gradient2
        gradients[f"{atom_types[atom1_number]}{atom1_number}"] += gradient1
        gradients[f"{atom_types[atom3_number]}{atom3_number}"] += gradient3




        print(f"Atom: {atom2_letter}{atom2_number}, Gradient: {gradient2}")
        
        # Calculate the gradient for atom 3
        
    return gradients

def calculate_dihedral_angle_gradient(file,atom_coords, bonds,atom_types):
    """
    Calculates the gradient of the dihedral angle with respect to the Cartesian coordinates.
    """
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    #initialize the gradient dictionary
    gradients = {f"{atom_types[str(i)]}{i}": np.zeros(3) for i in range(1, num_atoms + 1)}
    
    parameters = read_parameters()
    Aphi = parameters['Aphi']
    Aphi = float(Aphi)
    n = 3
    
    torsion_angles, chains_of_four = bond_angles.calculate_torsion_angle(atom_coords, bonds, atom_types)
    # print("Chains of four atoms:", chains_of_four)
    print("Torsion angles:", torsion_angles)
    # unique_torsion_angles = list(set(torsion_angles.values()))
    # print("Unique torsion angles:", unique_torsion_angles)
    # Para armazenar cadeias únicas
    processed_chains = set()

    # Iterar pelas cadeias de quatro átomos
    for chain in chains_of_four:
        # Normalizar a cadeia: garantir uma ordem única
        normalized_chain = tuple(sorted(chain, key=lambda x: (x[1], x[0])))
        
        # Ignorar se já processado
        if normalized_chain in processed_chains:
            continue
        
        # Adicionar ao conjunto de cadeias processadas
        processed_chains.add(normalized_chain)

        atom1, atom2, atom3, atom4 = chain
        print("Calculating dihedral angle gradient for chain:", atom1, atom2, atom3, atom4)
        
        
        # Coordinates of the atoms
        coord1 = atom_coords[str(atom1[1])]
        coord2 = atom_coords[str(atom2[1])]
        coord3 = atom_coords[str(atom3[1])]
        coord4 = atom_coords[str(atom4[1])]
        # print("coord1:",atom1, coord1)
        # print("coord2:",atom2, coord2)
        # print("coord3:", atom3, coord3)
        # print("coord4:", atom4, coord4)

        rab = np.array(coord2) - np.array(coord1)
        rbc = np.array(coord3) - np.array(coord2)
        rcd = np.array(coord4) - np.array(coord3)
        rac = np.array(coord3) - np.array(coord1)
        rbd = np.array(coord4) - np.array(coord2)
        # print("rab:", rab)
        # print("rbc:", rbc)
        # print("rcd:", rcd)

        t = np.cross(rab, rbc)
        u = np.cross(rbc, rcd)
        rbc_norm = np.linalg.norm(rbc)
        t_norm = np.linalg.norm(t)
        u_norm = np.linalg.norm(u)
        t_norm2 = t_norm**2
        u_norm2 = u_norm**2

        phi = torsion_angles[chain]
        phi = math.radians(phi)
        # print("phi:", phi)
        derivative = -n * Aphi * math.sin(n * phi)
        txrbc = np.cross(t,rbc)
        minusuxrbc = np.cross(-u,rbc)
        t2rbc = t_norm2 * rbc_norm
        u2rbc = u_norm2 * rbc_norm

        # Calculate the gradient
        # print("Aphi:", Aphi)
        # print("n:", n)
        # print("phi:", phi)
        # print("np.sin(n * phi):", math.sin(n * phi))
        # print("derivative:", derivative)
        gradient1 = derivative * np.cross(((np.cross(t,rbc))/(t_norm2 * rbc_norm)),rbc) 
        gradient4 = derivative * np.cross(((np.cross(-u,rbc))/(u_norm2*rbc_norm)),rbc)
        # gradient2 = derivative * () + 
        
        gradient2 = derivative * ((np.cross(rac,((np.cross(t,rbc))/(t_norm2 * rbc_norm)))) + (np.cross(((np.cross(-u,rbc))/(u_norm2*rbc_norm)),rcd)))
        gradient3 = derivative * ((np.cross((txrbc/(t_norm2*rbc_norm)),rab)) + (np.cross(rbd,(minusuxrbc/(u_norm2*rbc_norm)))))
        # print(f"Atom: {atom1}, Gradient: {gradient1}")
        # print(f"Atom: {atom4}, Gradient: {gradient4}")
        # print(f"Atom: {atom2}, Gradient: {gradient2}")
        # print(f"Atom: {atom3}, Gradient: {gradient3}")


        # print(f"Atom: {atom1}, Gradient: {gradients[f'{atom1}']}")
        # print(f"Atom: {atom2}, Gradient: {gradients[f'{atom2}']}")
        # print(f"Atom: {atom3}, Gradient: {gradients[f'{atom3}']}")
        # print(f"Atom: {atom4}, Gradient: {gradients[f'{atom4}']}")
        # print("Gradients:", gradients)
        # print(f"Before update: {atom1}, Gradient1: {gradients[f'{atom1}']}")
        # print(f"Before update: {atom2}, Gradient2: {gradients[f'{atom2}']}")
        # print(f"Before update: {atom3}, Gradient3: {gradients[f'{atom3}']}")
        # print(f"Before update: {atom4}, Gradient4: {gradients[f'{atom4}']}")
        gradients[f"{atom1}"] += gradient1
        gradients[f"{atom2}"] += gradient2
        gradients[f"{atom3}"] += gradient3
        gradients[f"{atom4}"] += gradient4
        print("Gradients:", gradients)
        # print(f"After update: {atom1}, Gradient1: {gradients[f'{atom1}']}")
        # print(f"After update: {atom2}, Gradient2: {gradients[f'{atom2}']}")
        # print(f"After update: {atom3}, Gradient3: {gradients[f'{atom3}']}")
        # print(f"After update: {atom4}, Gradient4: {gradients[f'{atom4}']}")
    return gradients

        
def calculate_vdw_gradient(file, atom_types):
    """
    This function calculates the gradient of the VDW energy with respect to the Cartesian coordinates.
    """
    # Read parameters
    parameters = read_parameters()
    sigma_i = parameters['sigma_i']
    epsilon_i = parameters['epsilon_i']
    
    # Get molecule information
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    
    # Calculate bond lengths
    bond_lengths = bond_angles.bond_length_all(file)

    sigma_H = float(sigma_i['H'])
    epsilon_H = float(epsilon_i['H'])
    sigma_C = float(sigma_i['C'])
    epsilon_C = float(epsilon_i['C'])
    epsilon_HC = math.sqrt(epsilon_H * epsilon_C)
    epsilon_HH = math.sqrt(epsilon_H * epsilon_H)
    epsilon_CC = math.sqrt(epsilon_C * epsilon_C)
    sigma_HC = 2*math.sqrt(sigma_H * sigma_C)
    sigma_HH = 2*math.sqrt(sigma_H * sigma_H)
    sigma_CC = 2*math.sqrt(sigma_C * sigma_C)
    
    # Initialize gradient dictionary
    gradients = {f"{atom_types[str(i)]}{i}": np.zeros(3) for i in range(1, num_atoms + 1)}
    
    # Calculate gradient for each unique pair of atoms
    unique_pairs = set()
    bonded_atoms = {}
    for bond in bonds:
        atom1, atom2, _ = bond[0], bond[1], bond[2]
        if atom1 not in bonded_atoms:
            bonded_atoms[atom1] = []
        if atom2 not in bonded_atoms:
            bonded_atoms[atom2] = []
        bonded_atoms[atom1].append(atom2)
        bonded_atoms[atom2].append(atom1)
    # print (bonded_atoms)
    
    for atom in atom_coords:
        for atom2 in atom_coords:
            atom=int(atom)
            atom2=int(atom2)
            # print("atom:", atom)
            # print("atom2:", atom2)
            if atom != atom2:
                pair = tuple(sorted([atom, atom2]))
                # print("pair:", pair)
                if pair not in unique_pairs:
                    # print("pair:", pair)
                    # print("and bonded:",bonded_atoms.get(atom, []))
                    if atom2 not in bonded_atoms.get(atom, []) and not any(atom2 in bonded_atoms.get(neighbor, []) for neighbor in bonded_atoms.get(atom, [])):
                        # print(bonded_atoms.get(atom))
                        
                        unique_pairs.add(pair)
                        atom1 = str(atom)
                        atom2 = str(atom2)
                        bond_key = f"{atom_types[atom1]}{atom_types[atom2]}"
                        # print("bond_key:", bond_key)
                        
                        if bond_key == 'HH':
                            sigma = sigma_HH
                            epsilon = epsilon_HH
                        elif bond_key == 'HC' or bond_key == 'CH':
                            sigma = sigma_HC
                            epsilon = epsilon_HC
                        elif bond_key == 'CC':
                            sigma = sigma_CC
                            epsilon = epsilon_CC
                        else:
                            continue
                    # print(atom_coords)
                        x1 = atom_coords[str(atom1)][0]
                        y1 = atom_coords[str(atom1)][1]
                        z1 = atom_coords[str(atom1)][2]
                        x2 = atom_coords[str(atom2)][0]
                        y2 = atom_coords[str(atom2)][1]
                        z2 = atom_coords[str(atom2)][2]
                        Aij = 4 * epsilon * sigma**12
                        Bij = 4 * epsilon * sigma**6
                        r = np.array(atom_coords[atom1]) - np.array(atom_coords[atom2])
                        r_norm = np.linalg.norm(r)
                        gradientx_atom1 = (x1 - x2) * (((-12 * Aij)/r_norm**14) + ((6 * Bij)/r_norm**8))
                        gradienty_atom1 = (y1 - y2) * (((-12 * Aij)/r_norm**14) + ((6 * Bij)/r_norm**8))
                        gradientz_atom1 = (z1 - z2) * (((-12 * Aij)/r_norm**14) + ((6 * Bij)/r_norm**8))
                        gradientx_atom2 = (x2 - x1) * (((-12 * Aij)/r_norm**14) + ((6 * Bij)/r_norm**8))
                        gradienty_atom2 = (y2 - y1) * (((-12 * Aij)/r_norm**14) + ((6 * Bij)/r_norm**8))
                        gradientz_atom2 = (z2 - z1) * (((-12 * Aij)/r_norm**14) + ((6 * Bij)/r_norm**8))
                        gradients[f"{atom_types[atom1]}{atom1}"] += np.array([gradientx_atom1, gradienty_atom1, gradientz_atom1])
                        gradients[f"{atom_types[atom2]}{atom2}"] += np.array([gradientx_atom2, gradienty_atom2, gradientz_atom2])
                        # print(f"Atom: {atom_types[atom1]}{atom1}, Gradient: {np.array([gradientx_atom1, gradienty_atom1, gradientz_atom1])}")
    print(f"Gradient:", gradients)
    return gradients

def gradient_full(file, atom_types, atom_coords, bonds, num_atoms):
    """
    This function calculates the full gradient of the molecule by summing all gradient contributions.
    """
    # Calculate bond stretching gradient
    bond_stretching_gradient = calculate_bond_stretching_gradient(file, atom_types)
    
    # Calculate angle bending gradient
    angle_bending_gradient = calculate_angle_bending_gradient(file, atom_types)
    
    # Calculate dihedral angle gradient
    dihedral_angle_gradient = calculate_dihedral_angle_gradient(file, atom_coords, bonds, atom_types)
    
    # Calculate VDW gradient
    vdw_gradient = calculate_vdw_gradient(file, atom_types)
    
    # Total gradient
    total_gradient = {f"{atom_types[str(i)]}{i}": np.zeros(3) for i in range(1, num_atoms + 1)}
    for atom in total_gradient:
        total_gradient[atom] = bond_stretching_gradient[atom] + angle_bending_gradient[atom] + dihedral_angle_gradient[atom] + vdw_gradient[atom]
    
    return total_gradient
