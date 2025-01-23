import numpy as np
import math
from reading import read_input
from reading import read_parameters
import bond_angles




def calculate_bond_stretching_gradient(file, atom_types, read_coordinates_from_file=True, coordinates=None):
    """
    This function calculates the gradient of the bond stretching energy with respect to the Cartesian coordinates.
    """
    # Read parameters
    parameters = read_parameters()
    kb = parameters['kb']
    r0 = parameters['r0']
    
    # Get molecule information
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    if read_coordinates_from_file == False:
        atom_coords = coordinates 
    
    # Calculate bond lengths
    bond_lengths = bond_angles.bond_length_all(file, read_coordinates_from_file=read_coordinates_from_file, coordinates=coordinates)
    
    # Initialize gradient dictionary
    gradients = {f"{atom_types[str(i)]}{i}": np.zeros(3) for i in range(1, num_atoms + 1)}
    gradients_bonds = {}
    # Calculate gradient for each bond
    for bond in bonds:
        atom1, atom2 = str(bond[0]), str(bond[1])
        bond_key_number = f"{atom1}-{atom2}"
        reverse_key_number = f"{atom2}-{atom1}"
        bond_key = f"{atom_types[atom1]}{atom_types[atom2]}"
        bond_key_full = f"{atom_types[atom1]}{atom1}-{atom_types[atom2]}{atom2}"
        reverse_key = f"{atom_types[atom2]}{atom_types[atom1]}"
        
        # Get bond length
        r = bond_lengths[0].get(bond_key_number) or bond_lengths[0].get(reverse_key_number)
        
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

        
            

        dr_dx_atom1 = (atom_coords[atom1][0] - atom_coords[atom2][0])/norm_delta_r
        dr_dy_atom1 = (atom_coords[atom1][1] - atom_coords[atom2][1])/norm_delta_r
        dr_dz_atom1 = (atom_coords[atom1][2] - atom_coords[atom2][2])/norm_delta_r

        dr_dx_atom2 = (atom_coords[atom2][0] - atom_coords[atom1][0])/norm_delta_r
        dr_dy_atom2 = (atom_coords[atom2][1] - atom_coords[atom1][1])/norm_delta_r
        dr_dz_atom2 = (atom_coords[atom2][2] - atom_coords[atom1][2])/norm_delta_r

        gradients_bonds[f"dr{bond_key_full}/d{atom_types[atom1]}{atom1}"] = np.array([dr_dx_atom1, dr_dy_atom1, dr_dz_atom1])
        gradients_bonds[f"dr{bond_key_full}/d{atom_types[atom2]}{atom2}"] = np.array([dr_dx_atom2, dr_dy_atom2, dr_dz_atom2])

    
    return gradients, gradients_bonds

def calculate_angle_bending_gradient(file, atom_types, read_coordinates_from_file=True, coordinates=None):
    """
    This function calculates the gradient of the angle bending energy with respect to the Cartesian coordinates.
    """
    # Read parameters
    parameters = read_parameters()
    ka = parameters['ka']
    theta0 = parameters['theta0']
    
    # Get molecule information
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    if read_coordinates_from_file == False:
        atom_coords = coordinates 
    # Calculate angles
    angles = bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types)
    
    # Initialize gradient dictionary
    gradients = {f"{atom_types[str(i)]}{i}": np.zeros(3) for i in range(1, num_atoms + 1)}
    gradients_angles={}
    # Calculate gradient for each angle
    for angle in angles:
        atom1, atom2, atom3, angle_value = angle
        
        # Get atom symbols
        atom1_letter = str(atom1[0])
        atom2_letter = str(atom2[0])
        atom3_letter = str(atom3[0])

        atom1_number = str(atom1[1:])
        atom2_number = str(atom2[1:])
        atom3_number = str(atom3[1:])
        
        # Extract possible keys for the angle
        angle_key_1 = f"{atom1_letter}{atom2_letter}{atom3_letter}"
        angle_key_2 = f"{atom3_letter}{atom2_letter}{atom1_letter}"
        angle_key_3 = f"{atom3_letter}{atom1_letter}{atom2_letter}"
        angle_key_4 = f"{atom1_letter}{atom3_letter}{atom2_letter}"
        angle_key_5 = f"{atom2_letter}{atom1_letter}{atom3_letter}"
        angle_key_6 = f"{atom2_letter}{atom3_letter}{atom1_letter}"

        angle_key_full = f"{atom1_letter}{atom1_number}-{atom2_letter}{atom2_number}-{atom3_letter}{atom3_number}"
        
        # Get the value of k_a and θ_0 for the angle
        if angle_key_1 in ka:
            k_angle = float(ka[angle_key_1])
            theta_eq = float(theta0[angle_key_1])
        elif angle_key_2 in ka:
            k_angle = float(ka[angle_key_2])
            theta_eq = float(theta0[angle_key_2])
        elif angle_key_3 in ka:
            k_angle = float(ka[angle_key_3])
            theta_eq = float(theta0[angle_key_3])
        elif angle_key_4 in ka:
            k_angle = float(ka[angle_key_4])
            theta_eq = float(theta0[angle_key_4])
        elif angle_key_5 in ka:
            k_angle = float(ka[angle_key_5])
            theta_eq = float(theta0[angle_key_5])
        elif angle_key_6 in ka:
            k_angle = float(ka[angle_key_6])
            theta_eq = float(theta0[angle_key_6])
        else:
            print(f"Warning: No parameters found for angle {angle_key_1}. Skipping.")
            continue

        
        coord1 = atom_coords[atom1_number]
        coord2 = atom_coords[atom2_number]
        coord3 = atom_coords[atom3_number]

        theta_eq = np.radians(theta_eq)

        rba = np.array(coord1) - np.array(coord2)
        rbc = np.array(coord3) - np.array(coord2)
        p = np.cross(rba, rbc)
        rab2= rba**2
        rbc2= rbc**2
        rab_norm = np.linalg.norm(rba)
        rab_norm2 = rab_norm**2
        p_norm = np.linalg.norm(p)
        rbc_norm = np.linalg.norm(rbc)
        rbc_norm2 = rbc_norm**2
        force = 2 * k_angle * (angle_value - theta_eq)
        p = np.cross(rba, rbc)
        gradient2 = force*((np.cross(-rba,p)/(rab_norm2 * p_norm))+(np.cross(rbc,p)/(rbc_norm2 * p_norm)))
        gradient1 = force*((np.cross(rba,p))/(rab_norm2 * p_norm))
        gradient3 = force*((np.cross(-rbc,p))/(rbc_norm2 * p_norm))
        gradients[f"{atom_types[atom2_number]}{atom2_number}"] += gradient2
        gradients[f"{atom_types[atom1_number]}{atom1_number}"] += gradient1
        gradients[f"{atom_types[atom3_number]}{atom3_number}"] += gradient3

        dtheta_dx_atom1 = (gradient1[0])/force
        dtheta_dy_atom1 = (gradient1[1])/force
        dtheta_dz_atom1 = (gradient1[2])/force

        dtheta_dx_atom2 = (gradient2[0])/force
        dtheta_dy_atom2 = (gradient2[1])/force
        dtheta_dz_atom2 = (gradient2[2])/force

        dtheta_dx_atom3 = (gradient3[0])/force
        dtheta_dy_atom3 = (gradient3[1])/force
        dtheta_dz_atom3 = (gradient3[2])/force

        gradients_angles[f"d{angle_key_full}/d{atom1}"] = np.array([dtheta_dx_atom1, dtheta_dy_atom1, dtheta_dz_atom1])
        gradients_angles[f"d{angle_key_full}/d{atom2}"] = np.array([dtheta_dx_atom2, dtheta_dy_atom2, dtheta_dz_atom2])
        gradients_angles[f"d{angle_key_full}/d{atom3}"] = np.array([dtheta_dx_atom3, dtheta_dy_atom3, dtheta_dz_atom3])


        
    return gradients, gradients_angles

def calculate_dihedral_angle_gradient(file,atom_coords, bonds,atom_types, read_coordinates_from_file=True, coordinates=None):
    """
    Calculates the gradient of the dihedral angle with respect to the Cartesian coordinates.
    """
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    if read_coordinates_from_file == False:
        atom_coords = coordinates 
    #initialize the gradient dictionary
    gradients = {f"{atom_types[str(i)]}{i}": np.zeros(3) for i in range(1, num_atoms + 1)}
    
    gradients_dihedrals = {}

    parameters = read_parameters()
    Aphi = parameters['Aphi']
    Aphi = float(Aphi)
    n = 3
    
    torsion_angles, chains_of_four = bond_angles.calculate_torsion_angle(atom_coords, bonds, atom_types)
    processed_chains = set()

    # Iterar pelas cadeias de quatro átomos
    for chain in chains_of_four:
        normalized_chain = tuple(sorted(chain, key=lambda x: int(x[1:])))
        if normalized_chain in processed_chains:
            continue
        
        processed_chains.add(normalized_chain)


        atom1, atom2, atom3, atom4 = chain
        dihedral_key_full = f"{atom_types[str(atom1[1:])]}{atom1[1:]}-{atom_types[str(atom2[1:])]}{atom2[1:]}-{atom_types[str(atom3[1:])]}{atom3[1:]}-{atom_types[str(atom4[1:])]}{atom4[1:]}"
        
        # Coordinates of the atoms
        coord1 = atom_coords[str(atom1[1:])]
        coord2 = atom_coords[str(atom2[1:])]
        coord3 = atom_coords[str(atom3[1:])]
        coord4 = atom_coords[str(atom4[1:])]


        rab = np.array(coord2) - np.array(coord1)
        rbc = np.array(coord3) - np.array(coord2)
        rcd = np.array(coord4) - np.array(coord3)
        rac = np.array(coord3) - np.array(coord1)
        rbd = np.array(coord4) - np.array(coord2)


        t = np.cross(rab, rbc)
        u = np.cross(rbc, rcd)
        rbc_norm = np.linalg.norm(rbc)
        t_norm = np.linalg.norm(t)
        u_norm = np.linalg.norm(u)
        t_norm2 = t_norm**2
        u_norm2 = u_norm**2

        phi = torsion_angles[chain]
        phi = math.radians(phi)
        derivative = -n * Aphi * math.sin(n * phi)
        txrbc = np.cross(t,rbc)
        minusuxrbc = np.cross(-u,rbc)
        t2rbc = t_norm2 * rbc_norm
        u2rbc = u_norm2 * rbc_norm


        gradient1 = derivative * np.cross(((np.cross(t,rbc))/(t_norm2 * rbc_norm)),rbc) 
        gradient4 = derivative * np.cross(((np.cross(-u,rbc))/(u_norm2*rbc_norm)),rbc)
        
        gradient2 = derivative * ((np.cross(rac,((np.cross(t,rbc))/(t_norm2 * rbc_norm)))) + (np.cross(((np.cross(-u,rbc))/(u_norm2*rbc_norm)),rcd)))
        gradient3 = derivative * ((np.cross((txrbc/(t_norm2*rbc_norm)),rab)) + (np.cross(rbd,(minusuxrbc/(u_norm2*rbc_norm)))))



        gradients[f"{atom1}"] += gradient1
        gradients[f"{atom2}"] += gradient2
        gradients[f"{atom3}"] += gradient3
        gradients[f"{atom4}"] += gradient4

        dphi_dx_atom1 = gradient1[0]/derivative
        dphi_dy_atom1 = gradient1[1]/derivative
        dphi_dz_atom1 = gradient1[2]/derivative

        dphi_dx_atom2 = gradient2[0]/derivative
        dphi_dy_atom2 = gradient2[1]/derivative
        dphi_dz_atom2 = gradient2[2]/derivative

        dphi_dx_atom3 = gradient3[0]/derivative
        dphi_dy_atom3 = gradient3[1]/derivative
        dphi_dz_atom3 = gradient3[2]/derivative

        dphi_dx_atom4 = gradient4[0]/derivative
        dphi_dy_atom4 = gradient4[1]/derivative
        dphi_dz_atom4 = gradient4[2]/derivative

        gradients_dihedrals[f"d{dihedral_key_full}/d{atom1}"] = np.array([dphi_dx_atom1, dphi_dy_atom1, dphi_dz_atom1])
        gradients_dihedrals[f"d{dihedral_key_full}/d{atom2}"] = np.array([dphi_dx_atom2, dphi_dy_atom2, dphi_dz_atom2])
        gradients_dihedrals[f"d{dihedral_key_full}/d{atom3}"] = np.array([dphi_dx_atom3, dphi_dy_atom3, dphi_dz_atom3])
        gradients_dihedrals[f"d{dihedral_key_full}/d{atom4}"] = np.array([dphi_dx_atom4, dphi_dy_atom4, dphi_dz_atom4])

    return gradients, gradients_dihedrals

        
def calculate_vdw_gradient(file, atom_types, read_coordinates_from_file=True, coordinates=None):
    """
    This function calculates the gradient of the VDW energy with respect to the Cartesian coordinates.
    """
    # Read parameters
    parameters = read_parameters()
    sigma_i = parameters['sigma_i']
    epsilon_i = parameters['epsilon_i']
    
    # Get molecule information
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    if read_coordinates_from_file == False:
        atom_coords = coordinates 
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
    
    for atom in atom_coords:
        for atom2 in atom_coords:
            atom=int(atom)
            atom2=int(atom2)

            if atom != atom2:
                pair = tuple(sorted([atom, atom2]))
                if pair not in unique_pairs:

                    if atom2 not in bonded_atoms.get(atom, []) and not any(atom2 in bonded_atoms.get(neighbor, []) for neighbor in bonded_atoms.get(atom, [])):
                        
                        unique_pairs.add(pair)
                        atom1 = str(atom)
                        atom2 = str(atom2)
                        bond_key = f"{atom_types[atom1]}{atom_types[atom2]}"
                        
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

    return gradients

def gradient_full(file, atom_types, atom_coords, bonds, num_atoms, read_coordinates_from_file=True, coordinates=None):
    """
    This function calculates the full gradient of the molecule by summing all gradient contributions.
    """
    # Calculate bond stretching gradient
    bond_stretching_gradient, _ = calculate_bond_stretching_gradient(file, atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=coordinates)
    
    # Calculate angle bending gradient
    angle_bending_gradient, _ = calculate_angle_bending_gradient(file, atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=coordinates)
    
    # Calculate dihedral angle gradient
    dihedral_angle_gradient, _ = calculate_dihedral_angle_gradient(file, atom_coords, bonds, atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=coordinates)
    
    # Calculate VDW gradient
    vdw_gradient = calculate_vdw_gradient(file, atom_types, read_coordinates_from_file=read_coordinates_from_file, coordinates=coordinates)
    
    # Total gradient
    total_gradient = {f"{atom_types[str(i)]}{i}": np.zeros(3) for i in range(1, num_atoms + 1)}
    for atom in total_gradient:
        total_gradient[atom] = bond_stretching_gradient[atom] + angle_bending_gradient[atom] + dihedral_angle_gradient[atom] + vdw_gradient[atom]
    
    return total_gradient

