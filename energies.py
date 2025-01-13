import numpy as np
import math
from reading import read_input
from reading import read_parameters
import bond_angles


def calculate_bond_stretching_energy(file, atom_types, print_energies=False):
    """
    This function calculates the bond stretching energy for all bonds in the molecule.
    The energy is computed using the formula: E_bond = kb * (r - r0)^2
    """
    # Read parameters
    parameters = read_parameters()
    kb = parameters['kb']
    r0 = parameters['r0']
    # print(kb)
    energies=[]
    # Get molecule information
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)

    # Calculate bond lengths
    bond_lengths = bond_angles.bond_length_all(file)
    print("Bond lengths:", bond_lengths)

    # Calculate bond stretching energy for each bond
    bond_stretching_energies = {}
    for bond in bonds:
        atom1, atom2 = str(bond[0]), str(bond[1])
        # {atom_types[str(atom1)]}{atom1}
        bond_key_number = f"{str(atom1)}-{str(atom2)}"
        reverse_key_number = f"{str(atom2)}-{str(atom1)}"
        bond_key = f"{atom_types[str(atom1)]}{atom_types[str(atom2)]}"
        reverse_key = f"{atom_types[str(atom2)]}{atom_types[str(atom1)]}"

        # Get bond length
        r = bond_lengths.get(bond_key_number) or bond_lengths.get(reverse_key_number)
        # print(r)

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

        # Calculate energy
        energy = k_bond * (r - r_eq) ** 2
        bond_stretching_energies[bond_key] = energy
        print(energy)
        energies.append(energy)
        if print_energies:
            print(f"kbond= {k_bond} Bond {bond_key}: Length = {r:.3f}, Energy = {energy:.3f} kcal/mol")
            
    #sum the energies
    sum_energy = sum(energies)
    print(f"Sum of bond stretching energies: {sum_energy:.6f} kcal/mol")

    return bond_stretching_energies, sum_energy


def calculate_angle_bending_energy(file, atom_types, print_energies=False):
    """
    Esta função calcula a energia de flexão para todos os ângulos na molécula.
    A energia é calculada usando a fórmula: E_ângulo = ka * (θ - θ0)^2
    """
    # Ler parâmetros
    parameters = read_parameters()
    ka = parameters['ka']
    theta0 = parameters['theta0']
    #tranform theta0 to radians
    for key in theta0:
        theta0[key] = math.radians(float(theta0[key]))
    energies=[]
    # print(ka)
    # Obter informações da molécula
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = read_input(file, dev=True)
    
    # Calcular os ângulos
    angles = bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types)
    
    # Calcular a energia de flexão para cada ângulo
    angle_bending_energies = {}
    for angle in angles:
        atom1, atom2, atom3, angle_value = angle
        
        #get only the atom symbol
        atom1_letter = atom1[0]
        atom2_letter = atom2[0]
        atom3_letter = atom3[0]

        # Extrair as possiveis chaves para o ângulo
        
        angle_key_1 = f"{atom1_letter}{atom2_letter}{atom3_letter}"
        angle_key_2 = f"{atom3_letter}{atom2_letter}{atom1_letter}"
        angle_key_3 = f"{atom3_letter}{atom1_letter}{atom2_letter}"
        angle_key_4 = f"{atom1_letter}{atom3_letter}{atom2_letter}"
        angle_key_5 = f"{atom2_letter}{atom1_letter}{atom3_letter}"
        angle_key_6 = f"{atom2_letter}{atom3_letter}{atom1_letter}"
        
        
        
        # Obter o valor de k_a e θ_0 para o ângulo
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
        
        # Calcular a energia
        print("angle_value:", angle_value)
        print("theta_eq:", theta_eq)
        energy = k_angle * (angle_value - theta_eq) ** 2
        angle_bending_energies[angle_key_1] = energy
        energies.append(energy)
        sum_energy = sum(energies)
        if print_energies:
            print(f"k_angle= {k_angle} Angle {angle_key_1}: Angle = {angle_value:.3f}, Energy = {energy:.3f} kcal/mol")
    print(f"Sum of angle bending energies: {sum_energy:.6f} kcal/mol")
    return angle_bending_energies, sum_energy



def calculate_torsion_energy(file, atom_types, print_energies=False):
    # Ler parâmetros
    parameters = read_parameters()
    Aphi = parameters['Aphi']
    Aphi = float(Aphi)
    print(Aphi)
    n = 3
    energies=[]
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    
    # Calcular ângulos de torção e garantir que sejam únicos
    torsion_angles = calculate_torsion_angle(atom_coords, bonds, atom_types)
    print("Torsion angles:", torsion_angles)
    # Remover duplicatas
    unique_torsion_angles = list(set([angle for angle in torsion_angles[0].values()]))
    
    print("Unique torsion angles:", unique_torsion_angles)
    
    torsion_energy = {}
    for angle in unique_torsion_angles:
        # print("Aphi:", Aphi)
        # print("n:", n)
        # print("angle:", angle)
        # print("math.cos(n * angle):", math.cos(n * math.radians(angle)))
        energy = Aphi * (1 + math.cos(n * math.radians(angle)))
        torsion_energy[angle] = energy
        energies.append(energy)
        if print_energies:
            print(f"Angle {angle:.3f}, Energy = {energy:.6f} kcal/mol")
    
    sum_energy = sum(energies)
    print(f"Sum of torsion energies: {sum_energy:.6f} kcal/mol")
    
    return torsion_energy, sum_energy


def calculate_VDW_energy(file, atom_types, print_energies=False, debug=False):
    sigma_i = read_parameters()['sigma_i']
    epsilon_i = read_parameters()['epsilon_i']
    energies=[]
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, _ = read_input(file, dev=True)
    # bond_lengths = bond_length_all(file)
    #get sigma H,C and epsilon H,C
    sigma_H = float(sigma_i['H'])
    epsilon_H = float(epsilon_i['H'])
    sigma_C = float(sigma_i['C'])
    epsilon_C = float(epsilon_i['C'])
    if debug:
        print(f"sigma_H= {sigma_H} epsilon_H= {epsilon_H}")
        print(f"sigma_C= {sigma_C} epsilon_C= {epsilon_C}")
    #calculate sigmaHC, sigmaHH, sigmaCC, epsilonHC, epsilonHH, epsilonCC
    epsilon_HC = math.sqrt(epsilon_H * epsilon_C)
    epsilon_HH = math.sqrt(epsilon_H * epsilon_H)
    epsilon_CC = math.sqrt(epsilon_C * epsilon_C)
    sigma_HC = 2*math.sqrt(sigma_H * sigma_C)
    sigma_HH = 2*math.sqrt(sigma_H * sigma_H)
    sigma_CC = 2*math.sqrt(sigma_C * sigma_C)
    if debug:
        print(f"sigma_HC= {sigma_HC} epsilon_HC= {epsilon_HC}")
        print(f"sigma_HH= {sigma_HH} epsilon_HH= {epsilon_HH}")
        print(f"sigma_CC= {sigma_CC} epsilon_CC= {epsilon_CC}")
    #calculate VDW energy for each unique pair of atoms
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
            if atom != atom2:
                pair = tuple(sorted([atom, atom2]))
                if pair not in unique_pairs:
                    if atom2 not in bonded_atoms.get(atom, []) and not any(atom2 in bonded_atoms.get(neighbor, []) for neighbor in bonded_atoms.get(atom, [])):
                        # print(bonded_atoms.get(atom))
                        # print(pair)
                        unique_pairs.add(pair)
                        atom1 = str(atom)
                        atom2 = str(atom2)
                        bond_key = f"{atom_types[atom1]}{atom_types[atom2]}"
                        r = np.linalg.norm(np.array(atom_coords[atom1]) - np.array(atom_coords[atom2]))
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
                        energy = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
                        energies.append(energy)
                        if print_energies:
                            print(f"epsilon= {epsilon} kcal/mol, sigma= {sigma} Angstroms, VDW Bond {atom_types[atom1]}{atom1}-{atom_types[atom2]}{atom2}: Length = {r:.3f} Angstroms, Energy = {energy:.3f} kcal/mol")

    sum_energy = sum(energies)
    print(f"Sum of VDW energies: {sum_energy:.6f} kcal/mol")
    return sum_energy

def total_energy(file, atom_types, print_energies=False):
    """
    This function calculates the total energy of the molecule by summing all energy contributions.
    """
    # Calculate bond stretching energies
    bond_stretching_energies = calculate_bond_stretching_energy(file, atom_types)
    sum_bond_stretching_energy = bond_stretching_energies[1]
    
    # Calculate angle bending energies
    angle_bending_energies = calculate_angle_bending_energy(file, atom_types)
    sum_angle_bending_energy = angle_bending_energies[1]
    
    # Calculate torsion energies
    torsion_energies = calculate_torsion_energy(file, atom_types)
    sum_torsion_energy = torsion_energies[1]
    
    # Calculate VDW energies
    VDW_energy = calculate_VDW_energy(file, atom_types)
    
    # Total energy
    total_energy = sum_bond_stretching_energy + sum_angle_bending_energy + sum_torsion_energy + VDW_energy
    if print_energies:
        print(f"Total energy: {total_energy:.6f} kcal/mol")

