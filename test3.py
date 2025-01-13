import numpy as np
import math

def read_input(file_name, print_num_atoms=False, print_num_bonds=False, print_num_atom_types=False, print_atom_coords=False, print_bonds=False, dev=True):
    """
    Reads the input file and returns:
    number of atoms, number of bonds, number of atom types,
    coordinates of atoms (dict), and bonds (list of lists)
    """
    if dev:
        file = 'inputs/' + file_name + '.mol2'
        
    
    with open(file, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].split()[0])
    num_bonds = int(lines[0].split()[1])
    num_atom_types = int(lines[0].split()[2])
    
    if print_num_atoms:
        print("Number of atoms: ", num_atoms)
    if print_num_bonds:
        print("Number of bonds: ", num_bonds)
    if print_num_atom_types:
        print("Number of atom types: ", num_atom_types)
    
    # Read atom coordinates and atom types
    atom_coords = {}
    atom_types = {}
    for i in range(1, num_atoms + 1):
        line = lines[i].split()
        atom_type = line[3]
        atom_coords[str(i)] = [float(x) for x in line[0:3]]
        atom_types[str(i)] = atom_type
    
    if print_atom_coords:
        print("Coordinates of atoms: ", atom_coords)
    
    # Read bonds and store them in an array
    bonds = []
    for i in range(num_atoms + 1, num_atoms + num_bonds + 1):
        bonds.append([int(x) for x in lines[i].split()[0:3]])
    
    if print_bonds:
        print("Bonds: ", bonds)
    
    return num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types


def bond_length_all(file, print_bond_length=False, check_bonds=False, print_dict=False):
    """
    This function calculates the bond length between all atoms in the molecule
    """
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = read_input(file, dev=True)
    bond_length = {}
    for bond in bonds:
        if check_bonds:
            print("Atom1: ", bond[0], "Atom2: ", bond[1], "Bond type: ", bond[2])
        atom1 = str(bond[0])
        atom2 = str(bond[1])
        bond_length[atom1 + '-' + atom2] = np.linalg.norm(np.array(atom_coords[atom1]) - np.array(atom_coords[atom2]))
        if print_bond_length:
            print("The bond length between ", atom1, " and ", atom2, " is: ", bond_length[atom1 + '-' + atom2], " Angstroms")
    if print_dict:
        print(bond_length)
    return bond_length


def calculate_angle(coord1, coord2, coord3):
    """
    Calculates the angle formed by three atoms (in degrees).
    coord1, coord2, coord3 are lists or tuples with [x, y, z] coordinates.
    """
    # Vectors
    vector1 = [coord1[i] - coord2[i] for i in range(3)]
    vector2 = [coord3[i] - coord2[i] for i in range(3)]
    
    # Dot product and magnitudes
    dot_product = sum(vector1[i] * vector2[i] for i in range(3))
    magnitude1 = math.sqrt(sum(vector1[i]**2 for i in range(3)))
    magnitude2 = math.sqrt(sum(vector2[i]**2 for i in range(3)))
    
    # Cosine and angle
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_rad = math.acos(cos_theta)  # In radians
    #convert to degrees
    angle_deg = math.degrees(angle_rad)
    return angle_rad

def calculate_angles_from_bonds(atom_coords, bonds, atom_types):
    """
    Calculate all angles from the bonds provided, using atom coordinates.
    The angles are formed between triplets of atoms: (atom1 - atom2 - atom3)
    """
    angles = []
    
    # Create a dictionary to store which atoms are bonded to each atom
    bonded_atoms = {}
    for bond in bonds:
        atom1, atom2, _ = bond[0], bond[1], bond[2]
        
        if atom1 not in bonded_atoms:
            bonded_atoms[atom1] = []
        if atom2 not in bonded_atoms:
            bonded_atoms[atom2] = []
        
        bonded_atoms[atom1].append(atom2)
        bonded_atoms[atom2].append(atom1)
    
    # For each atom, get all pairs of connected atoms to form angles
    for atom in bonded_atoms:
        connected_atoms = bonded_atoms[atom]
        
        # Check all combinations of two atoms bonded to the same central atom
        for i in range(len(connected_atoms)):
            for j in range(i+1, len(connected_atoms)):
                atom1 = connected_atoms[i]
                atom2 = connected_atoms[j]
                
                # Calculate the angle between atom1 - atom - atom2
                angle = calculate_angle(atom_coords[str(atom1)], atom_coords[str(atom)], atom_coords[str(atom2)])
                
                # Create atom labels like H7-C2-H8
                label1 = f"{atom_types[str(atom1)]}{atom1}"
                label2 = f"{atom_types[str(atom)]}{atom}"
                label3 = f"{atom_types[str(atom2)]}{atom2}"
                
                angles.append((label1, label2, label3, angle))
    
    return angles

def print_angles_with_atom_types(file_name):
    """
    Function to print angles with atom labels (type + atom number)
    """
    atom_coords = read_input(file_name, dev=True)[3]
    bonds = read_input(file_name, dev=True)[4]
    atom_types = read_input(file_name, dev=True)[5]
    
    angles = calculate_angles_from_bonds(atom_coords, bonds, atom_types)

    for angle in angles:
        print(f"Angle between atoms {angle[0]}-{angle[1]}-{angle[2]}: {angle[3]:.2f} degrees")


def read_parameters():
    """
    This function reads the parameters file and returns the parameters in a dictionary.
    """
    parameters = 'tiny.parameters'
    with open(parameters, 'r') as f:
        lines = f.readlines()
    
    sig_i = {}
    eps_i = {}
    kb = {}
    r0 = {}
    ka = {}
    theta0 = {}
    
    for line_or in lines:
        line = line_or.split()
        if line[0] == 'VDW':
            # Get the values of the next 2 lines
            for i in range(2):
                line = lines[lines.index(line_or) + 1 + i].split()
                sig_i[line[0]] = line[1]
                eps_i[line[0]] = line[2]
        elif line[0] == 'Bond':
            for i in range(2):
                line = lines[lines.index(line_or) + 1 + i].split()
                kb[line[0] + line[1]] = line[2]
                r0[line[0] + line[1]] = line[3]
        elif line[0] == 'Bending':
            for i in range(3):
                line = lines[lines.index(line_or) + 1 + i].split()
                ka[line[0] + line[1] + line[2]] = line[3]
                theta0[line[0] + line[1] + line[2]] = line[4]
        elif line[0] == 'Torsion':
            line = lines[lines.index(line_or) + 1].split()
            Aphi = line[0]

    parameters = {'sigma_i': sig_i, 'epsilon_i': eps_i, 'kb': kb, 'r0': r0, 'ka': ka, 'theta0': theta0, 'Aphi': Aphi}
    return parameters


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
    bond_lengths = bond_length_all(file)
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
    angles = calculate_angles_from_bonds(atom_coords, bonds, atom_types)
    
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


def calculate_torsion_angle(atom_coords, bonds):
    """
    Calculates the torsion angle formed by chains of four atoms (in degrees) based on atomic coordinates and bond information.
    """

    torsion_angles = []

    # Criar mapa de átomos conectados
    bonded_atoms = {}
    for bond in bonds:
        atom1, atom2, _ = bond[0], bond[1], bond[2]
        
        if atom1 not in bonded_atoms:
            bonded_atoms[atom1] = []
        if atom2 not in bonded_atoms:
            bonded_atoms[atom2] = []
        
        bonded_atoms[atom1].append(atom2)
        bonded_atoms[atom2].append(atom1)

    print("Conexões de átomos:", bonded_atoms)

    # Encontrar todas as cadeias válidas de 4 átomos
    chains_of_four = []
    for atom1 in bonded_atoms:
        for atom2 in bonded_atoms[atom1]:
            for atom3 in bonded_atoms[atom2]:
                if atom3 == atom1:  # Evitar voltar ao átomo inicial
                    continue
                for atom4 in bonded_atoms[atom3]:
                    if atom4 == atom2 or atom4 == atom1:  # Evitar loops ou repetições
                        continue
                    chains_of_four.append((atom1, atom2, atom3, atom4))

    print("Cadeias de quatro átomos encontradas:", chains_of_four)

    # Calcular os ângulos de torção para cada cadeia válida
    for chain in chains_of_four:
        atom1, atom2, atom3, atom4 = chain
        print("Calculando ângulo de torção para a cadeia:", atom1, atom2, atom3, atom4)
        # Coordenadas dos átomos
        # print("atom_coords[1]:", atom_coords['1'])
        coord1 = atom_coords[str(atom1)]
        coord2 = atom_coords[str(atom2)]
        coord3 = atom_coords[str(atom3)]
        coord4 = atom_coords[str(atom4)]
        
        print("Coordenadas dos átomos:", coord1, coord2, coord3, coord4)
        # Vetores
        vector1 = np.array([coord2[i] - coord1[i] for i in range(3)])
        vector2 = np.array([coord3[i] - coord2[i] for i in range(3)])
        vector3 = np.array([coord4[i] - coord3[i] for i in range(3)])

        # Produtos vetoriais
        cross1 = np.cross(vector1, vector2)
        cross2 = np.cross(vector2, vector3)

        # Normalizar produtos vetoriais
        norm1 = np.linalg.norm(cross1)
        norm2 = np.linalg.norm(cross2)

        if norm1 == 0 or norm2 == 0:
            continue  # Evitar divisão por zero

        cross1 /= norm1
        cross2 /= norm2

        # Ângulo entre os planos
        dot_product = np.dot(cross1, cross2)
        dot_product = max(-1.0, min(1.0, dot_product))  # Prevenir erros numéricos fora do intervalo [-1, 1]
        angle = np.arccos(dot_product)

        # Determinar o sinal do ângulo usando o produto misto
        sign = np.dot(cross1, vector3)
        if sign < 0:
            angle = -angle

        # Converter para graus
        angle_deg = math.degrees(angle)
        torsion_angles.append(angle_deg)

    if not torsion_angles:
        print("Nenhuma cadeia de quatro átomos conectados foi encontrada para calcular ângulos de torção.")
    else:
        print("Ângulos de torção calculados:", torsion_angles)

    return torsion_angles, chains_of_four

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
    torsion_angles = calculate_torsion_angle(atom_coords, bonds)
    print("Torsion angles:", torsion_angles)
    # Remover duplicatas
    unique_torsion_angles = list(set([angle for angle in torsion_angles[0]]))
    
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

# total_energy('ethane', read_input('ethane', dev=True)[5], print_energies=True)

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
    bond_lengths = bond_length_all(file)
    
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
    angles = calculate_angles_from_bonds(atom_coords, bonds, atom_types)
    
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
        


        
        
        # # print("force:", force)
         
        # # Calculate the gradient for atom 2
        # vector1 = np.array(coord1) - np.array(coord2)
        # vector2 = np.array(coord3) - np.array(coord2)
        # norm_vector1 = np.linalg.norm(vector1)
        # norm_vector2 = np.linalg.norm(vector2)
        # print("vector1:", vector1)
        # print("vector2:", vector2)
        # # print("norm_vector1:", norm_vector1)
        # # print("norm_vector2:", norm_vector2)
        # cos_theta = np.dot(vector1, vector2) / (norm_vector1 * norm_vector2)
        # print("cos_theta:", cos_theta)
        # if cos_theta <= 1 and cos_theta >= -1:
        #     sin_theta = np.sqrt(1 - cos_theta ** 2)
        #     # print("\ncos_theta:", cos_theta)
        #     # print("\nsin_theta:", sin_theta)
        # else:
        #     cos_theta = np.dot(vector2, vector1) / (norm_vector1 * norm_vector2)
        #     sin_theta = np.sqrt(1 - cos_theta ** 2)
        # # if sin_theta == 0:
        # #     continue
        # gradient2 = (force / (norm_vector1 * sin_theta)) * (vector2 / norm_vector2 - cos_theta * vector1 / norm_vector1)
        # gradients[f"{atom_types[atom2_number]}{atom2_number}"] += gradient2
        # # print(f"Atom: {atom2_letter}{atom2_number}, Force: {force}, Gradient: {gradient2}")

        # # Calculate the gradient for atom 1
        # gradient1 = -force / (norm_vector1 * sin_theta) * (vector2 / norm_vector2 - cos_theta * vector1 / norm_vector1)
        # gradients[f"{atom_types[atom1_number]}{atom1_number}"] += gradient1
        
        # # Calculate the gradient for atom 3
        # gradient3 = -force / (norm_vector2 * sin_theta) * (vector1 / norm_vector1 - cos_theta * vector2 / norm_vector2)
        # gradients[f"{atom_types[atom3_number]}{atom3_number}"] += gradient3


        # # Calculate the gradient for atom 1
        # # gradient1 = (-force / sin_theta) * (1/(norm_vector1 * norm_vector2)) * (vector2 - cos_theta * vector1 / norm_vector1)
        # # gradients[f"{atom_types[atom1_number]}{atom1_number}"] += gradient1

        # # gradient3 = (-force / sin_theta) * (1/(norm_vector1 * norm_vector2)) * (vector1 - cos_theta * vector2 / norm_vector2)
        # # gradients[f"{atom_types[atom3_number]}{atom3_number}"] += gradient3

        # # gradient2 = -gradient1 - gradient3
        # gradients[f"{atom_types[atom2_number]}{atom2_number}"] += gradient2
        




        print(f"Atom: {atom2_letter}{atom2_number}, Gradient: {gradient2}")
        
        # Calculate the gradient for atom 3
        
    return gradients


# calculate_bond_stretching_energy('methane', read_input('methane', dev=True)[5], print_energies=True)
# print(calculate_bond_stretching_gradient('ethane', read_input('ethane', dev=True)[5]))

def debug(file_name):
    """
    This function is used for debugging the whole process.
    It checks if the file is correctly read, calculates bond lengths and energies, and verifies angle calculations.
    """
    print(f"Starting debug for file: {file_name}")
    
    # Test reading file
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = read_input(file_name, 
                                                                                       print_num_atoms=True, 
                                                                                       print_num_bonds=True, 
                                                                                       print_num_atom_types=True, 
                                                                                       print_atom_coords=True, 
                                                                                       print_bonds=True, 
                                                                                       dev=True)
    
    # Check atoms and bonds
    print("\nAtoms and their coordinates:", atom_coords)
    print("\nAtom types:", atom_types)
    print("\nBonds:", bonds)
    
    # Test bond length calculation
    bond_length_dict = bond_length_all(file_name, print_bond_length=True, check_bonds=True, print_dict=True)
    
    # Test angle calculation
    print("\nCalculating angles...")
    angles = calculate_angles_from_bonds(atom_coords, bonds, atom_types)
    
    if angles:
        print("\nAngles successfully calculated:")
        for angle in angles:
            print(f"Angle between atoms {angle[0]}-{angle[1]}-{angle[2]}: {angle[3]:.2f} degrees")
    else:
        print("\nNo angles calculated. Please check bonds and file structure.")
    
    # Print bond stretching energies
    print("\nCalculating bond stretching energies...")
    bond_stretching_energies = calculate_bond_stretching_energy(file_name,atom_types, print_energies=True)
    # sum_energy = calculate_bond_stretching_energy(file_name,atom_types, print_energies=True)[1]
    
    # Show results with atom types
    print("\nFinal results for angles with atom types:")
    print_angles_with_atom_types(file_name)

    # Print angle bending energies
    print("\nCalculating angle bending energies...")
    angle_bending_energies = calculate_angle_bending_energy(file_name, atom_types, print_energies=True)
    
    # Print torsion energies
    print("\nCalculating torsion energies...")
    torsion_energies = calculate_torsion_energy(file_name, atom_types, print_energies=True)


    # Print VDW energies
    print("\nCalculating VDW energies...")
    calculate_VDW_energy(file_name, read_input('ethane', dev=True)[5], print_energies=True)

    print("\nDebugging completed.")

    # Calculate total energy
    print("\nCalculating total energy...")
    total_energy(file_name, atom_types, print_energies=True)

    # Calculate bond stretching gradient
    print("\nCalculating bond stretching gradient...")
    bond_stretching_gradient = calculate_bond_stretching_gradient(file_name, atom_types)
    print("Bond stretching gradient:")
    print(bond_stretching_gradient)

    # Calculate angle bending gradient
    print("\nCalculating angle bending gradient...")
    angle_bending_gradient = calculate_angle_bending_gradient(file_name, atom_types)
    print("Angle bending gradient:")
    print(angle_bending_gradient)

# Example usage of debug function
# debug('ethane')  # Replace 'ethane' with the name of your file
if __name__ == "__main__":
    debug('ethane')  # Replace 'ethane' with the name of your file
    # print(calculate_bond_stretching_gradient('ethane', read_input('ethane', dev=True)[5]))
    # calculate_angle_bending_energy('ethane', read_input('ethane', dev=True)[5], print_energies=True)
    # print(calculate_angle_bending_gradient('methane', read_input('methane', dev=True)[5]))
