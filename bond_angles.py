
import numpy as np
import math
import reading



def bond_length_all(file,read_coordinates_from_file=True, coordinates=None, print_bond_length=False, check_bonds=False, print_dict=False):
    """
    This function calculates the bond length between all atoms in the molecule
    """

    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file, dev=True)

    if read_coordinates_from_file == False:
        atom_coords = coordinates    

        
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
    # print (bond_length, atom_coords)
    return bond_length, atom_coords


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

def calculate_angles_from_bonds(atom_coords, bonds, atom_types, read_coordinates_from_file=True):
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
    atom_coords = reading.read_input(file_name, dev=True)[3]
    bonds = reading.read_input(file_name, dev=True)[4]
    atom_types = reading.read_input(file_name, dev=True)[5]
    
    angles = calculate_angles_from_bonds(atom_coords, bonds, atom_types)

    for angle in angles:
        print(f"Angle between atoms {angle[0]}-{angle[1]}-{angle[2]}: {angle[3]:.2f} degrees")



def calculate_torsion_angle(atom_coords, bonds, atom_types, read_coordinates_from_file=True):
    """
    Calculates the torsion angle formed by chains of four atoms (in degrees) based on atomic coordinates and bond information.
    """

    torsion_angles = {}

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

    # print("Conexões de átomos:", bonded_atoms)

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
                    label1 = f"{atom_types[str(atom1)]}{atom1}"
                    label2 = f"{atom_types[str(atom2)]}{atom2}"
                    label3 = f"{atom_types[str(atom3)]}{atom3}"
                    label4 = f"{atom_types[str(atom4)]}{atom4}"
                    chains_of_four.append((label1, label2, label3, label4))

    # print("Cadeias de quatro átomos encontradas:", chains_of_four)

    # Calcular os ângulos de torção para cada cadeia válida
    for chain in chains_of_four:
        atom1, atom2, atom3, atom4 = chain
        if read_coordinates_from_file == False:
            atom_coords = atom_coords
        # print("Calculando ângulo de torção para a cadeia:", atom1, atom2, atom3, atom4)

        # print("atom_coords:", atom_coords)
        coord1 = atom_coords[str(atom1[1:])]
        coord2 = atom_coords[str(atom2[1:])]
        coord3 = atom_coords[str(atom3[1:])]
        coord4 = atom_coords[str(atom4[1:])]

        
        
        # print("Coordenadas dos átomos:", coord1, coord2, coord3, coord4)
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
        torsion_angles[chain] = angle_deg

    if not torsion_angles:
        print("Nenhuma cadeia de quatro átomos conectados foi encontrada para calcular ângulos de torção.")
    # else:
    #     print("Ângulos de torção calculados:", torsion_angles)

    return torsion_angles, chains_of_four

