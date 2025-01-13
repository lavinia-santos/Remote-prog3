
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

