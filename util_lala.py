import numpy as np
import math
import gradients
import energies
import reading
import bond_angles



def debug(file_name):

    """
    This function is used for debugging the whole process.
    It checks if the file is correctly read, calculates bond lengths and energies, and verifies angle calculations.
    """
    print(f"Starting debug for file: {file_name}")
    
    # Test reading file
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name, 
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
    bond_length_dict = bond_angles.bond_length_all(file_name, print_bond_length=True, check_bonds=True, print_dict=True)
    
    # Test angle calculation
    print("\nCalculating angles...")
    angles = bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types)
    
    if angles:
        print("\nAngles successfully calculated:")
        for angle in angles:
            print(f"Angle between atoms {angle[0]}-{angle[1]}-{angle[2]}: {angle[3]:.2f} degrees")
    else:
        print("\nNo angles calculated. Please check bonds and file structure.")
    
    # Print bond stretching energies
    print("\nCalculating bond stretching energies...")
    bond_stretching_energies = energies.calculate_bond_stretching_energy(file_name,atom_types, print_energies=True)
    # sum_energy = calculate_bond_stretching_energy(file_name,atom_types, print_energies=True)[1]
    
    # Show results with atom types
    print("\nFinal results for angles with atom types:")
    bond_angles.print_angles_with_atom_types(file_name)

    # Print angle bending energies
    print("\nCalculating angle bending energies...")
    angle_bending_energies = energies.calculate_angle_bending_energy(file_name, atom_types, print_energies=True)
    
    # Print torsion energies
    print("\nCalculating torsion energies...")
    torsion_energies = energies.calculate_torsion_energy(file_name, atom_types, print_energies=True)


    # Print VDW energies
    print("\nCalculating VDW energies...")
    energies.calculate_VDW_energy(file_name, reading.read_input('ethane', dev=True)[5], print_energies=True)

    print("\nDebugging completed.")

    # Calculate total energy
    print("\nCalculating total energy...")
    energies.total_energy(file_name, atom_types, print_energies=True)

    # Calculate bond stretching gradient
    print("\nCalculating bond stretching gradient...")
    bond_stretching_gradient = gradients.calculate_bond_stretching_gradient(file_name, atom_types)
    print("Bond stretching gradient:")
    print(bond_stretching_gradient)

    # Calculate angle bending gradient
    print("\nCalculating angle bending gradient...")
    angle_bending_gradient = gradients.calculate_angle_bending_gradient(file_name, atom_types)
    print("Angle bending gradient:")
    print(angle_bending_gradient)

    # Calculate dihedral angle gradient
    print("\nCalculating dihedral angle gradient...")
    dihedral_angle_gradient = gradients.calculate_dihedral_angle_gradient(file_name,atom_coords, bonds, atom_types)
    # print("Dihedral angle gradient:")

    # Calculate VDW gradient
    print("\nCalculating VDW gradient...")
    vdw_gradient = gradients.calculate_vdw_gradient(file_name, atom_types)

    # Calculate full gradient
    print("\nCalculating full gradient...")
    full_gradient = gradients.gradient_full(file_name, atom_types, atom_coords, bonds, num_atoms)
    print("Full gradient:")
    print(full_gradient)
    

