import numpy as np
import math
import gradients
import energies
import reading
import bond_angles
import optimization_cartesian as opt_cartesian
import optimization_internal as opt_internal



def debug(file_name, write_output=False):

    """
    This function is used for debugging the whole process.
    It checks if the file is correctly read, calculates bond lengths and energies, and verifies angle calculations.
    """
    print(f"Starting debug for file: {file_name}")
    #write to output
    if write_output:
        with open(f"{file_name}_debug.log", "w") as output_file:
            output_file.write(f"Starting debug for file: {file_name}\n")

    
    # Test reading file
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name) 
                                                                                    #    print_num_atoms=False, 
                                                                                    #    print_num_bonds=False, 
                                                                                    #    print_num_atom_types=False, 
                                                                                    #    print_atom_coords=False, 
                                                                                    #    print_bonds=False, 
                                                                                    #    dev=False)
    
    # Check atoms and bonds
    # print("\nAtoms and their coordinates:", atom_coords)
    # print("\nAtom types:", atom_types)
    # print("\nBonds:", bonds)
    #write to output
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Atoms and their coordinates: {atom_coords}\n")
            output_file.write(f"Atom types: {atom_types}\n")
            output_file.write(f"Bonds: {bonds}\n")
    
    # Test bond length calculation
    bond_length_dict = bond_angles.bond_length_all(file_name, print_bond_length=False, check_bonds=False, print_dict=False)
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Bond lengths: {bond_length_dict}\n")
    
    # Test angle calculation
    print("\nCalculating angles...")
    angles = bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types)

    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Angles successfully calculated: {angles}\n")
            for angle in angles:
                output_file.write(f"Angle between atoms {angle[0]}-{angle[1]}-{angle[2]}: {angle[3]:.2f} degrees\n")

    #calculate torsional angles
    torsional_angles, chains = bond_angles.calculate_torsion_angle(atom_coords, bonds, atom_types, read_coordinates_from_file=True)
    print(torsional_angles.values())
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            #print the torsional angles with the respective atons of the chain of four
            output_file.write(f"Torsional atoms before eliminating duplicates:\n")
            print("\nTorsional atoms before eliminating duplicates:")
            # for torsional_angle in torsional_angles.values():
                # print(torsional_angle)
            for key, value in torsional_angles.items():
                output_file.write(f"{key}: {value}\n")
                print(f"{key}: {value}")


    
    # if angles:
    #     print("\nAngles successfully calculated:")
    #     for angle in angles:
    #         print(f"Angle between atoms {angle[0]}-{angle[1]}-{angle[2]}: {angle[3]:.2f} degrees")
    # else:
    #     print("\nNo angles calculated. Please check bonds and file structure.")
    
    # Print bond stretching energies
    print("\nCalculating bond stretching energies...")
    bond_stretching_energies = energies.calculate_bond_stretching_energy(file_name,atom_types, print_energies=False)
    # sum_energy = calculate_bond_stretching_energy(file_name,atom_types, print_energies=True)[1]
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Bond stretching energies: {bond_stretching_energies}\n")
    
    # Show results with atom types
    print("\nFinal results for angles with atom types:")
    bond_angles.print_angles_with_atom_types(file_name)

    # Print angle bending energies
    print("\nCalculating angle bending energies...")
    angle_bending_energies = energies.calculate_angle_bending_energy(file_name, atom_types, print_energies=False)
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Angle bending energies: {angle_bending_energies}\n")
    
    # Print torsion energies
    print("\nCalculating torsion energies...")
    torsion_energies = energies.calculate_torsion_energy(file_name, atom_types, print_energies=False)
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Torsion energies: {torsion_energies}\n")


    # Print VDW energies
    print("\nCalculating VDW energies...")
    energies.calculate_VDW_energy(file_name, reading.read_input(file_name, dev=True)[5], print_energies=False)
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"VDW energies: {energies.calculate_VDW_energy(file_name, reading.read_input(file_name, dev=True)[5], print_energies=False)}\n")

    # print("\nDebugging completed.")

    # Calculate total energy
    print("\nCalculating total energy...")
    energies.total_energy(file_name, atom_types, print_energies=True)
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Total energy: {energies.total_energy(file_name, atom_types, print_energies=False)}\n")

    # Calculate bond stretching gradient
    print("\nCalculating bond stretching gradient...")
    bond_stretching_gradient = gradients.calculate_bond_stretching_gradient(file_name, atom_types)
    print("Bond stretching gradient:")
    print(bond_stretching_gradient)
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Bond stretching gradient: {bond_stretching_gradient}\n")

    # Calculate angle bending gradient
    print("\nCalculating angle bending gradient...")
    angle_bending_gradient = gradients.calculate_angle_bending_gradient(file_name, atom_types)
    print("Angle bending gradient:")
    print(angle_bending_gradient)
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Angle bending gradient: {angle_bending_gradient}\n")

    # Calculate dihedral angle gradient
    print("\nCalculating dihedral angle gradient...")
    dihedral_angle_gradient = gradients.calculate_dihedral_angle_gradient(file_name,atom_coords, bonds, atom_types)
    print("Dihedral angle gradient:")
    print(dihedral_angle_gradient)
    if write_output:
        with open(f"{file_name}_debug.log", "a") as output_file:
            output_file.write(f"Dihedral angle gradient: {dihedral_angle_gradient[0]}\n")

    # Calculate VDW gradient
    print("\nCalculating VDW gradient...")
    vdw_gradient = gradients.calculate_vdw_gradient(file_name, atom_types)

    # Calculate full gradient
    print("\nCalculating full gradient...")
    full_gradient = gradients.gradient_full(file_name, atom_types, atom_coords, bonds, num_atoms)
    print("Full gradient:")
    print(full_gradient)

    #add other parts to this function
    #opt, etc
    

def regular_run (file_name, output_file, more_info=False):
    """
    This function is used for running the whole process.
    """

    opt_cartesian.optimize_bfgs_cartesian (file_name, output_file, more_info=more_info)
    #also add the internal opt
    opt_internal.optimize_bfgs_internal (file_name, output_file, more_info=more_info)