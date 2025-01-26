import numpy as np
import math
import gradients
import energies
import reading
import bond_angles
import internal_coord
import copy


def optimize_bfgs_cartesian (file_name, output_file, write_output = True, more_info=False):
    """
    This function is used for optimizing the geometry of a molecule using the BFGS algorithm.
    It reads the input file, calculates the gradient, and updates the coordinates of the atoms.
    """
    # Read the input file
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)

    

    #define output file
    # output_file = "optimization_" + file_name + ".out"

    if more_info:
        #write on the output file
        with open(output_file, "a") as f:
            f.write(f"############################Printing input data############################\n")
            f.write(f"Molecule name: {file_name}\n")
            f.write(f"Initial coordinates: {atom_coords}\n")
    
    # Get the initial gradient
    grad0 = gradients.gradient_full(file_name, atom_types, atom_coords, bonds, num_atoms)
    #put grad_r0 values into a matrix
    grad_0_values = np.array(list(grad0.values()))


    #calculates initial energy
    E0 = energies.total_energy(file_name, atom_types)
    # print("E0:",E0)
    if more_info:
        #append to the output file
        with open(output_file, "a") as f:
            f.write(f"Initial total gradient: {grad0}\n")
            f.write(f"Initial energy: {E0}\n")
            f.write(f"\n############################Starting cartesian optimization############################\n\n")


    grad_0_values_flat = grad_0_values.flatten()




    # Set the initial inverse Hessian approximation to the identity matrix
    M = np.identity(3 * num_atoms)
    M0 = M * (1/300)  # Set the initial inverse Hessian approximation to a small value, B^-1

    threshold = 0.001 

    step_max = 0.02

    for k in range(1, 500):
        
        if k == 1:

            pk1 = -np.dot(M0, grad_0_values_flat) #creates the search direction to update the coordinates due to bond stretching gradient
            pk1_flat = pk1.flatten()


            alpha = 0.8
            

            # Update the coordinates of the atoms
            atom_coords_new = {}
            sk1 = alpha * pk1_flat

            #checking if the full step is too big
            if np.linalg.norm(sk1) > step_max:
                sk1 = sk1 * (step_max / np.linalg.norm(sk1))


            #giving the step
            for i in range(1, num_atoms + 1):   
                step_k = alpha * pk1[i - 1] #sk = alphak * pk
                step_k_norm = np.linalg.norm(step_k)
                #checking if the step per atom is too big, this is the only reason
                #step_k is evaluate individually
                if step_k_norm > step_max:
                    step_k = step_k * (step_max / step_k_norm)
                #step is actually given to the coordinates from the flatten sk1 vector
                atom_coords_new[str(i)] = atom_coords[str(i)] + sk1[3 * (i-1):3 * (i)]

            E_k1 = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
            # print("E_k:",E_k1)

            #line search
            #check wolfe condition
            c1 = 0.1

            count = 1
            
            while E_k1 > E0 + (c1 * alpha * np.dot(pk1_flat,grad_0_values_flat)):
                if more_info:
                    #write on the output file
                    with open(output_file, "a") as f:
                        f.write(f"\nChecking Wolfe condition iteration {count} inside step {k}\n")
                        # f.write(f"Iteration: {count}\n")
                        f.write(f"Wolfe condition not satisfied\n\n")
                alpha = alpha * 0.8
                if more_info:
                    with open(output_file, "a") as f:
                        f.write(f"new alpha: {alpha}\n")
                        f.write(f"Ek1: {E_k1}\n")

                sk1 = alpha * pk1_flat

            
                for i in range(1, num_atoms + 1):
                    step_k = alpha * pk1[i - 1] #sk = alphak * pk
                    step_k_norm = np.linalg.norm(step_k)
                    if step_k_norm > step_max:
                        step_k = step_k * (step_max / step_k_norm)
                    atom_coords_new[str(i)] = atom_coords[str(i)] + sk1[3 * (i-1):3 * (i)]

                E_k1 = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)

            else:
                if more_info:
                    #write on the output file
                    with open(output_file, "a") as f:
                        f.write(f"\nChecking Wolfe condition iteration {count} inside step {k}\n")
                        f.write("Wolfe condition satisfied\n")
                        f.write(f"alpha:{alpha}\n")
                        f.write(f"E0:{E0}\n")
                        f.write(f"E_k:{E_k1}\n") #actual final energy for rk+1
                        f.write(f"Current coordinates:{atom_coords_new}\n")

                grad_1 = gradients.gradient_full(file_name, atom_types, atom_coords_new, bonds, num_atoms, read_coordinates_from_file=False, coordinates=atom_coords_new)
                grad_1_values = np.array(list(grad_1.values()))

                yk = grad_1_values - grad_0_values
                yk_flat = yk.flatten()
                vk = np.dot(M0,yk_flat)
                sk_dot_yk = np.dot(sk1,yk_flat)
                yk_dot_vk = np.dot(yk_flat,vk)
                sk_x_sk = np.outer(sk1,sk1)
                vk_x_sk = np.outer(vk,sk1)
                sk_x_vk = np.outer(sk1,vk)

                


                Mk1 = M0 + ((np.dot((sk_dot_yk + yk_dot_vk),sk_x_sk))/(sk_dot_yk**2)) - ((vk_x_sk + sk_x_vk)/(sk_dot_yk))


                delta_E = E_k1 - E0
                rms = np.sqrt(np.dot(grad_1_values.flatten(),grad_1_values.flatten())/len(grad_1_values.flatten()))
                print("old energy:",E0)
                print("new energy:",E_k1)

                print("delta_E:",delta_E,"k:",k)
                print("rms:",rms)
                print("k:",k)
                if rms <= threshold:
                    print("delta_E:",delta_E)
                    print("rms:",rms)
                    print("Convergence reached on step:",k)
                    if write_output:
                        with open(output_file, "a") as f:
                            f.write(f"Convergence reached on step: {k}\n")
                            f.write(f"Optimized geometry for {file_name}:\n")
                            f.write(f"Energy: {E_k1:.6f} kcal/mol\n")
                            f.write(f"GRMS: {rms}\n")
                            f.write(f"Coordinates:\n")
                            #write the atom type and the coordinates
                            for i in range(1, num_atoms + 1):
                                f.write(f"{atom_types[str(i)]} {atom_coords_new[str(i)][0]:.6f} {atom_coords_new[str(i)][1]:.6f} {atom_coords_new[str(i)][2]:.6f}\n")
                                
                    break

                count += 1
            

        else:
            

            grad_1_values_flat = grad_1_values.flatten()
            pk = -np.dot(Mk1, grad_1_values_flat)


            pk_flat = pk.flatten()


            alpha = 0.8
            
            
            sk = alpha * pk_flat
            # print("sk:",sk)
            if np.linalg.norm(sk) > step_max:
                sk = sk * (step_max / np.linalg.norm(sk))
            # print("atom_coords_old:",atom_coords_new)
            atom_coords = atom_coords_new
            atom_coords_new = {}

            for i in range(1, num_atoms + 1):   
                step_k = alpha * pk[i - 1] #sk = alphak * pk
                step_k_norm = np.linalg.norm(step_k)
                if step_k_norm > step_max:
                    step_k = step_k * (step_max / step_k_norm)

                atom_coords_new[str(i)] = atom_coords[str(i)] + sk[3 * (i-1):3 * (i)]

            
            
            E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
            #check wolfe condition
            c1 = 0.1

            i_count = 1

            while E_k > E_k1 + (c1 * alpha * np.dot(pk_flat,grad_1_values_flat)):
                if more_info:
                    with open(output_file, "a") as f:
                        f.write(f"\nChecking Wolfe condition iteration {i_count} inside step {k}\n")
                        f.write("Wolfe condition not satisfied\n")
                alpha = alpha * 0.8
                if more_info:
                    with open(output_file, "a") as f:
                        f.write(f"new alpha:{alpha}\n")
                        f.write(f"E_k previous:{E_k}\n")


                sk = alpha * pk_flat

                for i in range(1, num_atoms + 1):
                        step_k = alpha * pk[i - 1] #sk = alphak * pk
                        step_k_norm = np.linalg.norm(step_k)
                        if step_k_norm > step_max:
                            step_k = step_k * (step_max / step_k_norm)
                        atom_coords_new[str(i)] = atom_coords[str(i)] + sk[3 * (i-1):3 * (i)]
                E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)

                i_count += 1

            else:
                if more_info:
                    with open(output_file, "a") as f:
                        f.write(f"\nChecking Wolfe condition iteration {i_count} inside step {k}\n")
                        f.write(f"Wolfe condition satisfied\n")
                        f.write(f"alpha: {alpha}\n")
                        f.write(f"E_k-1: {E_k1}\n")
                        f.write(f"E_k:{E_k}\n") #actual final energy for rk+1
                        f.write(f"Current coordinates: {atom_coords_new}\n")
                        f.write(f"End of iteration {k}\n")
                grad_k_new = gradients.gradient_full(file_name, atom_types, atom_coords_new, bonds, num_atoms, read_coordinates_from_file=False, coordinates=atom_coords_new)
                grad_k_new_values = np.array(list(grad_k_new.values()))
                # print("grad_k_values:",grad_k_new_values)
                yk = grad_k_new_values - grad_1_values
                yk_flat = yk.flatten()
                vk = np.dot(Mk1,yk_flat)
                sk_dot_yk = np.dot(sk,yk_flat)
                yk_dot_vk = np.dot(yk_flat,vk)
                sk_x_sk = np.outer(sk,sk)
                vk_x_sk = np.outer(vk,sk)
                sk_x_vk = np.outer(sk,vk)

                delta_E = E_k - E_k1
                rms = np.sqrt(np.dot(grad_k_new_values.flatten(),grad_k_new_values.flatten())/len(grad_k_new_values.flatten()))
                print("old energy:",E_k1)
                print("new energy:",E_k)

                print("delta_E:",delta_E,"k:",k)
                print("rms:",rms)
                print("k:",k)
                if rms <= threshold:
                    print("Convergence reached on k:",k)
                    print("final energy:",E_k)
                    print("final coordinates:",atom_coords_new)
                    #write on output file
                    if write_output:
                        with open(output_file, "a") as f:
                            f.write("\n\n###################### Optimization completed ######################\n\n")
                            f.write(f"\nOptimized geometry for {file_name}:\n")
                            f.write(f"Convergence reached on step: {k}\n")
                            f.write(f"Energy: {E_k:.8f} kcal/mol\n")
                            f.write(f"GRMS: {rms}\n")
                            f.write(f"Coordinates:\n")
                            #write the atom type and the coordinates
                            for i in range(1, num_atoms + 1):
                                f.write(f"{atom_types[str(i)]} {atom_coords_new[str(i)][0]:.6f} {atom_coords_new[str(i)][1]:.6f} {atom_coords_new[str(i)][2]:.6f}\n")
                                
                    break


                Mk_new = Mk1 + ((np.dot((sk_dot_yk + yk_dot_vk),sk_x_sk))/(sk_dot_yk**2)) - ((vk_x_sk + sk_x_vk)/(sk_dot_yk))

                grad_0_values = grad_1_values
                grad_1_values = grad_k_new_values
                E_k1 = E_k
                Mk1 = Mk_new



