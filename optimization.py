import numpy as np
import math
import gradients
import energies
import reading
import bond_angles


def optimize_bfgs (file_name):
    """
    This function is used for optimizing the geometry of a molecule using the BFGS algorithm.
    It reads the input file, calculates the gradient, and updates the coordinates of the atoms.
    """
    # Read the input file
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)
    
    # Get the initial gradient
    grad0 = gradients.gradient_full(file_name, atom_types, atom_coords, bonds, num_atoms)
    grad_r0 = gradients.calculate_bond_stretching_gradient(file_name, atom_types)
    #put grad_r0 values into a matrix
    grad_r0_values = np.array(list(grad_r0.values()))
    grad_0_values = np.array(list(grad0.values()))

    print("atom_coords: ", atom_coords) 

    #calculates initial energy
    E0 = energies.total_energy(file_name, atom_types)
    print("E0:",E0)

    # print("grad_r0_values",grad_r0_values)
    #flateen grad_r0_values
    grad_r0_values_flat = grad_r0_values.flatten()
    grad_0_values_flat = grad_0_values.flatten()


    # r_0 = bond_angles.bond_length_all(file_name) #just to know the initial bond lengths
    # print(r_0)

    # Set the initial inverse Hessian approximation to the identity matrix
    M = np.identity(3 * num_atoms)
    M0 = M * (1/300)  # Set the initial inverse Hessian approximation to a small value, B^-1

    for k in range(1, 10):
        
        if k == 1:
            # Bk = M0 #B0^-1
            # Mk1 = M0 #just changing notation
            pk1 = -np.dot(M0, grad_r0_values_flat) #creates the search direction to update the coordinates due to bond stretching gradient
            # print("pk:",pk1)
            pk1_flat = pk1.flatten()
            # print(M)
            # print(-np.dot(M, grad_r0_values))

            alpha = 0.8
            

            # Update the coordinates of the atoms
            atom_coords_new = {}
            sk1 = alpha * pk1_flat
            # print("sk1:",sk1)
            if np.linalg.norm(sk1) > 0.3:
                sk1 = sk1 * (0.3 / np.linalg.norm(sk1))

            # for i in range(1, num_atoms + 1):
            #     atom_coords_new[str(i)] = atom_coords[str(i)] + sk1[3 * (i-1):3 * (i)]
            
            # wk = np.dot(M0,sk) #w_k = Bk * sk
            
            # print ("wk:",wk)


            for i in range(1, num_atoms + 1):   
                step_k = alpha * pk1[i - 1] #sk = alphak * pk
                step_k_norm = np.linalg.norm(step_k)
                if step_k_norm > 0.3:
                    step_k = step_k * (0.3 / step_k_norm)
                atom_coords_new[str(i)] = atom_coords[str(i)] + step_k # r_k+1 = r_k + sk
            

            print("atom_coords_new:",atom_coords_new)

            # w_k = M0 * step_k #w_k = Bk * sk

            # Calculate the new energy
            E_k1 = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
            # print("E_k:",E_k1)


            #check wolfe condition
            c1 = 0.1

            while E_k1 > E0 + (c1 * alpha * np.dot(pk1_flat,grad_r0_values_flat)):
                print("Wolfe condition not satisfied")
                alpha = alpha * 0.8
                # print("new alpha:",alpha)

                sk1 = alpha * pk1_flat
                # if np.linalg.norm(sk1) > 0.3:
                #     sk1 = sk1 * (0.3 / np.linalg.norm(sk1))

                # for i in range(1, num_atoms + 1):
                #     atom_coords_new[str(i)] = atom_coords[str(i)] + sk1[3 * (i-1) : 3 * (i)]
            
                for i in range(1, num_atoms + 1):
                    step_k = alpha * pk1[i - 1] #sk = alphak * pk
                    step_k_norm = np.linalg.norm(step_k)
                    if step_k_norm > 0.3:
                        step_k = step_k * (0.3 / step_k_norm)
                    atom_coords_new[str(i)] = atom_coords[str(i)] + step_k # r_k+1 = r_k + sk
                E_k1 = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
            else:
                print("Wolfe condition satisfied")
                print("alpha:",alpha)
                print("E0:",E0)
                print("E_k:",E_k1) #actual final energy for rk+1
                grad_rk1 = gradients.calculate_bond_stretching_gradient(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
                grad_rk1_values = np.array(list(grad_rk1.values()))
                print("grad_r0_values:",grad_r0_values)
                print("grad_rk_values:",grad_rk1_values)
                yk = grad_rk1_values - grad_r0_values
                yk_flat = yk.flatten()
                vk = np.dot(M0,yk_flat)
                sk_dot_yk = np.dot(sk1,yk_flat)
                yk_dot_vk = np.dot(yk_flat,vk)
                sk_x_sk = np.outer(sk1,sk1)
                vk_x_sk = np.outer(vk,sk1)
                sk_x_vk = np.outer(sk1,vk)


                Mk1 = M0 + ((np.dot((sk_dot_yk + yk_dot_vk),sk_x_sk))/(sk_dot_yk**2)) - ((vk_x_sk + sk_x_vk)/(sk_dot_yk))
                print("Mk:",Mk1)
                # print("Bk:",Bk)
                
                # print("yk:",yk)
                # print("wk:",wk)

        else:
            
            print("k is not 1, i is:",k)
            grad_rk_values_flat = grad_rk1_values.flatten()
            pk = -np.dot(Mk1, grad_rk_values_flat)
            print("pk:",pk)

            pk_flat = pk.flatten()

            alpha = 0.8

            
            sk = alpha * pk_flat
            print("sk:",sk)
            if np.linalg.norm(sk) > 0.3:
                sk = sk * (0.3 / np.linalg.norm(sk))
            print("atom_coords_old:",atom_coords_new)
            atom_coords_new = {}
            # for i in range(1, num_atoms + 1):
            #     atom_coords_new[str(i)] = atom_coords[str(i)] + sk[3 * (i-1):3 * (i)]
            for i in range(1, num_atoms + 1):   
                step_k = alpha * pk[i - 1] #sk = alphak * pk
                step_k_norm = np.linalg.norm(step_k)
                if step_k_norm > 0.3:
                    step_k = step_k * (0.3 / step_k_norm)
                print("step_k:",step_k)
                atom_coords_new[str(i)] = atom_coords[str(i)] + step_k # r_k+1 = r_k + sk
            
            print("atom_coords_new:",atom_coords_new)
            
            E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
            print("E_k:",E_k)
            #check wolfe condition
            c1 = 0.1

            i_lim = 100
            i_count = 0

            while E_k > E_k1 + (c1 * alpha * np.dot(pk_flat,grad_rk_values_flat)):
                print("Wolfe condition not satisfied")
                

                alpha = alpha * 0.8
                print("new alpha:",alpha)
                print("E_k-1:",E_k1)
                print("E_k:",E_k)
                print("wolf term:",E_k1 + (c1 * alpha * np.dot(pk_flat,grad_rk_values_flat)))

                sk = alpha * pk_flat
                # if np.linalg.norm(sk) > 0.3:
                #     sk = sk * (0.3 / np.linalg.norm(sk))

                # for i in range(1, num_atoms + 1):
                #     atom_coords_new[str(i)] = atom_coords[str(i)] + sk[3 * (i-1) : 3 * (i)]
            
                # E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)   
            
                for i in range(1, num_atoms + 1):
                        step_k = alpha * pk[i - 1] #sk = alphak * pk
                        step_k_norm = np.linalg.norm(step_k)
                        if step_k_norm > 0.3:
                            step_k = step_k * (0.3 / step_k_norm)
                        atom_coords_new[str(i)] = atom_coords[str(i)] + step_k # r_k+1 = r_k + sk
                E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
                # Incrementando o contador e verificando o limite de iterações
                i_count += 1
                print("i_count:",i_count)
                if i_count >= i_lim:
                    print("Iteration limit reached for debugging.")
                    print("k:",k)
                    break
            else:
                print("Wolfe condition satisfied")
                print("alpha:",alpha)
                print("E_k-1:",E_k1)
                print("E_k:",E_k) #actual final energy for rk+1
                print("grad_rk-1_values:",grad_rk1_values)
                grad_rk_new = gradients.calculate_bond_stretching_gradient(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
                grad_rk_new_values = np.array(list(grad_rk_new.values()))
                print("grad_rk_values:",grad_rk_new_values)
                yk = grad_rk_new_values - grad_rk1_values
                yk_flat = yk.flatten()
                vk = np.dot(Mk1,yk_flat)
                sk_dot_yk = np.dot(sk,yk_flat)
                yk_dot_vk = np.dot(yk_flat,vk)
                sk_x_sk = np.outer(sk,sk)
                vk_x_sk = np.outer(vk,sk)
                sk_x_vk = np.outer(sk,vk)


                Mk_new = Mk1 + ((np.dot((sk_dot_yk + yk_dot_vk),sk_x_sk))/(sk_dot_yk**2)) - ((vk_x_sk + sk_x_vk)/(sk_dot_yk))
                print("Mk:",Mk_new)

                grad_rk1_values = grad_rk_new_values
                E_k1 = E_k
                Mk1 = Mk_new
                




            

        



optimize_bfgs("ethane_dist")



    



    