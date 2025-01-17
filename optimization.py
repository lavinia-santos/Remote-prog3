import numpy as np
import math
import gradients
import energies
import reading
import bond_angles
import internal_coord


def optimize_bfgs_cartesian (file_name):
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

    threshold = 0.001 # to be confirmed

    step_max = 0.02

    for k in range(1, 250):
        
        if k == 1:
            # Bk = M0 #B0^-1
            # Mk1 = M0 #just changing notation
            pk1 = -np.dot(M0, grad_0_values_flat) #creates the search direction to update the coordinates due to bond stretching gradient
            print("pk1:",pk1)
            pk1_flat = pk1.flatten()
            # print(M)
            # print(-np.dot(M, grad_r0_values))

            alpha = 0.8
            

            # Update the coordinates of the atoms
            atom_coords_new = {}
            sk1 = alpha * pk1_flat
            # print("sk1:",sk1)
            #checking if the full step is too big
            if np.linalg.norm(sk1) > step_max:
                sk1 = sk1 * (step_max / np.linalg.norm(sk1))

            # for i in range(1, num_atoms + 1):
            #     atom_coords_new[str(i)] = atom_coords[str(i)] + sk1[3 * (i-1):3 * (i)]
            
            # wk = np.dot(M0,sk) #w_k = Bk * sk
            
            # print ("wk:",wk)

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
                # atom_coords_new[str(i)] = atom_coords[str(i)] + step_k # r_k+1 = r_k + sk
            

            # print("atom_coords_new:",atom_coords_new)

            # w_k = M0 * step_k #w_k = Bk * sk

            # Calculate the new energy
            E_k1 = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
            # print("E_k:",E_k1)

            #line search
            #check wolfe condition
            c1 = 0.1
            
            while E_k1 > E0 + (c1 * alpha * np.dot(pk1_flat,grad_0_values_flat)):
                print("Wolfe condition not satisfied")
                alpha = alpha * 0.8
                print("new alpha:",alpha)
                print("Ek1:",E_k1)

                sk1 = alpha * pk1_flat
                # if np.linalg.norm(sk1) > 0.3:
                #     sk1 = sk1 * (0.3 / np.linalg.norm(sk1))

                # for i in range(1, num_atoms + 1):
                #     atom_coords_new[str(i)] = atom_coords[str(i)] + sk1[3 * (i-1) : 3 * (i)]
            
                for i in range(1, num_atoms + 1):
                    step_k = alpha * pk1[i - 1] #sk = alphak * pk
                    step_k_norm = np.linalg.norm(step_k)
                    if step_k_norm > step_max:
                        step_k = step_k * (step_max / step_k_norm)
                    atom_coords_new[str(i)] = atom_coords[str(i)] + sk1[3 * (i-1):3 * (i)]

                    # atom_coords_new[str(i)] = atom_coords[str(i)] + step_k # r_k+1 = r_k + sk
                E_k1 = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
                print("E_k1 before else do while:",E_k1)
            else:
                print("Wolfe condition satisfied")
                print("alpha:",alpha)
                print("E0:",E0)
                print("E_k:",E_k1) #actual final energy for rk+1
                print("atom_coords_new:",atom_coords_new)
                # grad_rk1 = gradients.calculate_bond_stretching_gradient(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
                # grad_rk1_values = np.array(list(grad_rk1.values()))
                grad_1 = gradients.gradient_full(file_name, atom_types, atom_coords_new, bonds, num_atoms, read_coordinates_from_file=False, coordinates=atom_coords_new)
                grad_1_values = np.array(list(grad_1.values()))
                # print("grad_r0_values:",grad_r0_values)
                # print("grad_rk_values:",grad_rk1_values)
                print("grad_0_values:",grad_0_values)
                print("grad_1_values:",grad_1_values)
                # yk = grad_rk1_values - grad_r0_values
                yk = grad_1_values - grad_0_values
                yk_flat = yk.flatten()
                vk = np.dot(M0,yk_flat)
                sk_dot_yk = np.dot(sk1,yk_flat)
                # print("sk_dot_yk:",sk_dot_yk)
                yk_dot_vk = np.dot(yk_flat,vk)
                sk_x_sk = np.outer(sk1,sk1)
                vk_x_sk = np.outer(vk,sk1)
                sk_x_vk = np.outer(sk1,vk)

                


                Mk1 = M0 + ((np.dot((sk_dot_yk + yk_dot_vk),sk_x_sk))/(sk_dot_yk**2)) - ((vk_x_sk + sk_x_vk)/(sk_dot_yk))
                print("Mk:",Mk1)
                print("step k finalized:",k)

                delta_E = E_k1 - E0
                if np.abs(delta_E) <= threshold:
                    print("delta_E:",delta_E)
                    print("Convergence reached on k:",k)
                    break
                # print("Bk:",Bk)
                
                # print("yk:",yk)
                # print("wk:",wk)
            

        else:
            
            print("k is not 1, i is:",k)
            # grad_rk_values_flat = grad_rk1_values.flatten()
            grad_1_values_flat = grad_1_values.flatten()
            pk = -np.dot(Mk1, grad_1_values_flat)
            print("pk:",pk)

            pk_flat = pk.flatten()
            # print("pk_flat:",pk_flat)

            alpha = 0.8
            
            
            sk = alpha * pk_flat
            print("sk:",sk)
            if np.linalg.norm(sk) > step_max:
                sk = sk * (step_max / np.linalg.norm(sk))
            print("atom_coords_old:",atom_coords_new)
            atom_coords = atom_coords_new
            atom_coords_new = {}
            
            # for i in range(1, num_atoms + 1):
            #     atom_coords_new[str(i)] = atom_coords[str(i)] + sk[3 * (i-1):3 * (i)]
            for i in range(1, num_atoms + 1):   
                step_k = alpha * pk[i - 1] #sk = alphak * pk
                step_k_norm = np.linalg.norm(step_k)
                # print("step_k_norm:",step_k_norm)
                if step_k_norm > step_max:
                    step_k = step_k * (step_max / step_k_norm)
                # print("step_k:",step_k)
                # print("atom_coords_broken:",atom_coords[str(i)])
                # print("sk to be added:",sk[3 * (i-1):3 * (i)])
                atom_coords_new[str(i)] = atom_coords[str(i)] + sk[3 * (i-1):3 * (i)]
                # print("atom_coords_broken after update:",atom_coords_new[str(i)])
                # atom_coords_new[str(i)] = atom_coords[str(i)] + step_k # r_k+1 = r_k + sk
            # print("sk after update:",sk)
            # print("atom_coords_new:",atom_coords_new)
            # print("atom_coords_new:",atom_coords_new)

            
            
            E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
            # print("E_k:",E_k)
            #check wolfe condition
            c1 = 0.1

            i_lim = 10
            i_count = 0

            while E_k > E_k1 + (c1 * alpha * np.dot(pk_flat,grad_1_values_flat)):
                print("Wolfe condition not satisfied")
                alpha = alpha * 0.8
                print("new alpha:",alpha)
                # print("E_k-1:",E_k1)
                print("E_k anterior:",E_k)
                # print("wolf term:",E_k1 + (c1 * alpha * np.dot(pk_flat,grad_rk_values_flat)))

                sk = alpha * pk_flat
                # if np.linalg.norm(sk) > 0.3:
                #     sk = sk * (0.3 / np.linalg.norm(sk))

                # for i in range(1, num_atoms + 1):
                #     atom_coords_new[str(i)] = atom_coords[str(i)] + sk[3 * (i-1) : 3 * (i)]
            
                # E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)   
            
                for i in range(1, num_atoms + 1):
                        step_k = alpha * pk[i - 1] #sk = alphak * pk
                        step_k_norm = np.linalg.norm(step_k)
                        if step_k_norm > step_max:
                            print("step too big, normalizing to step_max:",step_max)
                            step_k = step_k * (step_max / step_k_norm)
                        # atom_coords_new[str(i)] = atom_coords[str(i)] + step_k # r_k+1 = r_k + sk
                        atom_coords_new[str(i)] = atom_coords[str(i)] + sk[3 * (i-1):3 * (i)]
                print("coords-new",atom_coords_new)
                E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
                # Incrementando o contador e verificando o limite de iterações
                print("E_k before else do while:",E_k)
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
                print("grad_k-1_values:",grad_1_values)
                # grad_rk_new = gradients.calculate_bond_stretching_gradient(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)
                # grad_rk_new_values = np.array(list(grad_rk_new.values()))
                grad_k_new = gradients.gradient_full(file_name, atom_types, atom_coords_new, bonds, num_atoms, read_coordinates_from_file=False, coordinates=atom_coords_new)
                grad_k_new_values = np.array(list(grad_k_new.values()))
                print("grad_k_values:",grad_k_new_values)
                yk = grad_k_new_values - grad_1_values
                yk_flat = yk.flatten()
                vk = np.dot(Mk1,yk_flat)
                sk_dot_yk = np.dot(sk,yk_flat)
                yk_dot_vk = np.dot(yk_flat,vk)
                sk_x_sk = np.outer(sk,sk)
                vk_x_sk = np.outer(vk,sk)
                sk_x_vk = np.outer(sk,vk)

                delta_E = E_k - E_k1
                print("delta_E:",delta_E,"k:",k)
                if np.abs(delta_E) <= threshold:
                    print("Convergence reached on k:",k)
                    print("final energy:",E_k)
                    print("final coordinates:",atom_coords_new)
                    break


                Mk_new = Mk1 + ((np.dot((sk_dot_yk + yk_dot_vk),sk_x_sk))/(sk_dot_yk**2)) - ((vk_x_sk + sk_x_vk)/(sk_dot_yk))
                print("Mk:",Mk_new)

                print("atom_coords_new:",atom_coords_new)

                print("step k finalized:",k)
                   

                # grad_rk1_values = grad_rk_new_values
                grad_0_values = grad_1_values
                grad_1_values = grad_k_new_values
                E_k1 = E_k
                Mk1 = Mk_new
        # if k == 2:
        #     break
                

def optimize_bfgs_internal (file_name):

    # Read the input file
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)
    # Get the initial gradient
    grad0 = gradients.gradient_full(file_name, atom_types, atom_coords, bonds, num_atoms)
    grad_0_values = np.array(list(grad0.values()))

    # print("atom_coords: ", atom_coords) 

    atom_coords_internal = internal_coord.cartesian_to_internal(file_name)
    print("atom_coords_internal:",atom_coords_internal)

    #calculates initial energy
    E0 = energies.total_energy(file_name, atom_types)

    grad_0_values_flat = grad_0_values.flatten()

    #get number of bond lengths
    num_bonds = len(bonds)
    # print("num bonds: ",num_bonds)

    #get number of angles
    num_angles = len(bond_angles.calculate_angles_from_bonds(atom_coords, bonds, atom_types))
    # print("num angles: ",num_angles)

    #get number of dihedrals
    torsion_angles, chains = bond_angles.calculate_torsion_angle(atom_coords, bonds, atom_types)
    num_dihedrals = len(torsion_angles)/2
    num_dihedrals = int(num_dihedrals)

    #build an array 
    M0_array = []

    for element in range(num_bonds):
        M0_array.append((1/600))
    for element in range(num_angles):
        M0_array.append((1/150))
    for element in range(num_dihedrals):
        M0_array.append((1/80))

    M0 = np.diag(M0_array)
    #print each line of the matrix with the line number in front of it
    # for i in range(len(M0)):
    #     print(i+1,M0[i])

    threshold = 0.001 # to be confirmed

    step_max = 0.02

    nq = num_bonds + num_angles + num_dihedrals
    nq = int(nq)





    for k in range(1, 2):
        
        if k == 1:
            grad0_cartesian = grad0
            grad0_cartesian_values = grad_0_values
            grad0_cartesian_values_flat = grad_0_values_flat
            # print("grad_cartesian_values_flat:",grad_cartesian_values_flat)

            # #print grad_cartesian_values_flat shape
            # print("grad_cartesian_values_flat shape:",grad_cartesian_values_flat.shape)

            # print("grad_cartesian_values:",grad_cartesian_values)

            B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name)
            # print("B:",B)
            # print("G_inverse:",G_inverse)
            #print shapes
            # print("B shape:",B.shape)
            # print("G_inverse shape:",G_inverse.shape)

            grad0_internal = np.dot(np.dot(G_inverse,B),grad0_cartesian_values_flat)
            # print("grad_internal:",grad_internal)

            pk1 = -np.dot(M0, grad0_internal)
            pk1_flat = pk1.flatten()

            alpha = 1.0

            atom_coords_new_internal = []

            sk1 = alpha * pk1_flat
            # print("sk1:",sk1)

            average_length = np.sqrt((sum([x**2 for x in sk1]))/nq)
            # print("average_length:",average_length)

            if average_length > step_max:
                sk1 = sk1 * (step_max / average_length)
                # print("sk1 after rescaling:",sk1)
            print("sk1 shape:",sk1.shape)
            #print atom_coord_internal shape
            print("atom_coords_internal shape:",atom_coords_internal.shape)

            #print each element of sk1
            for i in range(len(sk1)):
                print("sk1 element:",sk1[i])

            #update coordinates

            for i in range(1, len(atom_coords_internal) + 1):
                # print("i:",i)   
                print("atom_coords_internal[i] before:",atom_coords_internal[i-1])
                atom_coords_new_internal.append(atom_coords_internal[i-1] + sk1[i-1])
                # print("atom_coords_internal[i] after:",atom_coords_new_internal[i-1])
            
            print("atom_coords_new_internal:",atom_coords_new_internal)

            #step 5 non trivial step
            #convert atom_coords_new_internal to cartesian

            



            

        else:
            print("k is not 1, i is:",k)




            

    



            

        



# optimize_bfgs("ethane_dist")
optimize_bfgs_internal("ethane_dist")



    



    