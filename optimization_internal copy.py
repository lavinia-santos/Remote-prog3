import numpy as np
import math
import gradients
import energies
import reading
import bond_angles
import internal_coord
import copy

def normalize_2pi(radianos):
    # Reduz o ângulo para o intervalo [0, 2π)
    radianos = radianos % (2 * math.pi)
    # Ajusta para o intervalo [-π, π]
    if radianos > (math.pi):
        radianos -= 2 * math.pi
    return radianos


def optimize_bfgs_internal (file_name):
    """
    This function is used for optimizing the geometry of a molecule using the BFGS algorithm.
    It reads the input file, calculates the gradient, and updates the coordinates of the atoms.
    """
    # Read the input file
    #pagina 8 starters
    #First, one chooses an initial structure x0,
    num_atoms, num_bonds, num_atom_types, x0, bonds, atom_types = reading.read_input(file_name)
    
    #and converts it to an initial structure q0
    q0 = internal_coord.cartesian_to_internal(file_name)

    #One also makes an initial guess for the inverse Hessian in the space of internal coordinates, Mq,0

    num_bonds = len(bonds)

    num_angles = len(bond_angles.calculate_angles_from_bonds(x0, bonds, atom_types))

    torsion_angles, chains = bond_angles.calculate_torsion_angle(x0, bonds, atom_types)
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

    threshold = 0.001 # to be confirmed

    step_max = 0.02

    nq = num_bonds + num_angles + num_dihedrals
    nq = int(nq)

    for k in range(1, 250):
        
        if k == 1:

            #step 1 pag 8

            #Calculate the gradient in Cartesian coordinates gx,k,
            gx = gradients.gradient_full(file_name, atom_types, x0, bonds, num_atoms)
            # print("gx:",gx)
            gx_values = np.array(list(gx.values()))
            gx_values_flat = gx_values.flatten()

            #the Wilson B matrix Bk and its generalized inverse G−k,
            B0, G0_inverse = internal_coord.calculate_B_and_G_matrices(file_name)

            #and hence, using eqs. 19, obtain the gradient gq,k
            gq = np.dot(np.dot(G0_inverse,B0), gx_values_flat)

            ############################################################################################################
            #step 2 pag 8
            #Obtain the search direction in internal coordinates pq,k, where pq,k = −Mq,kgq,k.

            p_q = -np.dot(M0, gq) 
            # print("p_q:",p_q)

            ############################################################################################################
            #step 3 pag 8
            #For scaling, calculate the average length of pq,k, lq,k = sqrt(1/nq Σnq i=1 (pq,k,i)^2), where nq is the number of internal coordinates.
            #, and smaller than step_max
            #otherwise set pq,k = pq,k × (λmax/lq,k).

            alpha = 1.0

            l_q = np.sqrt((sum([x**2 for x in p_q]))/nq) #l_q,k
            if l_q > step_max:
                p_q = p_q * (step_max / l_q)

            ############################################################################################################
            #step 4 pag 8
            #Define sq,k = αkpq,k
            s_q = alpha * p_q

            #and hence qk+1 = qk + sq,k.
            q_k1 = q0 + s_q
            # print("q_k1:",q_k1)

            ############################################################################################################
            #step 5 pag 8
            #Convert the obtained updated structure in internal coordinates qk+1 into an updated structure
            #in Cartesian coordinates xk+1.

            #doing a previous step before the while loop

            s_q_0 = q_k1 - q0

            x_k1_0 = x0

            B0_transpose = np.transpose(B0)

            dx = np.dot(np.dot(B0_transpose,G0_inverse),s_q_0)
            # print("dx:",dx)

            for i in range(1, num_atoms + 1):
                x_k1_0[str(i)] = x0[str(i)] + dx[3 * (i-1):3 * (i)]

            q_k1_next = internal_coord.cartesian_to_internal(file_name, read_coordinates_from_file=False, coordinates=x_k1_0)

            s_q_next = q_k1 - q_k1_next
            

            l_q = np.sqrt((sum([x**2 for x in s_q_next]))/nq) #l_q,k
            if l_q > step_max:
                s_q_next = s_q_next * (step_max / l_q)

            """# Calculate the new energy
            # E_k1 = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new)


            #line search
            #check wolfe condition
            # c1 = 0.1"""
            delta_x_max = 1.0

            step_number = 1

            while delta_x_max > threshold:
                print("delta x max too large")
                print("Cartesian iteration:",step_number)
                # print("q_k1_next",q_k1_next)
                # print("s_q_next",s_q_next)
                s_q_current = s_q_next

                x_k1_next = x_k1_0.copy()

                B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name, read_coordinates_from_file=False, coordinates=x_k1_0)
                B_transpose = np.transpose(B)
                dx = np.dot(np.dot(B_transpose,G_inverse),s_q_current)

                for i in range(1, num_atoms + 1):
                    x_k1_next[str(i)] = x_k1_0[str(i)] + dx[3 * (i-1):3 * (i)]
                
                print("Corresponding Cartesian coordinates:",x_k1_next)

                q_k1_next = internal_coord.cartesian_to_internal(file_name, read_coordinates_from_file=False, coordinates=x_k1_next)

                s_q_next = q_k1 - q_k1_next
                for i in range((num_bonds+num_angles),len(s_q_next)+1):
                    s_q_next[i-1] = normalize_2pi(s_q_next[i-1])

                delta_x_full = []
                for atom in x_k1_next.keys():
                    delta_x = x_k1_next[atom] - x_k1_0[atom]
                    delta_x_full.append(delta_x)
                
                delta_x_max = np.max(np.abs(delta_x_full))

                gx = gradients.gradient_full(file_name, atom_types, x_k1_0, bonds, num_atoms, read_coordinates_from_file=False, coordinates=x_k1_0)
                gx_values = np.array(list(gx.values()))
                gx_values_flat = gx_values.flatten()
                B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name, read_coordinates_from_file=False, coordinates=x_k1_0)
                gq = np.dot(np.dot(G_inverse,B), gx_values_flat)
                x_k1_0 = x_k1_next.copy()
                

                print("delta_x_max before else do while:",delta_x_max)
                step_number += 1
            else:
                print("Cartesian iteration converged")
                print("delta_x_max:",delta_x_max)
                print("New structure in Cartesian coordinates:",x_k1_next)
                #step 6 pag 8
                #Calculate the gradients gx,k+1 and gq,k+1 at the new structure
                gx_1 = gradients.gradient_full(file_name, atom_types, x_k1_next, bonds, num_atoms, read_coordinates_from_file=False, coordinates=x_k1_next)
                gx_1_values = np.array(list(gx_1.values()))
                gx_1_values_flat = gx_1_values.flatten()
                B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name, read_coordinates_from_file=False, coordinates=x_k1_next)
                gq_1 = np.dot(np.dot(G_inverse,B), gx_1_values_flat)
                yk = gq_1 - gq
                vk = np.dot(M0,yk)
                sk_dot_yk = np.dot(s_q_current,yk)
                yk_dot_vk = np.dot(yk,vk)
                sk_x_sk = np.outer(s_q_current,s_q_current)
                vk_x_sk = np.outer(vk,s_q_current)
                sk_x_vk = np.outer(s_q_current,vk)
                           
                Mk1 = M0 + ((np.dot((sk_dot_yk + yk_dot_vk),sk_x_sk))/(sk_dot_yk**2)) - ((vk_x_sk + sk_x_vk)/(sk_dot_yk))
                print("Mk:",Mk1)
                print("step k finalized:",k)

                E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=x_k1_next)

                delta_gx_full = gx_1_values - gx_values
                delta_gx_full_flat = delta_gx_full.flatten()
                # for element in range(len(delta_gx_full_flat)):
                #     print("delta_gx_full[element]:",delta_gx_full_flat[element])
                delta_gx_max = np.max(np.abs(delta_gx_full_flat))
                
                


                if delta_gx_max <= threshold:
                    print("Convergence reached on k:",k)
                    print("final energy:",E_k)
                    print("final coordinates:",x_k1_next)
                    break


        else:
            
            print("k is not 1, i is:",k)
            pk = -np.dot(Mk1, gq_1)
            # print("pk:",pk)

            pk_flat = pk.flatten()
            # print("pk_flat:",pk_flat)

            alpha = 1.0
            
            
            sk = alpha * pk_flat

            l_q = np.sqrt((sum([x**2 for x in sk]))/nq) #l_q,k
            if l_q > step_max:
                sk = sk * (step_max / l_q)

            x_previous = x_k1_next.copy()
            x_k1_next = {}
            
            for i in range(1, num_atoms + 1):   
                x_k1_next[str(i)] = x_previous[str(i)] + sk[3 * (i-1):3 * (i)]


            delta_x_max = 1.0

            step_number = 1
            

            while delta_x_max > threshold:
                print("delta x max too large")
                print("Cartesian iteration:",step_number)

                s_q_current = sk
                
                x_k1_next = x_previous.copy()

                B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name, read_coordinates_from_file=False, coordinates=x_previous)
                B_transpose = np.transpose(B)
                dx = np.dot(np.dot(B_transpose,G_inverse),s_q_current)

                for i in range(1, num_atoms + 1):
                    x_k1_next[str(i)] = x_previous[str(i)] + dx[3 * (i-1):3 * (i)]
                
                print("Corresponding Cartesian coordinates:",x_k1_next)

                q_k1_next = internal_coord.cartesian_to_internal(file_name, read_coordinates_from_file=False, coordinates=x_k1_next)

                s_q_next = q_k1 - q_k1_next

                for i in range((num_bonds+num_angles),len(s_q_next)+1):
                    s_q_next[i-1] = normalize_2pi(s_q_next[i-1])

                delta_x_full = []

                for atom in x_k1_next.keys():
                    delta_x = x_k1_next[atom] - x_k1_0[atom]
                    delta_x_full.append(delta_x)
                
                delta_x_max = np.max(np.abs(delta_x_full))
                gx = gradients.gradient_full(file_name, atom_types, x_previous, bonds, num_atoms, read_coordinates_from_file=False, coordinates=x_previous)
                gx_values = np.array(list(gx.values()))
                gx_values_flat = gx_values.flatten()
                B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name, read_coordinates_from_file=False, coordinates=x_previous)
                gq = np.dot(np.dot(G_inverse,B), gx_values_flat)
                x_previous = x_k1_next.copy()

                print("delta_x_max before else do while:",delta_x_max)
                step_number += 1

                """if i_count >= i_lim:
                    print("Iteration limit reached for debugging.")
                    print("k:",k)
                    break"""
            else:
                print("Cartesian iteration converged")
                print("delta_x_max:",delta_x_max)
                print("New structure in Cartesian coordinates:",x_k1_next)
                #step 6 pag 8
                #Calculate the gradients gx,k+1 and gq,k+1 at the new structure
                gx_1 = gradients.gradient_full(file_name, atom_types, x_k1_next, bonds, num_atoms, read_coordinates_from_file=False, coordinates=x_k1_next)
                gx_1_values = np.array(list(gx_1.values()))
                gx_1_values_flat = gx_1_values.flatten()
                B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name, read_coordinates_from_file=False, coordinates=x_k1_next)
                gq_1 = np.dot(np.dot(G_inverse,B), gx_1_values_flat)
                yk = gq_1 - gq
                vk = np.dot(Mk1,yk)
                sk_dot_yk = np.dot(s_q_current,yk)
                yk_dot_vk = np.dot(yk,vk)
                sk_x_sk = np.outer(s_q_current,s_q_current)
                vk_x_sk = np.outer(vk,s_q_current)
                sk_x_vk = np.outer(s_q_current,vk)

                Mk1 = Mk1 + ((np.dot((sk_dot_yk + yk_dot_vk),sk_x_sk))/(sk_dot_yk**2)) - ((vk_x_sk + sk_x_vk)/(sk_dot_yk))
                print("Mk:",Mk1)
                print("step k finalized:",k)

                E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=x_k1_next)

                delta_gx = gx_1_values - gx_values

                if np.abs(delta_gx) <= threshold:
                    print("Convergence reached on k:",k)
                    print("final energy:",E_k)
                    print("final coordinates:",x_k1_next)
                    break
                
               



optimize_bfgs_internal("ethane_dist")