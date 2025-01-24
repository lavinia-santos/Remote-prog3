import numpy as np
import math
import gradients
import energies
import reading
import bond_angles
import internal_coord
import copy





def normalize_2pi(radianos):
    radianos = radianos % (2 * math.pi)
    if radianos > (math.pi):
        radianos -= 2 * math.pi
    return radianos

def optimize_bfgs_internal (file_name):

    # Read the input file
    num_atoms, num_bonds, num_atom_types, atom_coords, bonds, atom_types = reading.read_input(file_name)
    # Get the initial gradient
    grad0 = gradients.gradient_full(file_name, atom_types, atom_coords, bonds, num_atoms)
    grad_0_values = np.array(list(grad0.values()))

    # print("atom_coords: ", atom_coords) 

    atom_coords_internal0 = internal_coord.cartesian_to_internal(file_name)
    print("atom_coords_internal:",atom_coords_internal0)

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

    threshold = 0.0001 # to be confirmed

    step_max = 0.02

    nq = num_bonds + num_angles + num_dihedrals
    nq = int(nq)


    for k in range(1, 3):
        
        if k == 1:
            grad0_cartesian = grad0
            grad0_cartesian_values = grad_0_values
            grad0_cartesian_values_flat = grad_0_values_flat #g_x,k


            B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name) #B_k, G-_k 

            # print("B matrix:\n",B)
            # print("G inverse matrix:\n",G_inverse)


            grad0_internal = np.dot(np.dot(G_inverse,B),grad0_cartesian_values_flat) #g_q,k
            # print("grad_internal:",grad0_internal)

            pk1 = -np.dot(M0, grad0_internal)
            pk1_flat = pk1.flatten() #p_k,q

            alpha = 1.0

            atom_coords_previous_internal = np.array(atom_coords_internal0.copy()) #q_k
            # atom_coords_new_internal = [] 
            atom_coords_desired_internal = [] #q_k+1
            #equivalent to atom_coords_new in cartesian coordinates

            sk1 = alpha * pk1_flat #s_q,k = s_k (internal)
            print("Predicted update step in internal coordinates s_k (prior to possible scaling):\n",sk1)

            average_length = np.sqrt((sum([x**2 for x in sk1]))/nq) #l_q,k
            # print("average_length:",average_length)



            if average_length > step_max:
                sk1 = sk1 * (step_max / average_length)
                print("Scaled update step in internal coordinates s_k:\n",sk1)

            #update coordinates
            #giving the step
            #q_k+1 = q_k + s_q,k
            for i in range(1, len(atom_coords_internal0) + 1):
                # atom_coords_current_internal.append(atom_coords_internal0[i-1]) #before step
                atom_coords_desired_internal.append(atom_coords_previous_internal[i-1] + sk1[i-1]) #after step
                # q_k+1            =                          q_k +                      s_q,k
                

            ##desired internal coordinates = atom_coords_desired_internal##
            #calculate new energy
            #but for that, we need to convert to cartesian coordinates
            
            
            #preparing step for the next one
            s0 = atom_coords_desired_internal - atom_coords_previous_internal
            for i in range((num_bonds+num_angles),len(s0)+1):
                s0[i-1] = normalize_2pi(s0[i-1])

            #preparing to find the equivalent cartesian coordinates
            #define initial guess for cartesian coordinates as the the previous one // b
            atom_coords_previous_cartesian = atom_coords.copy()
            atom_coords_new_cartesian = {}
            #   x_k+1 ^0                        =     x_0


            threshold_cartesian = 0.00001

            B0, G_inverse0 = internal_coord.calculate_B_and_G_matrices(file_name,read_coordinates_from_file=False,coordinates=atom_coords_previous_cartesian) #B_k, G-_k 
            # print("B0 matrix:\n",B0)
            # print("G0 inverse matrix:\n",G_inverse0)


            B0_transpose = np.transpose(B0) 

            dx = np.dot(np.dot(B0_transpose,G_inverse0),s0) # dx = B^T * G- * sk
            print("Initially predicted dx = BTG-sk:\n",dx)

            

            for atom in atom_coords_previous_cartesian.keys(): #update cartesian coordinates (x_new)
                #updating x_k+1^(j+1) = x_k+1^j + dx
                atom = int(atom)
                """print("atom:",atom)
                        print("atom_coords_new_cartesian[atom] before:",atom_coords_previous_cartesian[str(atom)])
                        # print("dx positions:",3 * (atom - 1),"to",3 * atom)
                        # print("dx values:",dx[3 * (atom - 1):3 * atom])"""
                atom_coords_new_cartesian[str(atom)] = atom_coords_previous_cartesian[str(atom)] + dx[3 * (atom - 1):3 * atom] #c
                #x_k+1^(j+1)                         =             x_k+1^j                       +  dx
            

            
            # print("atom_coords_new_cartesian:",atom_coords_new_cartesian) #not printed in profs output
            atom_coords_new_internal = internal_coord.cartesian_to_internal(file_name, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian) # converting new cartesian structure to internal coordinates - d
            # print("atom_coords_new_internal",atom_coords_new_internal) #q_0

            #needed for next inner step
            #because we need it to evaluate the next dx
            s0 = atom_coords_desired_internal - atom_coords_new_internal 
            for i in range((num_bonds+num_angles),len(s0)+1):
                s0[i-1] = normalize_2pi(s0[i-1])


            # print("s0",s0)

            atom_coords_previous_cartesian = atom_coords_new_cartesian.copy()
            delta_x_max = 1
            print("\n\n iteration will start from here, current parameters are:\n")
            print("atom_coords_previous_internal:",atom_coords_previous_internal)
            print("atom_coords_desired_internal:",atom_coords_desired_internal)
            print("atom_coords_new_internal:",atom_coords_new_internal)

            ##########################################################################################################################################
            ################################################### starting step 5 iteration - non trivial step #########################################
            ##########################################################################################################################################
            # for steps_internal_to_cartesian in range(1, 10): #iterating j - c
            steps_internal_to_cartesian = 1
            while delta_x_max > threshold_cartesian:
                print("Starting cartesian iteration")
                

                print("\n\nCartesian fitting iteration number:\n",steps_internal_to_cartesian)
                print("k:",k)

                #print current set of internals
                print("current set of internals q_(k+1)^(j):\n",atom_coords_new_internal)

                s0 = atom_coords_desired_internal - atom_coords_new_internal
                for i in range((num_bonds+num_angles),len(s0)+1):
                    s0[i-1] = normalize_2pi(s0[i-1])



                print("difference between these internals (q_(k+1)^(j)) and the desired internals (q_k+1), s_q,k^j: \n",s0)
                #evaluate new dx
                # B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name,read_coordinates_from_file=False,coordinates=atom_coords_previous_cartesian)

                dx = np.dot(np.dot(B0_transpose,G_inverse0),s0) # dx = B^T * G- * sk
                print("predicted dx = BTG-sk:\n",dx)
                for atom in atom_coords_previous_cartesian.keys(): #update cartesian coordinates - c
                    #updating x_k+1^(j+1) = x_k+1^j + dx
                    atom = int(atom)
                    atom_coords_new_cartesian[str(atom)] = atom_coords_previous_cartesian[str(atom)] + dx[3 * (atom - 1):3 * atom] #c
                    #x_k+1^(j+1)                         =             x_k+1^j                       +  dx



                print("Corresponding Cartesians x+k+1^j:\n",atom_coords_new_cartesian)

                atom_coords_new_internal = internal_coord.cartesian_to_internal(file_name, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian) # converting new cartesian structure to internal coordinates - d
                
                delta_x_full = []
                for atom in atom_coords_new_cartesian.keys(): #see the difference from previous to current cartesian structure
                    delta_x = atom_coords_new_cartesian[str(atom)] - atom_coords_previous_cartesian[str(atom)]
                    delta_x_full.append(delta_x)
                
                # print("delta_x_full:",delta_x_full)
                # print("JUST ADDED, NOT SHOWING IN OUTPUT dx = BTG-sk:\n",dx)

                delta_x_max = np.max(np.abs(delta_x_full))
                print("Maximum change in x from previous iteration before else do while",delta_x_max)
                E_k1 = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian)

                # atom_coords_previous_internal = atom_coords_new_internal.copy()
                # atom_coords_new_cartesian = {}
                atom_coords_previous_cartesian = atom_coords_new_cartesian.copy()

            
                # if delta_x_max <= threshold_cartesian:
                steps_internal_to_cartesian += 1

                # if steps_internal_to_cartesian > 3:
                #     break
            else:
                #calculate B and G inverse for the new structure
                B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name,read_coordinates_from_file=False,coordinates=atom_coords_new_cartesian)
                print("Convergence reached on internal step:",steps_internal_to_cartesian)
                # print("atom_coords_previous_cartesian:",atom_coords_previous_cartesian)  
                print("atom_coords_new_cartesian converged:",atom_coords_new_cartesian)
                print("atom_coords_new_internal:",atom_coords_new_internal)
                print("B matrix in the new strcture:\n",B)
                print("G inverse matrix in the new strcture:\n",G_inverse)
                # print("atom_coords_previous_internal:",atom_coords_previous_internal)
                sk = atom_coords_new_internal - atom_coords_previous_internal
                print("Calculating new Mk...")
                
                grad1_cartesian = gradients.gradient_full(file_name, atom_types, atom_coords_new_cartesian, bonds, num_atoms, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian)
                grad1_cartesian_values = np.array(list(grad1_cartesian.values()))
                grad1_cartesian_values_flat = grad1_cartesian_values.flatten()
                grad1_internal = np.dot(np.dot(G_inverse,B),grad1_cartesian_values_flat)
                print("grad1_internal:",grad1_internal)
                # grad1_internal_values = np.array(list(grad1_internal.values()))
                y_qk = grad1_internal - grad0_internal
                print("sk:",sk)
                print("y_qk:",y_qk)
                v_qk = np.dot(M0,y_qk)
                print("v_qk:",v_qk)
                s_qk_dot_y_qk = np.dot(sk,y_qk)
                y_qk_dot_v_qk = np.dot(y_qk,v_qk)
                s_qk_x_s_qk = np.outer(sk,sk)
                v_qk_x_s_qk = np.outer(v_qk,sk)
                s_qk_x_v_qk = np.outer(sk,v_qk)
                M1 = M0 + ((np.dot((s_qk_dot_y_qk + y_qk_dot_v_qk),s_qk_x_s_qk))/(s_qk_dot_y_qk**2)) - ((v_qk_x_s_qk + s_qk_x_v_qk)/(s_qk_dot_y_qk))
                print("M1:",M1)
                # atom_coords_previous_internal = atom_coords_new_internal.copy()
                # atom_coords_current_internal = atom_coords_new_internal.copy()
                # atom_coords_previous_cartesian = atom_coords_new_cartesian.copy()
                # E_k1 = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian)
                delta_E = E_k1 - E0
                #print old and new energies
                print("E_old:",E0)
                print("E_new:",E_k1)
                grms = np.sqrt(np.dot(grad1_internal,grad1_internal))/len(grad1_internal)
                if grms <= threshold:
                    print("Convergence reached on k:",k)
                    print("final energy:",E_k1)
                    print("final coordinates cartesian:",atom_coords_new_cartesian)
                    print("final coordinates internal:",atom_coords_new_internal)
                    atom_coords_previous_internal = atom_coords_new_internal.copy()
                    break
            
                
                # atom_coords_previous_cartesian = atom_coords_new_cartesian.copy()


        else:
            print("k is not 1, it is:",k)
            # grad0_internal = grad1_internal
            # grad0_internal_values = grad1_internal_values
            # grad0_internal_values_flat = grad1_internal_values.flatten()

            pk = -np.dot(M1, grad1_internal)

            pk_flat = pk.flatten()

            alpha = 1.0

            atom_coords_previous_internal = np.array(atom_coords_new_internal.copy())
            # atom_coords_new_internal = []
            atom_coords_desired_internal = [] #equivalent to atom_coords_new in cartesian coordinates

            sk = alpha * pk_flat

            average_length = np.sqrt((sum([x**2 for x in sk]))/nq)

            if average_length > step_max:
                sk = sk * (step_max / average_length)
                # print("sk after rescaling:",sk)

            #update coordinates
            for i in range(1, len(atom_coords_internal0) + 1):
                atom_coords_desired_internal.append(atom_coords_previous_internal[i-1] + sk[i-1])
            
            s0 = atom_coords_desired_internal - atom_coords_previous_internal
            for i in range((num_bonds+num_angles),len(s0)+1):
                s0[i-1] = normalize_2pi(s0[i-1])

            atom_coords_previous_cartesian = atom_coords_new_cartesian.copy()
            atom_coords_new_cartesian = {}

            # B0, G_inverse0 = internal_coord.calculate_B_and_G_matrices(file_name,read_coordinates_from_file=False,coordinates=atom_coords_previous_cartesian)

            threshold_cartesian = 0.00001
            
            # B0_transpose = np.transpose(B0) 

            B_transpose = np.transpose(B)

            #calculando primeiro dx, antes do loop de internal to cartesian
            dx = np.dot(np.dot(B_transpose,G_inverse),s0) # dx = B^T * G- * sk
            # print("G inverse:\n",G_inverse0)
            
            print("Initially predicted dx = BTG-sk:\n",dx)

            for atom in atom_coords_previous_cartesian.keys(): #update cartesian coordinates (x_new)
                #updating x_k+1^(j+1) = x_k+1^j + dx
                atom = int(atom)
                atom_coords_new_cartesian[str(atom)] = atom_coords_previous_cartesian[str(atom)] + dx[3 * (atom - 1):3 * atom] #c
                #x_k+1^(j+1)                         =             x_k+1^j                       +  dx
            

            
            # print("atom_coords_new_cartesian:",atom_coords_new_cartesian) #not printed in profs output
            atom_coords_new_internal = internal_coord.cartesian_to_internal(file_name, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian) # converting new cartesian structure to internal coordinates - d
            # print("atom_coords_new_internal",atom_coords_new_internal) #q_0

            s0 = atom_coords_desired_internal - atom_coords_new_internal 

            for i in range((num_bonds+num_angles),len(s0)+1):
                s0[i-1] = normalize_2pi(s0[i-1])

            # print("s0",s0)

            atom_coords_previous_cartesian = atom_coords_new_cartesian.copy()
            # atom_coords_previous_cartesian = atom_coords_new_cartesian.copy()



            # for steps_internal_to_cartesian in range(1, 10):
            steps_internal_to_cartesian = 1
            delta_x_max = 1
            while delta_x_max > threshold_cartesian:

                
                print("\n\nCartesian fitting iteration number:\n",steps_internal_to_cartesian)

                #print current set of internals
                print("current set of internals q_(k+1)^(j):\n",atom_coords_new_internal)

                s0 = atom_coords_desired_internal - atom_coords_new_internal
                for i in range((num_bonds+num_angles),len(s0)+1):
                    s0[i-1] = normalize_2pi(s0[i-1])
                
                # B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name,read_coordinates_from_file=False,coordinates=atom_coords_previous_cartesian)                
                # B_transpose = np.transpose(B) 
                dx = np.dot(np.dot(B0_transpose,G_inverse0),s0)
                # print("dx:",dx)
                
                for atom in atom_coords_previous_cartesian.keys():
                    atom = int(atom)
                    atom_coords_new_cartesian[str(atom)] = atom_coords_previous_cartesian[str(atom)] + dx[3 * (atom - 1):3 * atom]

                print("Corresponding Cartesians x+k+1^j:\n",atom_coords_new_cartesian)

                atom_coords_new_internal = internal_coord.cartesian_to_internal(file_name, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian)


                delta_x_full = []

                for atom in atom_coords_new_cartesian.keys():
                    delta_x = atom_coords_new_cartesian[str(atom)] - atom_coords_previous_cartesian[str(atom)]
                    delta_x_full.append(delta_x)

                delta_x_max = np.max(np.abs(delta_x_full))


                E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian)
                atom_coords_previous_cartesian = atom_coords_new_cartesian.copy()
                steps_internal_to_cartesian += 1
                

            else:
                #calculate B and G inverse for the new structure
                B, G_inverse = internal_coord.calculate_B_and_G_matrices(file_name,read_coordinates_from_file=False,coordinates=atom_coords_new_cartesian)
                print("Convergence reached on internal step:",steps_internal_to_cartesian)
                print("atom_coords_previous_cartesian:",atom_coords_previous_cartesian)  
                print("atom_coords_new_cartesian converged:",atom_coords_new_cartesian)
                print("atom_coords_new_internal:",atom_coords_new_internal)
                print("B matrix in the new strcture:\n",B)
                print("G inverse matrix in the new strcture:\n",G_inverse)
                # print("atom_coords_previous_internal:",atom_coords_previous_internal)
                sk = atom_coords_new_internal - atom_coords_previous_internal
            
                grad1_cartesian = gradients.gradient_full(file_name, atom_types, atom_coords_new_cartesian, bonds, num_atoms, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian)
                grad1_cartesian_values = np.array(list(grad1_cartesian.values()))
                grad1_cartesian_values_flat = grad1_cartesian_values.flatten()
                grad1_internal = np.dot(np.dot(G_inverse,B),grad1_cartesian_values_flat)
                y_qk = grad1_internal - grad0_internal
                v_qk = np.dot(M1,y_qk)
                s_qk_dot_y_qk = np.dot(sk,y_qk)
                y_qk_dot_v_qk = np.dot(y_qk,v_qk)
                s_qk_x_s_qk = np.outer(sk,sk)
                v_qk_x_s_qk = np.outer(v_qk,sk)
                s_qk_x_v_qk = np.outer(sk,v_qk)
                Mk_new = M1 + ((np.dot((s_qk_dot_y_qk + y_qk_dot_v_qk),s_qk_x_s_qk))/(s_qk_dot_y_qk**2)) - ((v_qk_x_s_qk + s_qk_x_v_qk)/(s_qk_dot_y_qk))

                # E_k = energies.total_energy(file_name, atom_types, read_coordinates_from_file=False, coordinates=atom_coords_new_cartesian)
                print("E_old:",E_k1)
                print("E_new:",E_k)
                delta_E = E_k - E_k1
                grms = np.sqrt(np.dot(grad1_cartesian_values_flat,grad1_cartesian_values_flat)/len(grad1_cartesian_values_flat))
                print("grms:",grms)
                
                if grms <= threshold:
                    print("Convergence reached on k:",k)
                    print("final energy:",E_k)
                    print("grms:",grms)
                    print("final coordinates cartesian:",atom_coords_new_cartesian)
                    print("final coordinates internal:",atom_coords_new_internal)
                    atom_coords_previous_internal = atom_coords_new_internal.copy()
                    break
                
                # atom_coords_previous_cartesian = atom_coords_new_cartesian.copy()
                grad0_internal = grad1_internal
                E_k1 = E_k
                M1 = Mk_new
                




optimize_bfgs_internal("ethane")



    

