
import numpy as np

def transform_interaction_tensor(occupied_matrix, virtual_matrix,
                                 interaction_matrix, chi_tensor):
    '''Returns a transformed V tensor defined by the input occupied, virtual, & interaction matrices.'''
    chi2_tensor = np.einsum('qa,ri,qrp',
                            virtual_matrix,
                            occupied_matrix,
                            chi_tensor,
                            optimize=True)
    interaction_tensor = np.einsum('aip,pq,bjq->aibj',
                                   chi2_tensor,
                                   interaction_matrix,
                                   chi2_tensor,
                                   optimize=True)
    return interaction_tensor



def calculate_energy_mp2(fock_matrix, interaction_matrix, chi_tensor):
    '''Returns the MP2 contribution to the total energy defined by the input Fock & interaction matrices.'''
    E_occ, E_virt, occupied_matrix, virtual_matrix = partition_orbitals(fock_matrix)
    V_tilde = transform_interaction_tensor(occupied_matrix, virtual_matrix,
                                           interaction_matrix, chi_tensor)

    energy_mp2 = 0.0
    num_occ = len(E_occ)
    num_virt = len(E_virt)
    for a in range(num_virt):
        for b in range(num_virt):
            for i in range(num_occ):
                for j in range(num_occ):
                    energy_mp2 -= (
                        (2.0 * V_tilde[a, i, b, j]**2 -
                         V_tilde[a, i, b, j] * V_tilde[a, j, b, i]) /
                        (E_virt[a] + E_virt[b] - E_occ[i] - E_occ[j]))
    return energy_mp2

