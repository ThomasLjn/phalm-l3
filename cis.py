import numpy as np
import psi4

def Ha_to_eV(x):
    return ha_to_ev*x

def Ha_to_cm(x):
    return ha_to_cm*x

def Ha_to_Ry(x):
    return 2*x
def calc_C(Ca, Cb):
    dim = Ca.shape[0] + Cb.shape[0]
    C = np.zeros((dim, dim))
    for i in range(dim):
        if i%2 == 0:
            #alpha "en haut"
            C[:Ca.shape[0], i] = Ca[:, i//2]
        else:
            C[Cb.shape[0]:,i] = Cb[:, i//2]
    return C

def calc_excitations(nb_occ, nb_so):
    #a-nocc in {0...nvir-1}
    excitations = []
    for i in range(nb_occ):
        for a in range(nb_occ, nb_so):
            excitations.append((i, a))
    return excitations

def transfo_I(I, C):
    #Création de la matrice des intégrales par bloc de spin
    A = np.block([[I, np.zeros_like(I)],
                 [np.zeros_like(I), I]])
    
    I_oa = np.block([[A.T, np.zeros_like(A.T)],
                 [np.zeros_like(A.T), A.T]])

    #Changement de notation et antisymétrisation
    I_oa = I_oa.transpose(0, 2, 1, 3) - I_oa.transpose(0, 2, 3, 1)

    I_om = np.einsum('pQRS, pP -> PQRS',
          np.einsum('pqRS, qQ -> pQRS',
          np.einsum('pqrS, rR -> pqRS',
          np.einsum('pqrs, sS -> pqrS', I_oa, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)
    return I_om

def calc_H(nb_occ, nb_virt, E_HF, Iom):
    I2 = np.transpose(Iom, (2, 0, 1, 3))[:nb_occ, nb_occ:, :nb_occ, nb_occ:] #1, 2, 0, 3
    Eps = np.zeros((nb_occ*nb_virt, nb_occ*nb_virt))
    for i in range(nb_occ):
        for a in range(nb_virt):
            Eps[i*nb_virt + a, i*nb_virt + a] = E_HF[a+nb_occ] - E_HF[i]
    Hi = np.reshape(I2, (nb_occ*nb_virt, nb_occ*nb_virt))
    Hcis = Hi + Eps
    return Hcis

def output(Ecis, Excit, Ccis):
    for i, ex in enumerate(Excit):
        j, b = ex
        #print("Etat ", i, " transition ", j, " -> ", b, " énergie: ", Ecis[i], " Ha", " = ", Ha_to_eV(Ecis[i]), " eV")
        print(("Etat {} transition {} -> {} énergie: %.5f Ha" % Ecis[i]).format(i+1, j, b))
        Contrib = Ccis**2
        for k, c in enumerate(Contrib[:,i]):
            if c >= seuil_contrib:
                p,q = Excit[k]
                print(("Fonction {} -> {} contribuion: %.3f" % c).format(p, q))
                
        #print("Contribution: ", Contrib[:, i]) #On récupère la colonne i
        print(" ")


def calc_CIS(mol, hf_e, hf_wfn, mints, basis='6-311++G**',ref='rhf', output=False):
    psi4.set_memory("3 GB")
    psi4.set_options({'basis':        basis,
                      'scf_type':     'pk',
                      'reference':    ref,
                      'mp2_type':     'conv',
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})
    hf_e, hf_wfn = psi4.energy('scf', return_wfn=True)
    mints = psi4.core.MintsHelper(hf_wfn.basisset())
    
    nbf = mints.nbf() #Nombre de fonctions de base
    na = hf_wfn.nalpha() #Nombre d'ELECTRONS alpha
    nb = hf_wfn.nbeta()
    nocc = na + nb #Nombre de spin-orbitales occupées
    nso = 2*nbf #Nombre de spin orbitales: deux par fonction de base: psi^alpha et psi^beta
    nvir = nso-nocc #Nombre d'orbitales vacantes/virtuelles
    
    #Pour éviter les problèmes de symétries, on récupère toutes les orbitales (sans les classer par symétrie)
    eps_a = np.asarray(hf_wfn.epsilon_a_subset("AO", "ALL"))
    eps_b = np.asarray(hf_wfn.epsilon_b_subset("AO", "ALL"))
    eps = np.append(eps_a, eps_b)
    
    #On def la matrice des coef comme une alternance de colonnes orthogonales alpha et beta
    Ca = np.asarray(hf_wfn.Ca_subset("AO", "ALL"))
    Cb = np.asarray(hf_wfn.Cb_subset("AO", "ALL"))
    C = np.block([ [Ca, np.zeros_like(Cb)], [np.zeros_like(Ca), Cb] ])
    C = C[:, eps.argsort()]

    eps = np.sort(eps)
    
    I = mints.ao_eri()
    I_om = transfo_I(I, C)
    
    #Excitations = calc_excitations(nocc, nso)
    
    #def calc_H(Exc, nb_occ, nb_virt, E_HF, Iom):
    H_cis = calc_H(nocc, nvir, eps, I_om)
    return H_cis, I_om, C, eps
    '''ECIS, CCIS = np.linalg.eigh(H_cis)
    if output:
        output(ECIS, Excitations, CCIS)
    return ECIS, CCIS'''





