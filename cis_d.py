import psi4
import numpy as np
import cis
ev = 27.2114
cm = 219474.63
psi4.core.clean()

'''mol = psi4.geometry("""
0 1
C 0.0 0.0 -0.610748815948
O 0.0 0.0 0.609252792186
H -0.934044866868 0.0 -1.198601207606
H 0.934044866868 0.0 -1.198601207606
units angstrom
""")'''

mol = psi4.geometry("""
0 1
He
""")
basis = 'aug-cc-pvdz'
ref = 'rhf'
molname = "he"
cis_mp2 = False
fname = "{}_{}_cis_d_py.out".format(molname, basis)
f = open(fname, "w")
psi4.set_memory("20 GB")
psi4.core.set_output_file('{}_{}_cis_d_psi.dat'.format(molname, basis), True)

psi4.set_options({'basis':        basis,
                  'scf_type':     'pk',
                  'reference':    ref,
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})
#psi4.optimize('scf')#t = time.time()
hf_e, hf_wfn = psi4.energy('scf', return_wfn=True)
#print("Calcul SCF terminé \nEnergie SCF: ", hf_e)
mints = psi4.core.MintsHelper(hf_wfn.basisset())

nbf = mints.nbf() #Nombre de fonctions de base
na = hf_wfn.nalpha() #Nombre d'ELECTRONS alpha
nb = hf_wfn.nbeta()
nocc = na + nb #Nombre de spin-orbitales occupées
nso = 2*nbf #Nombre de spin orbitales: deux par fonction de base: psi^alpha et psi^beta
nvir = nso-nocc #Nombre d'orbitales vacantes/virtuelles

H_cis, I_om, C, eps = cis.calc_CIS(mol, hf_e, hf_wfn, mints, basis, ref )
#print("Temps d'execution avant diagonalisation: ", time.time()-t)
ECIS, CCIS = np.linalg.eigh(H_cis)

f.write("Molecule: {} ; base: {}\n".format(molname, basis))

f.write("Energy HF: {}\n".format(hf_e))
f.write("------------------------ CIS Excitations Energy ------------------------")
for i, e in enumerate(ECIS):
    f.write("Excitation {}  Ha: {}   eV: {}\n".format(i, e, e*ev))

#calcul CIS(D)
pmin = 0
pmax = 10

B = [[CCIS[j:j+nvir,k+pmin] for j in range(0, nvir*nocc, nvir)] for k in range(pmax-pmin)]

DeltaE = [[[[eps[a]+eps[b]-eps[i]-eps[j] for a in range(nocc, nso)] for b in range(nocc, nso)] for j in range(nocc)] for i in range(nocc)]
A = -np.divide(I_om[:nocc, :nocc, nocc:nso, nocc:nso], DeltaE, where=DeltaE!=0)

N1 = np.einsum('pib,ba -> pia', B, np.einsum('jkbc,jkca->ba', I_om[:nocc, :nocc, nocc:nso, nocc:nso],A))


N2 = np.einsum('pja,ji -> pia', B, np.einsum('jkbc, ikcb -> ji', I_om[:nocc, :nocc, nocc:nso, nocc:nso],A))
#print(np.allclose(N2[:, :nocc, nocc:nso], N2_2))

N3 = 2*np.einsum('pkc, ikac -> pia', np.einsum('jkbc, pjb -> pkc', I_om[:nocc, :nocc, nocc:nso, nocc:nso], B), A)
#somme sur j et b de B[p,j,a]*NU2[j,i,b]
Nu = (N1 + N2 + N3)/2
N1, N2, N3 = None, None, None

E2 = np.einsum('pia, pia -> p', B, Nu)
Nu = None #vider la mémoire

U1 = np.einsum('icab, pjc -> pijab', I_om[:nocc, nocc:nso, nocc:nso, nocc:nso], B)
U2 = -np.einsum('jcab, pic -> pijab', I_om[:nocc, nocc:nso, nocc:nso, nocc:nso], B)


U3 = -np.einsum('ijak, pkb -> pijab', I_om[:nocc, :nocc, nocc:nso, :nocc], B)
U4 = np.einsum('ijbk, pka -> pijab', I_om[:nocc, :nocc, nocc:nso, :nocc], B)


U = U1 + U2 + U3 + U4
U1, U2, U3, U4 = None, None, None, None
#print(np.allclose(U[:, :nocc, :nocc, nocc:nso, nocc:nso], U_2))

Deno = np.zeros((pmax-pmin, nocc, nocc, nvir, nvir))
#Deno = Delta[E]-ECIS[pmin:pmax]
for p in range(pmin, pmax):
    Deno[p-pmin] = DeltaE-ECIS[p]
#Deno[p,i,j,a,b] = DeltaE[i,j,a,b] - ECIS[p]
E1 = -np.sum(np.divide(U**2, Deno), axis=(1,2,3,4))/4
#voir comment on fait pour supprimer un array de la mémoire
E = E1+E2
f.write("------------------------ CIS(D) Excitations Energy ------------------------\n")

for i in range(len(E)):
    f.write("Etat {}: CIS(eV): {}, CIS(D) (eV): {}\n".format(i+pmin, ECIS[i+pmin]*ev, (ECIS[i+pmin]*ev+E[i]*ev)))




def calc_u_d(i, j, a, b):
    s = 0
    for c in range(nocc, nso):
        s += (I_om[a, b, c, j] - I_om[a, b, j, c])*CCIS[c, i] - (I_om[a, b, c, i] - I_om[a, b, i, c])*CCIS[c, j]
    for k in range(nocc):
        s += (I_om[k, a, i, j] - I_om[k, a, j, i])*CCIS[b, k] - (I_om[k, b, i, j] - I_om[k, b, j, i]) * CCIS[a, k]
    return s

def calc_u_t(i, j, k, a, b, c):
    s = (I_om[j,k,b,c] - I_om[j,k,c,b])*CCIS[a, i] + (I_om[j,k,c,a]-I_om[j,k,a,c])*CCIS[b,i] + (I_om[j,k,a,b] - I_om[j,k,b,a])*CCIS[c,i]\
        + I_om[k,i,b,c]*CCIS[a,j] + I_om[k,i,c,a]*CCIS[b,j] + (I_om[k,i,a,b] - I_om[k,i,b,a])*CCIS[c,j]\
        + (I_om[i,j,b,c]-I_om[i,j,c,b])*CCIS[a,k] + (I_om[i,j,c,a]-I_om[i,j,a,c])*CCIS[b,k] + I_om[i,j,a,b]*CCIS[c,k]
    return s


def calc_cismp2(p):
    omega = ECIS[p]
    print("Excitation: ", Excitations[p])
    Ed= 0
    for i in range(nocc):
        for j in range(i,nocc):
            for a in range(nocc, nso):
                for b in range(a, nso):
                    Ed += ( calc_u_d(i, j, a, b)**2 )/(-eps[i]-eps[j]+eps[a]+eps[b]-omega)
    Ed = -Ed
    Et = 0
    for i in range(nocc):
        for j in range(nocc):
            for k in range(nocc):
                for a in range(nocc, nso):
                    for b in range(nocc, nso):
                        for c in range(nocc, nso):
                            Et += (calc_u_t(i,j,k,a,b,c)**2)/(-eps[i]-eps[j]-eps[k]+eps[a]+eps[b]+eps[c]-omega)
    Et = -Et/36
    return Ed + Et
if(cis_mp2):   
    I = [0, 3, 7]
    f.write("------------------ CIS MP2 -------------------------\n")
    for i in I:    
        f.write("Etat {}: CIS(eV): {}, CIS-MP2 (eV): {}\n".format(i, ECIS[i]*ev, (ECIS[i]*ev+Ecismp*ev)))
