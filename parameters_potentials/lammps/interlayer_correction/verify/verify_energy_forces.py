
import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import ase
from ase import io
from ase import visualize

import flatgraphene as fg
from latte_dat_io import write_lammps
#just let this be a monolithic file!

def getNormal(r,xyzperiodic, tol, midplane):    
    """
    computes the normal vector of a single atom given
    r : coordinate of atom
    xyzperiodic : coordinates of all atoms in 9 periodic cells
    """

    rarray = np.tile(r, (xyzperiodic.shape[0],1)) #creating array of "r" coordinates only
    dis1 = xyzperiodic - rarray  #%distance of atom "r" from each atom, coordinates. vector convention end to start
    dis = dis1[abs(dis1[:,2])<=tol,:]#%tol is for corrugation. To make sure atom has in plane neighbors
    disnorm = la.norm(dis,ord=2,axis = 1)   #normalized the coordinates to get a scalar distance

    #taking 4 atoms including r and taking biggest 3
    idx = np.argpartition(disnorm, 4)#find indices from low to high. 
    dist = (disnorm[idx[:4]])#get elements from index, but they are not sorted
    ind = idx[np.argsort(dist)[1:]]#indices of the neareset neighbors, coordinates of idx(inplane nieghbors)


    inplane = dis[ind,:] #gets the distance vector of the inplane nearest neighbors
    prod1 = np.cross(inplane[0,:], inplane[1,:])
    prod2 = np.cross(inplane[1,:], inplane[2,:])
    prod3 = np.cross(inplane[2,:], inplane[0,:])
    
    #normalizing the products
    prod1 = np.sign(prod1[2])*prod1/la.norm(prod1) #%making sure all three normals are positive
    prod2 = np.sign(prod2[2])*prod2/la.norm(prod2)
    prod3 = np.sign(prod3[2])*prod3/la.norm(prod3)
    
    normal =(prod1+prod2+prod3)/3
    
    if r[2] > midplane: #%if atom "r" is in upper layer of graphene, normal is downward
        normal= -normal

    return normal


def get_E_KC_inspired_pair(ri, rj, ni, nj, curConstants):
    #computes the pairwise interlayer energy for a single pair of atoms

    #C0 = curConstants[0]
    #C2 = curConstants[1]
    #C4 = curConstants[2]
   
    #Kolmogorov Crespi paper parameters
    #C = 3.03
    #delta = 0.578
    #lamda = 3.629
    #A = 10.238#curConstants[2]#
    #C0 = 15.71
    #C2 =12.29
    #C4 = 4.933

    # order of curConstants:  C0    C2     C4   delta C       A6    A8    A10
    z0    = 3.34 #AB bilayer experimentnal separation in Angstrom
    C0    = curConstants[0]
    C2    = curConstants[1]
    C4    = curConstants[2]
    delta = curConstants[3]
    C     = curConstants[4]
    A6    = curConstants[5]
    A8    = curConstants[6]
    A10   = curConstants[7]
    
    rij = la.norm(ri-rj)
    rijvec = rj-ri
 
    rhoij =np.sqrt(rij**2-(np.dot(ni,rijvec))**2)
    rhoji =np.sqrt(rij**2-(np.dot(nj,rijvec))**2)
    
    frhoij = (np.exp(-np.power(rhoij/delta,2)))*(C0*np.power(rhoij/delta,0)+C2*np.power(rhoij/delta,2)+ C4*np.power(rhoij/delta,4))
    frhoji = (np.exp(-np.power(rhoji/delta,2)))*(C0*np.power(rhoji/delta,0)+C2*np.power(rhoji/delta,2)+ C4*np.power(rhoji/delta,4))
    
<<<<<<< Updated upstream
    V = -(C+frhoij+frhoji)*(A6*np.power(rij/z0,-6)+A8*np.power(rij/z0,-8)+A10*np.power(rij/z0,-10))
=======
    v_reg = (C+frhoij+frhoji)
    V=-(C+frhoij+frhoji)*(A6*np.power(rij/z0,-6)+A8*np.power(rij/z0,-8)+A10*np.power(rij/z0,-10))
>>>>>>> Stashed changes
    #V=-(A6*np.power(rij/z0,-6)+A8*np.power(rij/z0,-8)+A10*np.power(rij/z0,-10))
    #V=(np.exp((-lamda)*(rij-z0)))*(C+frhoij+frhoji)#KC paper parameter
    #V=(np.exp((-lamda)*(rij-z0)))#KC paper parameter
    #V=-(np.exp((-lamda)*(rij-z0)))*(C+frhoij+frhoji)-A*np.power(rij/z0,-6)
    #V=-(C+frhoij+frhoji)*A*np.power(rij/z0,-6)
    #V=-(C-frhoij-frhoji)*(A6*np.power(rij/z0,-6)+A8*np.power(rij/z0,-8)+A10*np.power(rij/z0,-10))

    """
    # prints for debugging 
<<<<<<< Updated upstream
    print(f'PYTHON')
    print(f'delta : {delta}')
    print(f'delta2inv : {np.power(delta,-2.0)}')
    print(f'C0 : {C0}')
    print(f'C2 : {C2}')
    print(f'C4 : {C4}')
    print(f'C : {C}')
    print(f'rho_ijsq : {np.power(rhoij,2)}')
    print(f'frho_ij : {frhoij}')
    print(f'C + frhoij + frhoji : {C+frhoij+frhoji}')
    print(f'inverse 6-8-10 : {(A6*np.power(rij/z0,-6)+A8*np.power(rij/z0,-8)+A10*np.power(rij/z0,-10))/1000}') #divided by 1000 to convert to eV
    print(f'V_pair : {V}')
    print(f'\n')
    """
=======
    # print(f'PYTHON')
    # print(f'delta : {delta}')
    # print(f'delta2inv : {np.power(delta,-2.0)}')
    # print(f'C0 : {C0}')
    # print(f'C2 : {C2}')
    # print(f'C4 : {C4}')
>>>>>>> Stashed changes
  
    return V,v_reg # [meV]


def get_E_KC_inspired_total(xyz, periodicR1, periodicR2, curConstants, tol, r_cut, normal=None):
    #computes the total pairwise interlayer energy of an atomic configuration
    
    natoms = xyz.shape[0]
    #only works when one of the bilayer is at zero (fails spectacularly otherwise)
    #sep = 2*np.mean(np.abs(xyz[:,2])) #%average of the interlayer separation which is half of sep. So multiplied by 2
    #never fails spectacularly, but not as accurate as average when one layer at zero
    sep = np.max(xyz[:,2]) - np.min(xyz[:,2])

    if (normal is None):
        xyzperiodic = np.zeros((xyz.shape[0]*9,3)) #%creating 9 boxes of atoms
        indices = [0, -1, 1] #%to force the first atom at 000
        for   periodicI in range (0,3,1):
            i2 = indices[periodicI]
            for periodicJ in range (0,3,1):
                j2 = indices[periodicJ];
                for i in range (0,xyz.shape[0],1): #%goint to each atom one by one
                    index = natoms*(periodicI)+natoms*3*(periodicJ)+i #%index of all the 9 boxes
                    xyzperiodic[index,:] = xyz[i,:] + i2*periodicR1 + j2*periodicR2#%will not append them to make 9 blocks   


        normal = np.zeros((xyz.shape[0],3)) #%normal is a vector
        for i in range(xyz.shape[0]):
            normal[i,:]=getNormal(xyzperiodic[i,:], xyzperiodic, tol, sep/2)
 
    #%calculation of pairwise potential
    Vrepel = 0;
    Vregistry= 0
    pairwiseEnergy = np.zeros((xyz.shape[0]))
    
    for periodicI in [-1, 0, 1]:
        for periodicJ in [-1, 0, 1]:    

            for i in range (0,xyz.shape[0], 1): #%going to each atom one by one
                ri = xyz[i,:]      #%+ periodicI*periodicR1' + periodicJ*periodicR2';
                normi = normal[i,:] 
                
                for j in range (i+1, xyz.shape[0], 1): #% avoiding 'i' to avoid self interaction
                    rj = xyz[j,:] +periodicI*periodicR1 + periodicJ*periodicR2  #; %it is searching within xyz
                    dist = la.norm(rj - ri)
                    vertical = np.abs(rj[2]-ri[2]) 
                    normj = normal[j, :]
                        
                    #%INTERLAYER INTERACTION, NOT APPLYING R_CUT, RIGIDLY CHOOSING NEIGHBOR
                    if dist < r_cut and vertical > tol: #only interlayer pairwise energy
                         Vij , v_reg = get_E_KC_inspired_pair(ri, rj,normi, normj,  curConstants)
                         Vrepel += Vij #got rid of double counting (BAD VARIABLE NAME)
                         Vregistry += v_reg
    return Vrepel, Vregistry


def get_energy_forces(file_name,curConstants,atoms=None,normal=None,r_cut=10,fd_delta=1e-6,
                      skip_forces=False):
    #gets the energy and forces of atomic configuration given in lammps file at file_name
    # or in the ASE atoms object atoms (using finite differences for force)
    if (atoms is None): #only read atoms from file name if not provided by option
        atoms = ase.io.read(file_name,format='lammps-data')
        cell=atoms.get_cell()
        cell[1,0]*=-1
        atoms.set_cell(cell)
        
    xyz = atoms.get_positions()
    n_atoms = xyz.shape[0]
    box_vectors = atoms.get_cell()
    periodicR1 = box_vectors[0]
    periodicR2 = box_vectors[1]

    #hardcoded variables
    tol = 0.26 #corrugation amount (BAD VARIABLE NAME)

    #check that r_cut will not reach beyond periodic images
    #compute portion of second lattice vector which is perpendicular to first
    periodicR2_perp = periodicR2 - np.dot(periodicR1,periodicR2)/(np.dot(periodicR1,periodicR1))*periodicR1
    if ((r_cut >= np.linalg.norm(periodicR1)) or
        (r_cut >= np.linalg.norm(periodicR2_perp))):
        print('WARNING: YOUR SELECTED CUTOFF RADIUS IS TOO')
        print('LARGE FOR THE SYSTEM ... NOT RUNNING')
        print(f'periodicR1: {periodicR1}')
        print(f'periodicR2: {periodicR2}')
        print(f'periodicR2_perp: {periodicR2_perp}\n')
        return (None, None)

    #compute energy
    E = get_E_KC_inspired_total(xyz, periodicR1, periodicR2, curConstants, tol, r_cut, normal=normal)

    #compute forces (using finite difference)
    if (skip_forces):
        f_array = None
    else:
        xyz_flat = xyz.ravel()
        f_flat = np.zeros(xyz_flat.shape) #flattened array for forces
        for i_q, q in enumerate(xyz_flat):
            #backwards step
            xyz_bck_flat = xyz_flat.copy()
            xyz_bck_flat[i_q] -= fd_delta
            xyz_bck = np.reshape(xyz_bck_flat,[n_atoms,3])
            E_bck = get_E_KC_inspired_total(xyz_bck,periodicR1,periodicR2,curConstants,
                                            tol,r_cut,normal=normal)

            #forward step
            xyz_fwd_flat = xyz_flat.copy()
            xyz_fwd_flat[i_q] += fd_delta
            xyz_fwd = np.reshape(xyz_fwd_flat,[n_atoms,3])
            E_fwd = get_E_KC_inspired_total(xyz_fwd,periodicR1,periodicR2,curConstants,
                                            tol,r_cut,normal=normal)

            #compute force using centered difference formula
            f_flat[i_q] = -(E_fwd - E_bck)/(2*fd_delta)

        #reshape forces into array where f_array[i,j] is jth compoent of force on atom i
        f_array = np.reshape(f_flat,[n_atoms,3])

    return E, f_array


def check_convergence(file_name,curConstants,atoms=None,normal=None,r_cut=10):
    #checks convergence of the force-computing finite difference scheme
    #    on an input geometry
    fd_delta_vec = np.logspace(-1,-12,12)
    force_array_list = [0]*fd_delta_vec.shape[0]
    print(f'running convergence check')
    for i_d, fd_delta in enumerate(fd_delta_vec):
        print(f'  running delta = {fd_delta}')
        E, forces = get_energy_forces(file_name,curConstants,atoms=atoms,normal=normal,r_cut=r_cut,fd_delta=fd_delta)
        force_array_list[i_d] = forces
    f_z_abs_average_vec = np.array([np.mean(np.abs(force_array[:,2])) for force_array in force_array_list])
    #compute (f_{i+1}-f_i)/f_{i+1}
    f_percent_change_vec = np.abs(f_z_abs_average_vec[1:] - f_z_abs_average_vec[:-1])/f_z_abs_average_vec[1:]

    #atom_number = #atom number whose forces' convergence will be tracked
    plt.loglog(fd_delta_vec[1:],f_percent_change_vec,'k')
    plt.xlabel(r'finite difference step size, $\Delta$')
    plt.ylabel(r'percent change of absolute z force, $\frac{|f_{i+1} - f_{i}|}{f_{i+1}}$')
    plt.show()


def create_geometries():
    #create test configurations

    #fg.help()

    #rectangular AA, 8 total atoms
    atoms = fg.shift.make_graphene(stacking=['A','A'],cell_type='rect',n_1=1,n_2=1,lat_con=0,
                                   n_layer=2,sep=3.5,sym=["B",'Ti'],mass=[12.01,12.02],a_nn=1.43,h_vac=3)
    #ase.io.write('configurations/AA_rect_8.lmp',atoms,format='lammps-data',atom_style='full')
    write_lammps('configurations/AA_rect_8.lmp',atoms)
    
    #hexagonal AA, 4 total atoms
    atoms = fg.shift.make_graphene(stacking=['A','A'],cell_type='hex',n_1=1,n_2=1,lat_con=0,
                                   n_layer=2,sep=3.5,sym=["B",'Ti'],a_nn=1.43,h_vac=3)
    ase.io.write('configurations/AA_hex_4.lmp',atoms,format='lammps-data',atom_style='full')

    #rectangular AA, 32 total atoms
    atoms = fg.shift.make_graphene(stacking=['A','A'],cell_type='rect',n_1=2,n_2=2,lat_con=0,
                                   n_layer=2,sep=3.5,sym=["B",'Ti'],a_nn=1.43,h_vac=3)
    ase.io.write('configurations/AA_rect_32.lmp',atoms,format='lammps-data',atom_style='full')

    #rectangular AA, 36 atoms total
    atoms = fg.shift.make_graphene(stacking=['A','A'],cell_type='hex',n_1=3,n_2=3,lat_con=0,
                                       n_layer=2,sep=3.5,sym=["B",'Ti'],a_nn=1.43,h_vac=3)
    ase.io.write('configurations/AA_hex_36.lmp',atoms,format='lammps-data',atom_style='full')


def run_convergence_check():
    #check convergence on system
    atoms_hex_large = fg.shift.make_graphene(stacking=['A','A'],cell_type='hex',n_1=3,n_2=3,lat_con=0,
                                             n_layer=2,sep=3.5,sym=["B",'Ti'],mass=[12.01,12.02],a_nn=1.43,h_vac=3)
    check_convergence('does not matter since overwritten by atoms',cur_constants,atoms=atoms_hex_large,r_cut=4)


if __name__ == "__main__":
    #                C0    C2     C4   delta C       A6    A8    A10
    cur_constants = [15.71,12.29,4.933,0.578,73.288,-0.257,0.397,0.639]

    r_cut = 4 #cutoff radius for all tests, [Angstroms]

    #procedure to get normals can fail (returns NaNs), because colinear vectors may be crossed with themselves
    # returning the zero vector
    #to be clear, this is only a danger for oversimiplified systems, not for any actual graphene configurations
    #therefore, this array supplies the normals by hand
    two_atom_normals = np.array([[0,0,1],
                                 [0,0,-1]])

    #create lammps data files in configurations/
    create_geometries()

    #CONVERGENCE CHECK
    #run_convergence_check()

    #SMALL SYSTEM TESTS
    #rectangular cell, 2 atoms total (stacked on top of each other)
    E_fake, forces_fake = get_energy_forces('configurations/fake_rect_2.lmp',cur_constants,normal=two_atom_normals,r_cut=r_cut)
    if (not (E_fake is None)):
        print(f'2 atoms (stacked directly atop one another): \nenergy per atom: {E_fake/2}\nforces:{forces_fake}\n')

    #rectangular AA, 8 total atoms
    E_rect, forces_rect = get_energy_forces('configurations/AA_rect_8.lmp',cur_constants,r_cut=r_cut)
    if (not (E_rect is None)):
        print(f'rectangular AA: \nenergy per atom: {E_rect/8}\nforces:{forces_rect}\n')

    #hexagonal AA, 4 total atoms (using ASE atoms object input)
    atoms_hex = fg.shift.make_graphene(stacking=['A','A'],cell_type='hex',n_1=1,n_2=1,lat_con=0,
                                       n_layer=2,sep=3.5,a_nn=1.43,h_vac=3)
    E_hex, forces_hex = get_energy_forces('does not matter since overwritten by atoms',cur_constants,atoms=atoms_hex,r_cut=r_cut)
    if (not (E_hex is None)):
        print(f'hexagonal AA (ASE object): \nenergy per atom: {E_hex/4}\nforces:{forces_hex}\n')

    #hexagonal AA, 4 total atoms (using LAMMPS data file, which causes change to cell vectors)
    E_hex, forces_hex = get_energy_forces('configurations/AA_hex_4.lmp',cur_constants,r_cut=r_cut)
    if (not (E_hex is None)):
        print(f'hexagonal AA (LAMMPS): \nenergy per atom: {E_hex/4}\nforces:{forces_hex}\n')

    #LARGER SYSTEM CONVERGENCE TESTS
    #rectangular AA, 32 atoms total
    E_rect_large, forces_rect_large = get_energy_forces('configurations/AA_rect_32.lmp',cur_constants,r_cut=r_cut)
    if (not (E_rect_large is None)):
        print(f'rectangular AA (large): \nenergy per atom: {E_rect_large/32}\nforces:{forces_rect_large}\n')

    #hexagonal AA, 36 atoms total
    atoms_hex_large = fg.shift.make_graphene(stacking=['A','A'],cell_type='hex',n_1=3,n_2=3,lat_con=0,
                                       n_layer=2,sep=3.5,a_nn=1.43,h_vac=3)
    E_hex_large, forces_hex_large = get_energy_forces('does not matter since overwritten by atoms',cur_constants,atoms=atoms_hex_large,r_cut=r_cut)
    if (not (E_hex_large is None)):
        print(f'hexagonal AA (large): \nenergy per atom: {E_hex_large/36}\nforces:{forces_hex_large}\n')


    # DAN'S CELL VECTOR TEST
    #hexagonal AA, 4 total atoms (using LAMMPS data file, which causes change to cell vectors)
    energy_hex, forces_hex = get_energy_forces('configurations/AA_hex_4.lmp',cur_constants)
    temp_atom=ase.io.read('configurations/AA_hex_4.lmp',format='lammps-data')
    cell=temp_atom.get_cell()
    cell[1,0]*=-1
    temp_atom.set_cell(cell)
    print('hexagonal AA (LAMMPS data file, PROBABLY WRONG): \n',forces_hex,'\n')
