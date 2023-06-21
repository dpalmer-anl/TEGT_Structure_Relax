
import numpy as np
import matplotlib.pyplot as plt
import ase
from ase import io, visualize
import flatgraphene as fg
import verify_energy_forces as vf
import compare_lammps_forces as cpr


if __name__ == "__main__":
    #                C0    C2     C4   delta C       A6    A8    A10
    cur_constants = [15.71,12.29,4.933,0.578,73.288,-0.257,0.397,0.639]
    r_cut = 10.0 #cutoff radius for all tests, [Angstroms]

    lammps_script="../../../../kc_insp_scripts/forces_rcut_10.simple" # base script
    working_file_name = 'configurations/working_pbc.lmp'


    rep_max_vec = np.arange(5,9)

    n_atoms_vec  = np.empty(rep_max_vec.shape[0])
    E_python_vec = np.empty(rep_max_vec.shape[0])
    E_lammps_vec = np.empty(rep_max_vec.shape[0])


    for i_r, rep_max in enumerate(rep_max_vec):
        #create geometry and write to file
        atoms = fg.shift.make_graphene(stacking=['A','A'],cell_type='rect',n_1=rep_max,n_2=rep_max,
                                       lat_con=0,n_layer=2,sep=3.5,sym=['B','Ti'],mass=[12.01,12.02],
                                       a_nn=1.43,h_vac=12)
        ase.io.write(working_file_name,atoms,format='lammps-data',atom_style='full')
        #get number of atoms
        n_atoms_vec[i_r] = atoms.get_positions().shape[0]
        #run comparison
        comp_dict = cpr.run_comparison(working_file_name,lammps_script,cur_constants,r_cut=r_cut,
                                       skip_forces=True)
        E_python_vec[i_r] = comp_dict["E_python"]
        E_lammps_vec[i_r] = comp_dict["E_lammps"]


    relative_error_vec = np.abs((E_python_vec-E_lammps_vec)/E_python_vec)

    print(f'E_python : {E_python_vec}')
    print(f'E_lammps : {E_lammps_vec}')
    print(f'relative error: {relative_error_vec}')

    # skip this plot since they are so similar
    """
    #energy versus number of atoms
    plt.plot(n_atoms_vec,E_python_vec/n_atoms_vec,color='black',label='Python')
    plt.plot(n_atoms_vec,E_lammps_vec/n_atoms_vec,color='tab:orange',linestyle='dashed',label='LAMMPS')
    plt.xlabel(r'number of atoms')
    plt.ylabel(r'energy per atom, $E_N$ [eV/atom]')
    plt.legend()
    plt.show()
    """

    #relative error versus number of atoms
    plt.semilogy(n_atoms_vec,relative_error_vec,'k')
    plt.xlabel(r'number of atoms')
    plt.ylabel(r'relative error')
    plt.show()

