
import numpy as np
import matplotlib.pyplot as plt
import ase
from ase import io, visualize
import verify_energy_forces as vf
import compare_lammps_forces as cpr


def create_modified_cube(geometry_file_name,delta,d=10):
    # delta : displacement of top layer in cube structure
    # d : cube edge length
    cell_array = np.array([[3*d,0,0],
                            [0,3*d,0],
                            [0,0,3*d]])
    xyz = np.array([[d,   d,   d  ],
                    [2*d, d,   d  ],
                    [d,   2*d, d  ],
                    [2*d, 2*d, d  ],
                    [d,   d,   2*d],
                    [2*d, d,   2*d],
                    [d,   2*d, 2*d],
                    [2*d, 2*d, 2*d]],dtype=float)
    xyz[4:,:] += np.tile(delta,(4,1)) #displace top layer in cube structure

    atoms = ase.Atoms(cell=cell_array,
                      positions=xyz)
    atoms.symbols = ['B']*4 + ['Ti']*4 #set symbols
    atoms.set_masses = np.concatenate((12.01*np.ones(4),12.02*np.ones(4))) #set masses
    atoms.set_array('mol-id',np.array([1]*4 + [2]*4),dtype=np.int8) #set molecule IDs
    ase.io.write(geometry_file_name,atoms,format='lammps-data',atom_style='full')


def run_sweep(offsets,d,lammps_script,cur_constants,r_cut,skip_forces=False):
    # offsets: vector displacements for top layer, shape (num_offsets, 3)
    # d: cube edge length
    working_file_name = 'configurations/cube_working.lmp'

    # list of results dictionaries
    dict_list = [0]*offsets.shape[0]

    for i_d,delta_vec in enumerate(offsets):
        create_modified_cube(working_file_name,delta_vec,d=d)
        cube_dict = cpr.run_comparison(working_file_name,lammps_script,cur_constants,r_cut=r_cut,
                                       skip_forces=skip_forces)
        dict_list[i_d] = cube_dict
    return dict_list


if __name__ == "__main__":
    #                C0    C2     C4   delta C       A6    A8    A10
    cur_constants = [15.71,12.29,4.933,0.578,73.288,-0.257,0.397,0.639]
    d = 3.5 #cube edge length (should be near the cutoff set in lammps_script)
    r_cut = 4.3 #cutoff radius for all tests, [Angstroms]
    working_file_name = 'configurations/cube_working.lmp'

    # base script for LAMMPS run
    lammps_script="../../../../kc_insp_scripts/forces_rcut.simple"

    gridpoints = np.linspace(0,0.5,20)
    n_gridpoints = gridpoints.shape[0]

    # sweep in x direction
    offsets_x = np.zeros((n_gridpoints,3))
    offsets_x[:,0] = gridpoints
    dict_list_x = run_sweep(offsets_x,d,lammps_script,cur_constants,r_cut,skip_forces=True)
    E_python_x_vec = np.array([d["E_python"] for d in dict_list_x])/4 # divide by number of pairs
    E_lammps_x_vec = np.array([d["E_lammps"] for d in dict_list_x])/4
    print(f'mean relative error: {np.mean(np.abs((E_python_x_vec-E_lammps_x_vec)/E_python_x_vec))}')
    displacements_x = gridpoints  #displacement of x coordinate for two atoms in pair
    plt.plot(displacements_x,E_python_x_vec,color='black',linestyle='solid',label=r'Python')
    plt.plot(displacements_x,E_lammps_x_vec,color='orange',linestyle='dashed',label=r'LAMMPS')
    plt.xlabel(r'$x$ displacment, [A]')
    plt.ylabel(r'pair energy, [eV]')
    plt.legend()
    plt.show()

    # sweep in y direction
    offsets_y = np.zeros((n_gridpoints,3))
    offsets_y[:,1] = gridpoints
    dict_list_y = run_sweep(offsets_y,d,lammps_script,cur_constants,r_cut,skip_forces=True)
    E_python_y_vec = np.array([d["E_python"] for d in dict_list_y])/4 # divide by number of pairs
    E_lammps_y_vec = np.array([d["E_lammps"] for d in dict_list_y])/4
    print(f'mean relative error: {np.mean(np.abs((E_python_y_vec-E_lammps_y_vec)/E_python_y_vec))}')
    displacements_y = gridpoints  #displacement of y coordinate for two atoms in pair
    plt.plot(displacements_y,E_python_y_vec,color='black',linestyle='solid',label=r'Python')
    plt.plot(displacements_y,E_lammps_y_vec,color='orange',linestyle='dashed',label=r'LAMMPS')
    plt.xlabel(r'$y$ displacment, [A]')
    plt.ylabel(r'pair energy, [eV]')
    plt.legend()
    plt.show()

    # sweep in z direction
    offsets_z = np.zeros((n_gridpoints,3))
    offsets_z[:,2] = gridpoints
    dict_list_z = run_sweep(offsets_z,d,lammps_script,cur_constants,r_cut,skip_forces=True)
    E_python_z_vec = np.array([d["E_python"] for d in dict_list_z])/4 # divide by number of pairs
    E_lammps_z_vec = np.array([d["E_lammps"] for d in dict_list_z])/4
    print(f'mean relative error: {np.mean(np.abs((E_python_z_vec-E_lammps_z_vec)/E_python_z_vec))}')
    displacements_z = d*np.ones(n_gridpoints) + gridpoints  #displacement of z coordinate for two atoms in pair
    plt.plot(displacements_z,E_python_z_vec,color='black',linestyle='solid',label=r'Python')
    plt.plot(displacements_z,E_lammps_z_vec,color='orange',linestyle='dashed',label=r'LAMMPS')
    plt.xlabel(r'$z$ displacment, [A]')
    plt.ylabel(r'pair energy, [eV]')
    plt.legend()
    plt.show()


    """
    # run a single pair at 10 A
    r1 = np.array([0,0,0])
    n1 = np.array([0,0,1])
    r2 = np.array([0,0,d])
    n2 = np.array([0,0,-1])
    E_single_meV = vf.get_E_KC_inspired_pair(r1,r2,n1,n2,cur_constants)
    E_single = E_single_meV/1000 #[eV]
    print(E_single)
    """

