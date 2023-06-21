
import numpy as np
import matplotlib.pyplot as plt
import ase
from ase import io, visualize
import verify_energy_forces as vf
import compare_lammps_forces as cpr
import pair_energy_verify as pe

if __name__ == "__main__":
    #                C0    C2     C4   delta C       A6    A8    A10
    cur_constants = [15.71,12.29,4.933,0.578,73.288,-0.257,0.397,0.639]
    d = 3.5 #cube edge length (should be near the cutoff set in lammps_script)
    r_cut = 4.3 #cutoff radius for all tests, [Angstroms]
    working_file_name = 'configurations/cube_working.lmp'

    # base script for LAMMPS run
    lammps_script="../../../../kc_insp_scripts/forces_rcut_4_3.simple"

    gridpoints = np.linspace(0,0.5,50)
    n_gridpoints = gridpoints.shape[0]

    # sweep in x direction
    offsets_x = np.zeros((n_gridpoints,3))
    offsets_x[:,0] = gridpoints
    displacements_x = gridpoints  #displacement of x coordinate for two atoms in pair
    dict_list_x = pe.run_sweep(offsets_x,d,lammps_script,cur_constants,r_cut,skip_forces=False)
    E_x_vec = np.array([d["E_python"] for d in dict_list_x])/4 #divide by number of pairs
    F_post_fd_x_vec = -np.gradient(E_x_vec,displacements_x)
    F_pre_fd_x_vec = np.array([d["F_python"][-1,0] for d in dict_list_x]) # grab from upper layer atom
    plt.plot(displacements_x,F_post_fd_x_vec,color='black',linestyle='solid',label=r'post FD')
    plt.plot(displacements_x,F_pre_fd_x_vec,color='orange',linestyle='dashed',label=r'pre FD')
    plt.xlabel(r'$x$ displacment, [A]')
    plt.ylabel(r'force, [eV A$^{-1}$]')
    plt.legend()
    plt.show()

    # sweep in y direction
    offsets_y = np.zeros((n_gridpoints,3))
    offsets_y[:,1] = gridpoints
    displacements_y = gridpoints  #displacement of y coordinate for two atoms in pair
    dict_list_y = pe.run_sweep(offsets_y,d,lammps_script,cur_constants,r_cut,skip_forces=False)
    E_y_vec = np.array([d["E_python"] for d in dict_list_y])/4 #divide by number of pairs
    F_post_fd_y_vec = -np.gradient(E_y_vec,displacements_y)
    F_pre_fd_y_vec = np.array([d["F_python"][-1,1] for d in dict_list_y]) # grab from upper layer atom
    plt.plot(displacements_y,F_post_fd_y_vec,color='black',linestyle='solid',label=r'post FD')
    plt.plot(displacements_y,F_pre_fd_y_vec,color='orange',linestyle='dashed',label=r'pre FD')
    plt.xlabel(r'$y$ displacment, [A]')
    plt.ylabel(r'force, [eV A$^{-1}$]')
    plt.legend()
    plt.show()

    # sweep in z direction
    offsets_z = np.zeros((n_gridpoints,3))
    offsets_z[:,2] = gridpoints
    displacements_z = d*np.ones(n_gridpoints) + gridpoints  #displacement of z coordinate for two atoms in pair
    dict_list_z = pe.run_sweep(offsets_z,d,lammps_script,cur_constants,r_cut,skip_forces=False)
    E_z_vec = np.array([d["E_python"] for d in dict_list_z])/4 #divide by number of pairs
    F_post_fd_z_vec = -np.gradient(E_z_vec,displacements_z)
    F_pre_fd_z_vec = np.array([d["F_python"][-1,2] for d in dict_list_z]) # grab from upper layer atom
    plt.plot(displacements_z,F_post_fd_z_vec,color='black',linestyle='solid',label=r'post FD')
    plt.plot(displacements_z,F_pre_fd_z_vec,color='orange',linestyle='dashed',label=r'pre FD')
    plt.xlabel(r'$z$ displacment, [A]')
    plt.ylabel(r'force, [eV A$^{-1}$]')
    plt.legend()
    plt.show()



