# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:23:48 2022

@author: danpa
"""

import os
import subprocess
import numpy as np
import ase
import matplotlib.pyplot as plt
import lammps_logfile
import flatgraphene as fg
import verify_energy_forces as vf

def rel_error(A_measured,A_true):
    #computes the relative error between two arrays,
    #even when one of them contains near zero values
    zero_threshold = 1e-14
    error_abs_val = np.abs(A_measured - A_true)

    #replaces all zeros with ones to prevent divide by zero errors
    safe_denominator = np.where((np.abs(A_true)<zero_threshold),1.0,np.abs(A_true))
    error_rel = error_abs_val/safe_denominator
    return error_rel


def get_lammps_setting(keyword, input_file):
    with open(input_file,"r") as f:
        lines=f.readlines()
        for l in lines:
            if keyword in l:
                setting=l.replace(keyword,"",1)
                setting=setting.strip()
                return setting


def get_lammps_energy_forces(data_file,input_file):
    """atoms_obj: (ase.atoms) atoms object to calculate forces of
    input_file: (str) path to lammps input file. must already run in its current directory
    """
    data_cmd=get_lammps_setting("read_data",input_file)
    data_fname=data_cmd.split(" ")[-1]
    dump_cmd=get_lammps_setting("dump",input_file)
    dump_cmd=dump_cmd.split(" ")
    dump_name=dump_cmd[4]
    dump_file_cmd="dump mydump2 all custom 1 "+dump_name+" id type x y z fx fy fz"
    
    lammps_dir=input_file.split("/")
    lammps_sim=lammps_dir[-1]
    lammps_dir="/".join(lammps_dir[:-1])
    
    cwd=os.getcwd()
    
    subprocess.call("cp "+data_file+" "+lammps_dir+"/"+data_fname,shell=True)
    #subprocess.call("cd "+lammps_dir,shell=True)
    os.chdir(lammps_dir)
    # print("line=`grep -w \"read_data\" " +lammps_sim+"`")
    # subprocess.call("line= `grep -w read_data " +lammps_sim+"`",shell=True)
    # print("sed -i s+$line+\"read_data "+data_fname+"\"+g "+lammps_sim)
    # subprocess.call("sed -i s+$line+\"read_data "+data_fname+"\"+g "+lammps_sim,shell=True)
    # subprocess.call("line= `grep -w \"dump\" " +lammps_sim+"`",shell=True)
    # subprocess.call("sed -i s+$line+"+dump_file_cmd+"+g "+lammps_sim,shell=True)
    
    #subprocess.call("~/lammps/src/lmp_serial<"+lammps_sim,shell=True)
    subprocess.call("${LMP_SERIAL}<"+lammps_sim,shell=True)
    log = lammps_logfile.File("log.lammps") # assumes logfile name is log.lammps
    energy = log.get('v_Evdw') #this grabs only the interlayer KC-inspired correction
    atoms=ase.io.read(dump_name,format="lammps-dump-text")
    os.remove(dump_name)
    os.remove(data_fname)
    os.chdir(cwd)
    return energy, atoms.get_forces()


def run_comparison(geometry_file,lammps_script,cur_constants,r_cut=10,skip_forces=False):
    energy_meV, force_fd_meV = vf.get_energy_forces(geometry_file,cur_constants,r_cut=r_cut,
                                                    skip_forces=skip_forces)

    out_dict = {}
    if (energy_meV is not None):
        energy = energy_meV/1000 
        energy_lammps, force_lammps = get_lammps_energy_forces(geometry_file,lammps_script)
        energy_lammps = energy_lammps[0] #energy lammps is a single element list
        error_rel_energy = np.abs((energy - energy_lammps)/energy)
        E_dict = {
            "E_python" : energy, #[eV]
            "E_lammps" : energy_lammps, #[eV]
            "E_err_rel" : error_rel_energy,
            }
        out_dict.update(E_dict)

    if (force_fd_meV is not None):
        force_fd = force_fd_meV/1000
        error_rel_force = rel_error(force_lammps,force_fd)
        error_rel_ave_force = np.mean(error_rel_force)
        F_dict = {
            "F_python" : force_fd, #[eV/A]
            "F_lammps" : force_lammps, #[eV/A]
            "F_err_rel" : error_rel_force,
            "F_err_rel_ave" : error_rel_ave_force,
            }
        out_dict.update(F_dict)
    return out_dict


if __name__=="__main__":
    ##########################################################################
    
    # Run from linux command line
    
    ##########################################################################
    
    # parameters of KC-inspired interlayer corrective potential
    cur_constants = [15.71,12.29,4.933,0.578,73.288,-0.257,0.397,0.639]
    r_cut = 4 #cutoff radius for all tests, [Angstroms]

    # forces from Lammps implementation
    lammps_script="../../../../kc_insp_scripts/forces.simple"

    #procedure to get normals can fail (returns NaNs), because colinear vectors may be crossed with themselves
    # returning the zero vector
    #to be clear, this is only a danger for oversimiplified systems, not for any actual graphene configurations
    #therefore, this array supplies the normals by hand
    two_atom_normals = np.array([[0,0,1],
                                 [0,0,-1]])

    vf.create_geometries()

    # RUN COMPARISONS AND STORES RESULTS
    # twisted system
    twist_dict = run_comparison('configurations/twist_21_79.data',lammps_script,
                                              cur_constants,r_cut=r_cut)

    # rectangular A, 32 atoms total
    rect_dict = run_comparison('configurations/AA_rect_32.lmp',lammps_script,cur_constants,r_cut=r_cut)

    # PRINT OUT RESULTS AFTER ALL RUNS (SINCE LAMMPS OUTPUT MAKES IT HARD TO READ)
    print('')
    print(f'TWISTED:')
    print(f'  energy (Python)               : {twist_dict["E_python"]}')
    print(f'  energy (LAMMPS)               : {twist_dict["E_lammps"]}')
    print(f'  average relative energy error : {twist_dict["E_err_rel"]}')
    print(f'  average relative force error  : {twist_dict["F_err_rel_ave"]}')
    print(f'RECTANGULAR:')
    print(f'  energy (Python)               : {rect_dict["E_python"]}')
    print(f'  energy (LAMMPS)               : {rect_dict["E_lammps"]}')
    print(f'  average relative energy error : {rect_dict["E_err_rel"]}')
    print(f'  average relative force error  : {rect_dict["F_err_rel_ave"]}')
    print('')
    
    """
    force_fake_lammps=get_lammps_forces('configurations/fake_rect_2.lmp', lammps_script)
    
    force_rect_lammps=get_lammps_forces('configurations/AA_rect_8.lmp', lammps_script)
    
    force_hex_lammps=get_lammps_forces('configurations/AA_hex_4.lmp', lammps_script)
    
    
    force_ab_lammps=get_lammps_forces('configurations/AB_rect_8.lmp',lammps_script)
    """

