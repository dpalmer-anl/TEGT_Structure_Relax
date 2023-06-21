# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:27:22 2022

@author: danpa
"""

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

import ase
from ase import io
from ase import visualize

import flatgraphene as fg
from verify_energy_forces import get_E_KC_inspired_total
import pandas as pd

def rel_error(A_measured,A_true):
    #computes the relative error between two arrays,
    #even when one of them contains near zero values
    zero_threshold = 1e-14
    error_abs_val = np.abs(A_measured - A_true)

    #replaces all zeros with ones to prevent divide by zero errors
    safe_denominator = np.where((np.abs(A_true)<zero_threshold),1.0,np.abs(A_true))
    error_rel = error_abs_val/safe_denominator
    return error_rel
if __name__=="__main__":
    
    #files=["metadata_AA.txt","metadata_AB.txt"]
    files=["metadata_AA_no_offset.txt","metadata_AB_no_offset.txt"]
    labels=["AA","AB"]
    
    curConstants=[15.71,12.29,4.933,0.578,73.288,-0.257,0.397,0.639]
    for i,f in enumerate(files):
        data =  pd.read_csv(f, sep=" ",header=0)
        sep_=data["layer_sep"][:-1]
        natom=data["num_atoms"][0]
        e_kc_lammps=np.array(data["v_Evdw"]/natom*1000)[:-1] #meV/atom
        e_kc_lammps_reg=np.array(data["v_Ereg"])[:-1] 
        n_cell=natom/2/4
        a=2.529
        e_kc_python=np.zeros_like(sep_)
        e_reg_python=np.zeros_like(sep_)
        for j,s in enumerate(sep_):
            stacking=[labels[i][0],labels[i][1]]
            atoms_obj=fg.shift.make_graphene(stacking=stacking,cell_type='rect',
                                n_layer=2,n_1=5,n_2=5,lat_con=a,
                                sep=s,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)
            xyz=atoms_obj.positions
            cell=atoms_obj.get_cell()
            periodicR1=cell[:,0]
            periodicR2=cell[:,1]
            tol=0.26
            r_cut=18
            e_kc_python[j],e_reg_python[j]=get_E_KC_inspired_total(xyz, periodicR1, periodicR2, curConstants, tol, r_cut, normal=None)
        e_kc_python/= natom
        #e_kc_python*=1000
        
        plt.plot(sep_,e_kc_python,label="python energy pair "+labels[i])
        plt.plot(sep_,e_kc_lammps,label="lammps energy pair "+labels[i])
        plt.legend()
        plt.title("no offset KC inspired potential energy for "+labels[i])
        plt.legend()
        plt.show()
        
        error=e_kc_lammps-e_kc_python #rel_error(e_kc_lammps,e_kc_python)
        plt.plot(sep_,error,label="error energy "+labels[i])
        plt.legend()
        plt.title("no offset difference between python and lammps ")
        plt.show()    
        
        # plt.plot(sep_,e_reg_python,label="python energy registry "+labels[i])
        # plt.plot(sep_,e_kc_lammps_reg,label="lammps energy registry "+labels[i])
        # plt.title("registry dependent term in lammps")
        # plt.legend()
        # plt.show() 