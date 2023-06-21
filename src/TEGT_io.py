# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:23:54 2023

@author: danpa
"""
import numpy as np 
import os
import re
from ase import Atoms
import ase.io

def write_latte_dat(ase_obj,filename,electron_file=None):
    """write latte coordinate file from ase.atom.Atom object
    
    :param ase_obj: (ase.atom.Atom obj) ase atoms object where geometry is stored
    
    :param filename: (str) filename to write data file to
    """
    cell=np.array(ase_obj.get_cell())
    rx_=" ".join(map(str,cell[0,:]))
    ry_=" ".join(map(str,cell[1,:]))
    rz_=" ".join(map(str,cell[2,:]))
    
    xyz=ase_obj.get_positions()
    natom=np.shape(xyz)[0]
    
    #compare mass in ase object to mass in electrons file to find correct element
    if electron_file!=None:
        with open(electron_file) as f:
            lines=f.readlines()
            mass_dict={}
            for i,l in enumerate(lines):
                if i<2:
                    continue
                properties=l.split(" ")
                temp_dict={float(properties[7]) : properties[0]} # {Mass : Element}
                mass_dict.update(temp_dict)
                
    with open(filename,'w+') as f:
        f.write("      ")
        f.write('%d\n'%(natom))
        f.write(rx_+" \n")
        f.write(ry_+" \n")
        f.write(rz_+" \n")
        
        for a in ase_obj:
            if electron_file!=None:
                get_mass=a.mass
                for key in mass_dict.keys():
                    if np.isclose(float(key),get_mass,rtol=0.00001):
                        symbol=mass_dict[key]
            else:
                symbol=a.symbol
                
            f.write(symbol+" ")
            pos=np.array(a.position)
            str_pos=" ".join(map(str,pos))
            f.write(str_pos+" \n")
            
def read_latte_dat(filename,electron_file=None):
    """read latte data file into ase.Atoms object
    
    :param filename: (str) filename of latte data file to read
    
    :returns: (ase.Atoms) ase.Atoms object containing chemical symbols, positions
              and cell of system"""
              
    with open(filename,"r") as f:
        lines=f.readlines()
        
        natom=int(re.findall(r'[-+]?[.]?[:\.\d]+',lines[0])[0])
        pos=np.zeros((natom,3))
        symbols=np.empty(natom,dtype=np.unicode_)
        cell_x=re.findall(r'[-+]?[.]?[:\.\d]+',lines[1])
        cell_y=re.findall(r'[-+]?[.]?[:\.\d]+',lines[2])
        cell_z=re.findall(r'[-+]?[.]?[:\.\d]+',lines[3])
        
        for i,l in enumerate(lines[4:]):
            pos[i,:]=re.findall(r'[-+]?[.]?[:\.\d]+',l)
            sym=l.split(" ")[0]
            symbols[i]=sym
            
        #include masses in object
        if electron_file!=None:
            with open(electron_file) as f:
                lines=f.readlines()
                mass_dict={}
                for i,l in enumerate(lines):
                    if i<2:
                        continue
                    properties=l.split(" ")
                    temp_dict={properties[0] : float(properties[7])} # {Mass : Element}
                    mass_dict.update(temp_dict)
                    
            masses=np.zeros(natom)
            for k in mass_dict.keys():
                ind=np.where(symbols==k)
                masses[ind]=mass_dict[k]
            
            atom_obj=Atoms(positions=pos,\
                       cell=np.array([cell_x,cell_y,cell_z]))
            atom_obj.set_masses(masses)
        else:
            try:
                atom_obj=Atoms(symbols,positions=pos,\
                           cell=np.array([cell_x,cell_y,cell_z]))
            except:
                print("atomic labels in .dat file may not be atomic symbols. Try passing associated electrons.dat file")
        
    return atom_obj

