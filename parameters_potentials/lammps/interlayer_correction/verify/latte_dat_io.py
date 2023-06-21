# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:42:04 2021

@author: danpa
"""

import numpy as np 
import re
from ase import Atoms
import ase.io



def write_latte_dat(ase_obj,filename,electron_file=None):
    """write latte coordinate file from ase.Atoms object
    
    :param ase_obj: (ase.Atoms obj) ase atoms object where geometry is stored
    
    :electron_file: (str) electron.dat file containing masses of atomic species.
        Masses in electron file are matched with masses in ase.Atoms object in order
        to set symbols in latte data file
    
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
    
    :electron_file: (str) electron.dat file containing masses of atomic species.
        Masses in electron file are used to set masses in Ase.atoms object, if 
        symbols in latte data file are not chemical species.
    
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

def cell_to_bounds(cell):
    """convert simulation cell to box lengths and skews for lammps
    
    :param cell: (arr) 3x3 array defining simulation cell
    
    :returns: (float,float,float,float,float,float) length in x, length in y,
            length in z, xy skew, xz skew, yz skew. """
    #vector algebra to get unit cell skews
    cell=cell.T
    a_=cell[:,0]
    b_=cell[:,1]
    c_=cell[:,2]
    
    lx=a_[0]
    ly=b_[1]
    lz=c_[2]
    xy=b_[0]
    xz=c_[0]
    yz=c_[1]
    
    
    return lx,ly,lz,xy,xz,yz


def write_lammps(fname,ase_obj):
    """write lammps data file from ase.atom.atoms object. This function is 
    necessary because ase.io.write writes triclinic cells incorrectly.
    
    :param fname: (str) filename to write object to
    
    :param ase_obj: (object) ase.atom.atoms object to write to file
    
    """
    cell=np.array(ase_obj.get_cell())
    xyz=ase_obj.get_positions()
    natom=np.shape(xyz)[0]
    
    lx,ly,lz,xy,xz,yz=cell_to_bounds(cell)
    
    with open(fname,'w+') as f:
        
        f.write(fname+ " (written by ASE)      \n\n")
        f.write(str(natom)+" 	 atoms \n")
        f.write("2 atom types \n")
        f.write("0.0      "+str(lx)+"  xlo xhi \n") #fix xhi 
        f.write("0.0      "+str(ly)+ " ylo yhi \n")
        f.write("0.0      "+str(lz)+" zlo zhi \n")
        f.write("    "+str(xy)+"                       "+str(xz)+"                       "+str(yz)+"  xy xz yz \n\n\n")
        f.write("Atoms \n\n")
        
        m1=ase_obj.get_masses()[0]
        for i,a in enumerate(ase_obj):
            if a.mass==m1:
                atom_type="1"
            else:
                atom_type="2"
                
            f.write(str(i+1)+" "+atom_type+" "+atom_type+" 0 ")
            pos=np.array(a.position)
            str_pos=" ".join(map(str,pos))
            f.write(str_pos+" \n")
    
if __name__=="__main__":
    import flatgraphene as fg
    filename="test.dat"
    a_nn=2.529/np.sqrt(3)
    sep=3.35
    atoms=fg.shift.make_graphene(stacking=['A','B'],cell_type='hex',n_layer=2,
		        n_1=5,n_2=5,lat_con=0.0,a_nn=a_nn,mass=[12.01,12.02],sep=sep,sym=['B','Ti'],h_vac=3)
    efile="C:/Users/danpa/Documents/research/twisted-graphene-geometry-optimization/parameters_potentials/latte/Porezag_Popov_Van_Alsenoy/latte/electrons.dat"
    write_latte_dat(atoms,"test_coords.dat",electron_file=efile)
    #obj=read_latte_dat("test_coords.dat",electron_file=efile)
    #print(obj.get_masses())
    fname="ab_lammps.data"
    write_lammps(fname,atoms)
    
    