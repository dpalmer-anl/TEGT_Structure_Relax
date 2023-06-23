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
#from TEGT_Structure_Relax.potential_lib import option_to_file

repo_root = "/".join(os.getcwd().split("/")[:-1])
option_to_file={"Porezag":os.path.join(repo_root,"parameters_potentials/latte/Porezag/latte"),
                     "Nam Koshino":os.path.join(repo_root,"parameters_potentials/latte/Nam_Koshino/latte"),
                     "Popov Van Alsenoy":os.path.join(repo_root,"parameters_potentials/latte/Porezag_Popov_Van_Alsenoy/latte"),
                     "Porezag Pairwise":os.path.join(repo_root,"parameters_potentials/lammps/intralayer_correction/porezag_c-c.table"),
                     "Rebo":os.path.join(repo_root,"parameters_potentials/lammps/intralayer_correction/CH.rebo"),
                     "Pz pairwise":os.path.join(repo_root,"parameters_potentials/lammps/intralayer_correction/pz_pairwise_correction.table"),
                     "Pz rebo":os.path.join(repo_root,"parameters_potentials/lammps/intralayer_correction/CH_pz.rebo"),
                     "kolmogorov crespi":os.path.join(repo_root,"parameters_potentials/lammps/interlayer_correction/fullKC.txt"),
                     "KC inspired":os.path.join(repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp.txt"),
                     "Pz KC inspired":os.path.join(repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp_pz.txt"),
                     "Pz kolmogorov crespi":os.path.join(repo_root,"parameters_potentials/lammps/interlayer_correction/fullKC_pz.txt"),
                     "Pz kolmogorov crespi + KC inspired":[os.path.join(repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp.txt"),
                                                           os.path.join(repo_root,"parameters_potentials/lammps/interlayer_correction/fullKC_pz_correction.txt")],
                    }

def write_lammps_input(input_filename,intralayer_pot,interlayer_pot,calc_type,use_latteLammps):
    """intra_corrective_potential_options= Porezag Pairwise, Rebo , Pz pairwise, Pz rebo
        inter_corrective_potential_options = kolmogorov crespi, KC inspired, Pz KC inspired
                                  , Pz kolmogorov crespi , Pz kolmogorov crespi + KC inspired """
    
    intro="\n\
            units		metal\n\
            atom_style	full\n\
            atom_modify    sort 0 0.0\n\
            box tilt large\n\
            read_data datafile\n\
            group top type 1\n\
            group bottom type 2\n\
            set group top mol 1\n\
            set group bottom mol 2\n\
            mass 1 12.0100\n\
            mass 2 12.0200\n\
            velocity	all create 0.0 87287 loop geom\n"
            
    if intralayer_pot!=None and interlayer_pot!=None:
        pair_style="pair_style       hybrid/overlay "
    else:
        pair_style = "pair_style "
    pair_coeff=""
    #select intralayer potential
    if intralayer_pot=="Porezag Pairwise":
        pair_style+=" table linear 500 "
        pair_coeff+="pair_coeff	1 1 table "+option_to_file["Porezag Pairwise"]+" POREZAG_C\n"
        pair_coeff+= "pair_coeff	2 2 table "+option_to_file["Porezag Pairwise"]+" POREZAG_C\n"
    elif intralayer_pot=="Rebo":
        pair_style+=" rebo "
        pair_coeff+="pair_coeff	* * rebo "+option_to_file["Rebo"]+"        C C\n"
    elif intralayer_pot == "Pz pairwise":
        pair_style+=" table linear 500 "
        pair_coeff+="pair_coeff	1 1 table "+option_to_file["Pz pairwise"]+" POREZAG_sxy_C\n"
        pair_coeff+="pair_coeff	2 2 table "+option_to_file["Pz pairwise"]+" POREZAG_sxy_C\n"
    elif intralayer_pot == "Pz rebo":
        pair_style+=" rebo "
        pair_coeff+="pair_coeff	* * rebo "+option_to_file["Pz rebo"]+"        C C\n"
    else:
        pair_style="pair_style none\n"
        pair_coeff=""
    
    #select interlayer potential
    if interlayer_pot=="KC inspired":
        pair_style += " reg/dep/poly 10.0 0 "
        pair_coeff+="pair_coeff       * *   reg/dep/poly  "+option_to_file["KC inspired"]+"   C C\n"
        
    elif interlayer_pot == "kolmogorov crespi":
        pair_style += " kolmogorov/crespi/full 10.0 0 "
        pair_coeff+="pair_coeff       * *   kolmogorov/crespi/full  "+option_to_file["kolmogorov crespi"]+"   C C\n"
    
    elif interlayer_pot == "Pz KC inspired":
        pair_style += " reg/dep/poly 10.0 0 "
        pair_coeff+="pair_coeff       * *   reg/dep/poly  "+option_to_file["Pz KC inspired"]+"   C C\n"
        
    elif interlayer_pot == "Pz kolmogorov crespi":
        pair_style += " kolmogorov/crespi/full 10.0 0 "
        pair_coeff+="pair_coeff       * *   kolmogorov/crespi/full  "+option_to_file["Pz kolmogorov crespi"]+"   C C\n"
    
    elif interlayer_pot == "Pz kolmogorov crespi + KC inspired":
        pair_style += " kolmogorov/crespi/full 10.0 0 reg/dep/poly 10.0 0 "
        pair_coeff+="pair_coeff       * *   kolmogorov/crespi/full  "+option_to_file["Pz kolmogorov crespi + KC inspired"][1]+"   C C\n"
        pair_coeff+="pair_coeff       * *   reg/dep/poly  "+option_to_file["Pz kolmogorov crespi + KC inspired"][0]+"   C C\n"
    pair_style+="\n"
    #delete_atoms overlap 1.3 all all\n\ 
    calc_set="neighbor	2.0 bin\n\
              neigh_modify	delay 0 one 10000\n\
              timestep 0.00025\n"
    if use_latteLammps:
        calc_set+="variable latteE equal \"(ke + f_2)\"\n\
                variable kinE equal \"ke\"\n\
                variable potE equal \"f_2\"\n\
                thermo 1\n\
                thermo_style   custom step pe ke etotal temp epair v_latteE\n\
                fix   2 all latte NULL\n"
    else:
        calc_set+="thermo 1\n\
                thermo_style   custom step pe ke etotal temp epair\n"
                
    calc_set+="dump           mydump2 all custom 1 dump.latte id type x y z fx fy fz\n\
                log log.output\n\
                fix		1 all nve\n"
                
    if calc_type=="static":
        calc_set+="run 0\n"
    elif calc_type=="structure relaxation":
        calc_set+="min_style fire\n minimize       1e-8 1e-9 3000 10000000\n"
    else:
        print("WARNING: calc type must be structure relaxation or static")
    
    with open(input_filename,"w+") as f:
        f.write(intro)
        f.write(pair_style)
        f.write(pair_coeff)
        f.write(calc_set)
    return None
    
def write_electron_file(orbitals,filename):
    lines = ["Noelem= 2 \n",
             "Element basis Numel Es Ep Ed Ef Mass HubbardU Wss Wpp Wdd Wff\n"]
    
    if orbitals=="pz":
        lines.append("T pz 1.0 -13.7388 -5.2887 0.0000 0.0000 12.0100 9.9241 0.0000 0.0000 0.0000 0.0000\n")
        lines.append("B pz 1.0 -13.7388 -5.2887 0.0000 0.0000 12.0200 9.9240 0.0000 0.0000 0.0000 0.0000\n")
    elif orbitals=="s,px,py,pz":
        lines.append("T sp 4.0 -13.7388 -5.2887 0.0000 0.0000 12.0100 9.9241 0.0000 0.0000 0.0000 0.0000 \n")
        lines.append("B sp 4.0 -13.7388 -5.2887 0.0000 0.0000 12.0200 9.9240 0.0000 0.0000 0.0000 0.0000 \n")
    else:
        print("WARNING: only valid options for orbitals are pz and s,px,py,pz")
        exit()
        
    with open(filename,"w+") as f:
        f.writelines(lines)
    
def get_lammps_setting(keyword, input_file):
    """get lammps setting from lammps input file
    
    :param keyword: (str) setting to extract from lammps input file
    
    :param input_file: (str) path to lammps input file
    """
    with open(input_file,"r") as f:
        lines=f.readlines()
        for l in lines:
            
            if keyword in l:
                if keyword == "dump":
                    words = l.split(" ")
                    words = [value for value in words if value != '']
                    
                    return words[5] #this should be the dumpfile name, but might not be robust
                else:
                    setting=l.replace(keyword,"",1)
                    setting=setting.strip()
                    return setting
            
        
def set_lammps_setting(input_path, setting, val):
    """set lammps setting in lammps input file
    
    :param input_path: (str) path to lammps input file
    
    :param setting: (str) setting to set in lammps input file
    
    :param val: (str or float or int) setting value
    """
    with open(input_path,"r") as f:
        lines=f.readlines()
        for i,l in enumerate(lines):
            if setting in l:
                new_line=setting+ " "+str(val)+" \n"
                lines[i]=new_line
                break
        
    with open(input_path,"w") as f:
        f.writelines(lines)

def set_lammps_data_setting(input_path,setting,val):
    """set lammps setting in lammps data file
    
    :param input_path: (str) path to lammps data file
    
    :param setting: (str) setting to set in lammps data file
    
    :param val: (str or float or int) setting value
    """
    with open(input_path,"r") as f:
        lines=f.readlines()
        for i,l in enumerate(lines):
            if setting in l:
                new_line=str(val)+" "+setting+ " \n"
                lines[i]=new_line
                break
        
    with open(input_path,"w") as f:
        f.writelines(lines)
def get_latte_setting(keyword, input_file):
    """get setting from latte input file """
    with open(input_file,"r") as f:
        lines=f.readlines()
        for l in lines:
            if keyword in l:
                setting=l.replace(keyword,"",1)
                setting=setting.strip()
                return setting
            
def set_latte_setting(input_path, setting, val):
    """set setting in latte input file """
    with open(input_path,"r") as f:
        lines=f.readlines()
        for i,l in enumerate(lines):
            if setting in l:
                if "latte" in input_path:
                    new_line="  "+setting+ "= \'"+str(val)+"\' \n"
                else:
                    new_line="  "+setting+ " "+str(val)+" \n"
                lines[i]=new_line
                break

    with open(input_path,"w") as f:
        f.writelines(lines)
        f.close() 
        
def set_latte_kpoints(input_path,kdict):
    """set kpoints in latte input file """
    with open(input_path,"r") as f:
        lines=f.readlines()
        found_kmesh=False
        for i,l in enumerate(lines):
            if 'KMESH' in l:
                found_kmesh=True
            if found_kmesh:
                for setting in kdict:
                    if setting in l:
                        new_line="  "+setting+ "= \'"+str(kdict[setting])+"\' \n"
                        lines[i]=new_line
                        break

    with open(input_path,"w") as f:
        f.writelines(lines)
        f.close() 
        
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

def read_latte_forces(filename):
    return np.loadtxt(filename,skiprows=1)[:,4:]
    
def read_latte_log(latte_output):
    with open(latte_output,"r") as f:
        lines=f.readlines()
        for l in lines:
            #if "Total energy (zero K) =" in l:
            if "FREE ENERGY =" in l:
                
                return {"TotEng":float(re.findall(r'[-+]?[.]?[:\.\d]+',l)[0])}