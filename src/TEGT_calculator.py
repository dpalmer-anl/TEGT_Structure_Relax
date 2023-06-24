# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:14:50 2023

@author: danpa
"""

import ase.io
import numpy as np
import subprocess
import TEGT_io
import os
from datetime import datetime
import json
from re import compile as re_compile, IGNORECASE
import numpy as np
from ase.parallel import paropen
from ase.calculators.lammps import CALCULATION_END_MARK
import optimizer
import lammps_logfile
from ase.calculators.calculator import Calculator, all_changes


#build ase calculator objects that calculates classical forces in lammps
#and tight binding forces from latte in parallel

class TEGT_Calc(Calculator):
    
    implemented_properties = ['energy',  'forces','potential_energy']
    def __init__(self,model_dict=None,restart_file=None,lammps_command="${HOME}/lammps/src/lmp_serial ", 
                 latte_command="${HOME}/LATTE/LATTE_DOUBLE ",**kwargs):
        Calculator.__init__(self, **kwargs)
        self.model_dict=model_dict
        self.lammps_command = lammps_command
        self.latte_command = latte_command
        self.repo_root = "/".join(os.getcwd().split("/")[:-1])
        self.option_to_file={"Porezag":os.path.join(self.repo_root,"parameters_potentials/latte/Porezag/latte"),
                     "Nam Koshino":os.path.join(self.repo_root,"parameters_potentials/latte/Nam_Koshino/latte"),
                     "Popov Van Alsenoy":os.path.join(self.repo_root,"parameters_potentials/latte/Porezag_Popov_Van_Alsenoy/latte"),
                     "Porezag Pairwise":os.path.join(self.repo_root,"parameters_potentials/lammps/intralayer_correction/porezag_c-c.table"),
                     "Rebo":os.path.join(self.repo_root,"parameters_potentials/lammps/intralayer_correction/CH.rebo"),
                     "Pz pairwise":os.path.join(self.repo_root,"parameters_potentials/lammps/intralayer_correction/pz_pairwise_correction.table"),
                     "Pz rebo":os.path.join(self.repo_root,"parameters_potentials/lammps/intralayer_correction/CH_pz.rebo"),
                     "Pz rebo nkp225":os.path.join(self.repo_root,"parameters_potentials/lammps/intralayer_correction/CH_pz.rebo_nkp225"),
                     "kolmogorov crespi":os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/fullKC.txt"),
                     "KC inspired":os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp.txt"),
                     "Pz KC inspired":os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp_pz.txt"),
                     "Pz KC inspired nkp225":os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp_pz.txt_nkp225"),
                     "Pz kolmogorov crespi":os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/fullKC_pz.txt"),
                     "Pz kolmogorov crespi + KC inspired":[os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp.txt"),
                                                           os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/fullKC_pz_correction.txt")],
                    }
        if type(restart_file)==str:
            f = open(restart_file,'r')
            self.model_dict = json.load(f)
            self.calc_dir = restart_file.replace('model.json',"")
            self.electron_file = "TBparam/electrons.dat"
            self.lammps_input="bilayer_graphene.md"
            if self.model_dict["tight binding parameters"] == None:
                self.use_latte=False
            else:
                self.use_latte=True

        else:
            self.create_model(self.model_dict)
        self.forces=None
        self.potential_energy=None
        self.optimizer_type = 'LAMMPS'
    
    def run_lammps(self,atoms):
        ase.io.write("datafile",atoms,format="lammps-data",atom_style="full")
        TEGT_io.set_lammps_data_setting('datafile',"atom types",2)
        
        subprocess.call(self.lammps_command+" <"+self.lammps_input,shell=True)

        dump_file=TEGT_io.get_lammps_setting("dump", self.lammps_input)
        
        atoms_dump = ase.io.read(dump_file,format="lammps-dump-text",index=":")
        forces = []
        for a in atoms_dump:
            forces.append(a.get_forces())
        log = lammps_logfile.File("log.output")
        results = {}
        for key in log.get_keywords():
            e_tmp = log.get(key)
            results.update({key:e_tmp})
        return np.squeeze(forces), results
    
    def run_latte(self,atoms):
        TEGT_io.write_latte_dat(atoms,'latte.dat',electron_file=self.electron_file)
        subprocess.call(self.latte_command+" < latte.in>log.latte",shell=True)
        #figure how to output latte forces, read in latte forces, read in latte energies
        return TEGT_io.read_latte_forces('ForceDump.dat'), TEGT_io.read_latte_log('log.latte')
        
    def calculate(self, atoms, properties=['energy','forces','potential_energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        cwd=os.getcwd()
        os.chdir(self.calc_dir)
        #if self.model_dict['calc type']=='static':
            #run lammps part on a single processor
        self.Lammps_forces, self.Lammps_results = self.run_lammps(atoms)
        self.forces = self.Lammps_forces
        self.potential_energy = self.Lammps_results['PotEng']
        #run lammps part first then run latte part. Sum the two
        if self.model_dict['use mpi'] and self.use_latte:
            self.Latte_forces,self.Latte_results = self.run_latte(atoms)
            self.results['forces'] = self.Lammps_forces + self.Latte_forces
            self.results['potential_energy'] = self.Lammps_results['PotEng'] + self.Latte_results['TotEng']
            self.results['energy'] = self.Lammps_results['TotEng'] + self.Latte_results['TotEng']
        # elif self.model_dict['calc type']=="structure relaxation":
        #     if self.model_dict['use mpi']:
        #         #ase optimizer
        #         # dyn = optimizer.TEGT_FIRE(atoms, restart='qn.pckl',logfile='log.output')
        #         # dyn.run(fmax=1e-5)
        #         print("run externally")
        #         exit()
        else:
           self.results['forces'] = self.Lammps_forces
           self.results['potential_energy'] = self.Lammps_results['PotEng']
           self.results['energy'] = self.Lammps_results['TotEng']
        #atoms.calc.forces = self.forces
        #atoms.calc.potential_energy = self.potential_energy
        os.chdir(cwd)
        
    def run(self,atoms):
        self.calculate(atoms)
    ##########################################################################
    
    #creating total energy tight binding model, performing calculations w/ model
    
    ##########################################################################
    def create_model(self,input_dict):
        """setup total energy model based on input dictionary parameters 
        using mpi and latte will result in ase optimizer to be used, 
        else lammps runs relaxation """
        cwd=os.getcwd()
        
        model_dict={"tight binding parameters":None,
             "orbitals":None,
             "intralayer potential":None,
             "interlayer potential":None,
             "calc type": None,
             "label":"",
             "nkp":1,
             "use mpi":True,
             'NGPU':0,
             } 
        for k in input_dict.keys():
            model_dict[k] = input_dict[k]
        self.model_dict = model_dict
        
        calc_hash = str(hash(datetime.now()) )
        self.calc_dir = os.path.join(cwd,"calc_"+self.model_dict['label']+"_"+calc_hash)
        os.mkdir(self.calc_dir)
        os.chdir(self.calc_dir)
        #write model parameters to json file
        with open("model.json", "w+") as outfile:
            json.dump(self.model_dict, outfile)
        if self.model_dict["tight binding parameters"] == None:
            use_latte=False
        else:
            use_latte=True
            use_latteLammps=True
            latte_file = os.path.join(self.repo_root,"parameters_potentials/latte/LatteLammps.in")
        self.use_latte = use_latte
        if self.model_dict['nkp']>1:
            if self.model_dict["intralayer potential"]:
                if self.model_dict["intralayer potential"].split(" ")[-1]!='nkp225':
                    self.model_dict["intralayer potential"] = self.model_dict["intralayer potential"]+' nkp225'
            if self.model_dict["interlayer potential"]:
                if self.model_dict["interlayer potential"].split(" ")[-1]!='nkp225':
                    self.model_dict["interlayer potential"] = self.model_dict["interlayer potential"]+' nkp225'
            
        #write lammps input file based on given potentials and whether or not to use latte
        self.lammps_input="bilayer_graphene.md"
        calc_type = self.model_dict['calc type']
        if self.model_dict['use mpi']:
            self.optimizer_type = 'ASE'
            use_latteLammps = False
            latte_file = os.path.join(self.repo_root,"parameters_potentials/latte/LatteStatic.in")
            #lammps input file must do static calcs for ase optimizer
            calc_type = 'static'
        TEGT_io.write_lammps_input(self.lammps_input,self.model_dict["intralayer potential"],
                                self.model_dict["interlayer potential"],
                                calc_type,use_latteLammps)
        
        TEGT_io.set_lammps_setting(self.lammps_input, "read_data", "datafile")
        TEGT_io.set_lammps_setting(self.lammps_input,"log","log.output")
        #copy lammps potential files to calculation directory
        if self.model_dict["intralayer potential"]!=None:
            subprocess.call("cp "+ self.option_to_file[self.model_dict["intralayer potential"]] + " .",shell=True)
        if self.model_dict["interlayer potential"]!=None:
            if type(self.option_to_file[self.model_dict["interlayer potential"]]) ==list:
                for f in self.option_to_file[self.model_dict["interlayer potential"]]:
                    subprocess.call("cp "+ f + " .",shell=True)
            else:
                subprocess.call("cp "+ self.option_to_file[self.model_dict["interlayer potential"]] + " .",shell=True)
    
        #copy latte input file and parameters to calculation directory
        if use_latte:
            subprocess.call("cp "+latte_file+" ./latte.in",shell=True)
            
            if self.model_dict["tight binding parameters"]!=None:
                subprocess.call("cp -r "+self.option_to_file[self.model_dict["tight binding parameters"]]+" TBparam",shell=True)
            
            if self.model_dict['nkp']>1:
                TEGT_io.set_latte_setting("latte.in", "KON", "1")
                kdict = {'NKX':int(np.sqrt(self.model_dict['nkp'])),
                         'NKY':int(np.sqrt(self.model_dict['nkp'])),
                         'NKZ':1,
                         'KSHIFTX':   0,
                         'KSHIFTY':   0,
                         'KSHIFTZ':   0}
                TEGT_io.set_latte_kpoints('latte.in',kdict)
                
            if 'kdict' in self.model_dict.keys():
                self.set_latte_kpoints('latte.in',kdict)
                
            TEGT_io.set_latte_setting("latte.in", "PARAMPATH", "TBparam")
            TEGT_io.set_latte_setting("latte.in","NGPU",str(self.model_dict['NGPU']))
            TEGT_io.write_electron_file(self.model_dict["orbitals"],"TBparam/electrons.dat")
            self.electron_file = "TBparam/electrons.dat"
           
        os.chdir(cwd)
        

        

if __name__=="__main__":
    import flatgraphene as fg
    import matplotlib.pyplot as plt

    lammps_command = "jsrun -n1 -c42 -a1 -g1 -bpacked:42 -dpacked -EOMP_NUM_THREADS=168 ${HOME}/lammps/src/lmp_serial "
    latte_command = "jsrun -n1 -c42 -a1 -g6 -bpacked:42 -dpacked -EOMP_NUM_THREADS=168 ${HOME}/LATTE/LATTE_MPI_DOUBLE "
    a=2.46
    d=np.linspace(3,5,9)
    energy_ll = np.zeros(9)
    tbenergy_ll = np.zeros(9)
    """model_dict = dict({"tight binding parameters":"Popov Van Alsenoy",
                     "orbitals":"pz",
                     "nkp":225,
                     "intralayer potential":"Pz rebo",
                     "interlayer potential":"Pz KC inspired",
                     "use mpi":False,
                     'NGPU':6,
                     'calc type':'static',
                     'label':""})
    
    calc = TEGT_Calc(model_dict,latte_command=latte_command,lammps_command=lammps_command)
    for i,sep in enumerate(d):
        atoms = fg.shift.make_graphene(stacking=['A','B'],cell_type='rect',
                            n_layer=2,n_1=5,n_2=5,lat_con=2.46,
                            sep=sep,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)   
        
        
        calc.run(atoms)
        energy_ll[i] = calc.get_potential_energy()
        tbenergy_ll[i] = calc.Lammps_results['v_latteE']
    """
    energy_ase = np.zeros(9)
    tbenergy_ase = np.zeros(9)
    kcenergy_ase = np.zeros(9)
    model_dict = dict({"tight binding parameters":"Popov Van Alsenoy",
                      "orbitals":"pz",
                      "nkp":225,
                      "intralayer potential":"Pz rebo nkp225",
                      "interlayer potential":"Pz KC inspired nkp225",
                      "use mpi":True,
                      'NGPU':6,
                      'calc type':'static',
                      'label':""})
    
    
    calc = TEGT_Calc(model_dict) #,latte_command=latte_command,lammps_command=lammps_command)
    for i,sep in enumerate(d):
        atoms = fg.shift.make_graphene(stacking=['A','B'],cell_type='rect',
                            n_layer=2,n_1=5,n_2=5,lat_con=2.46,
                            sep=sep,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)   
        
        
        atoms.calc = calc
        energy_ase[i] = atoms.get_potential_energy()/atoms.get_global_number_of_atoms()
        tbenergy_ase[i] = atoms.calc.Latte_results['TotEng']/atoms.get_global_number_of_atoms()
        kcenergy_ase[i] = atoms.calc.Lammps_results['TotEng']/atoms.get_global_number_of_atoms()
    
    #plt.plot(d,energy_ll-energy_ll[3],label='total energy latte-lammps')
    plt.plot(d,energy_ase-energy_ase[2] ,label='total energy ase')
    plt.plot(d, kcenergy_ase-kcenergy_ase[2],label='lammps energy')
    #plt.plot(d,tbenergy_ll-tbenergy_ll[-1] ,label='tb energy latte-lammps')
    plt.plot(d,tbenergy_ase -tbenergy_ase[2] ,label='tb energy ase')
    plt.xlabel("layer sep")
    plt.legend()
    plt.ylim(-0.013,0.030)
    plt.ylabel("energy (meV)")
    plt.savefig("ab_energy.png")
    plt.show()
    
    
