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

#build ase calculator objects that calculates classical forces in lammps
#and tight binding forces from latte in parallel

class TEGT_Calc():
    
    def __init__(self,model_dict=None,restart_file=None,lammps_command="${HOME}/lammps/src/lmp_serial ", 
                 latte_command="${HOME}/LATTE/LATTE_DOUBLE "):
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
                     "kolmogorov crespi":os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/fullKC.txt"),
                     "KC inspired":os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp.txt"),
                     "Pz KC inspired":os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp_pz.txt"),
                     "Pz kolmogorov crespi":os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/fullKC_pz.txt"),
                     "Pz kolmogorov crespi + KC inspired":[os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/KC_insp.txt"),
                                                           os.path.join(self.repo_root,"parameters_potentials/lammps/interlayer_correction/fullKC_pz_correction.txt")],
                    }
        if type(restart_file)==str:
            f = open(restart_file,'r')
            self.model_dict = json.load(f)
            self.calc_dir = "/".join(restart_file.split("/")[-1])
            self.electron_file = "TBparam/electrons.dat"
        else:
            self.create_model(self.model_dict)
        self.forces=None
        self.potential_energy=None
        self.optimizer_type = 'LAMMPS'
        
        
    def get_forces(self):
        if self.forces==None:
            self.run()
        return self.forces
    
    def get_potential_energy(self,force_consistent=None):
        if self.potential_energy==None:
            self.run()
        return self.potential_energy
    
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
        
    def run(self,atoms):
        cwd=os.getcwd()
        os.chdir(self.calc_dir)
        if self.model_dict['calc type']=='static':
            #run lammps part on a single processor
            self.Lammps_forces, self.Lammps_results = self.run_lammps(atoms)
            self.forces = self.Lammps_forces
            self.potential_energy = self.Lammps_results['PotEng']
            
            #run lammps part first then run latte part. Sum the two
            if self.model_dict['use mpi']:
                self.Latte_forces,self.Latte_results = self.run_latte(atoms)
                self.forces = self.Lammps_forces + self.Latte_forces
                self.potential_energy = self.Lammps_results['PotEng'] + self.Latte_results['TotEng']
        elif self.model_dict['calc type']=="structure relaxation":
            if self.model_dict['use mpi']:
                #ase optimizer
                dyn = optimizer.TEGT_FIRE(atoms, restart='qn.pckl',logfile='log.output')
                dyn.run(fmax=1e-5)
            else:
                #run lammps part on a single processor
                self.Lammps_forces, self.Lammps_results = self.run_lammps(atoms)
                self.forces = self.Lammps_forces
                self.potential_energy = self.Lammps_results['PotEng']
        atoms.calc.forces = self.forces
        atoms.calc.potential_energy = self.potential_energy
        os.chdir(cwd)
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
        
        if self.model_dict['nkp']>1:
            if self.model_dict["intralayer potential"]:
                self.option_to_file[self.model_dict["intralayer potential"]] = \
                self.option_to_file[self.model_dict["intralayer potential"]]+'_nkp225'
            if self.model_dict["interlayer potential"]:
                self.option_to_file[self.model_dict["interlayer potential"]] = \
                                        self.option_to_file[self.model_dict["interlayer potential"]]+'_nkp225'
            
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
        
    def read_lammps_log(self,lammps_log):
        # !TODO: somehow communicate 'thermo_content' explicitly
        """Method which reads a LAMMPS output log file."""

        if isinstance(lammps_log, str):
            fileobj = paropen(lammps_log, "wb")
            close_log_file = True
        else:
            # Expect lammps_in to be a file-like object
            fileobj = lammps_log
            close_log_file = False

        # read_log depends on that the first (three) thermo_style custom args
        # can be capitalized and matched against the log output. I.e.
        # don't use e.g. 'ke' or 'cpu' which are labeled KinEng and CPU.
        mark_re = r"^\s*" + r"\s+".join(
            [x.capitalize() for x in self.parameters.thermo_args[0:3]]
        )
        _custom_thermo_mark = re_compile(mark_re)

        # !TODO: regex-magic necessary?
        # Match something which can be converted to a float
        f_re = r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|nan|inf))"
        n_args = len(self.parameters["thermo_args"])
        # Create a re matching exactly N white space separated floatish things
        _custom_thermo_re = re_compile(
            r"^\s*" + r"\s+".join([f_re] * n_args) + r"\s*$", flags=IGNORECASE
        )

        thermo_content = []
        line = fileobj.readline().decode("utf-8")
        while line and line.strip() != CALCULATION_END_MARK:
            # check error
            if 'ERROR:' in line:
                if close_log_file:
                    fileobj.close()
                raise RuntimeError(f'LAMMPS exits with error message: {line}')

            # get thermo output
            if _custom_thermo_mark.match(line):
                bool_match = True
                while bool_match:
                    line = fileobj.readline().decode("utf-8")
                    bool_match = _custom_thermo_re.match(line)
                    if bool_match:
                        # create a dictionary between each of the
                        # thermo_style args and it's corresponding value
                        thermo_content.append(
                            dict(
                                zip(
                                    self.parameters.thermo_args,
                                    map(float, bool_match.groups()),
                                )
                            )
                        )
            else:
                line = fileobj.readline().decode("utf-8")

        if close_log_file:
            fileobj.close()

        return thermo_content
        

if __name__=="__main__":
    import flatgraphene as fg
    import matplotlib.pyplot as plt
    a=2.46
    d=np.linspace(3,5,5)
    energy_ll = np.zeros(5)
    model_dict = dict({"tight binding parameters":"Popov Van Alsenoy",
                     "orbitals":"pz",
                     "nkp":225,
                     "intralayer potential":"Pz rebo",
                     "interlayer potential":"Pz KC inspired",
                     "use mpi":False,
                     'NGPU':0,
                     'calc type':'static',
                     'label':""})
    
    calc = TEGT_Calc(model_dict)
    for i,sep in enumerate(d):
        atoms = fg.shift.make_graphene(stacking=['A','B'],cell_type='rect',
                            n_layer=2,n_1=5,n_2=5,lat_con=2.46,
                            sep=sep,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)   
        
        
        calc.run(atoms)
        energy_ll[i] = calc.get_potential_energy()
    
    energy_ase = np.zeros(5)
    model_dict = dict({"tight binding parameters":"Popov Van Alsenoy",
                     "orbitals":"pz",
                     "nkp":225,
                     "intralayer potential":"Pz rebo",
                     "interlayer potential":"Pz KC inspired",
                     "use mpi":True,
                     'NGPU':0,
                     'calc type':'static',
                     'label':""})
    
    calc = TEGT_Calc(model_dict)
    for i,sep in enumerate(d):
        atoms = fg.shift.make_graphene(stacking=['A','B'],cell_type='rect',
                            n_layer=2,n_1=5,n_2=5,lat_con=2.46,
                            sep=sep,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)   
        
        
        calc.run(atoms)
        energy_ase[i] = calc.get_potential_energy()
    
    plt.plot(d,energy_ll)
    plt.plot(d,energy_ase)
    plt.xlabel("layer sep")
    plt.ylabel("energy (meV)")
    plt.savefig("ab_energy.png")
    plt.show()
    
    