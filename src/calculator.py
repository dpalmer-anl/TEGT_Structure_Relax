# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:14:50 2023

@author: danpa
"""

from ase import Atoms
import ase.io
import numpy as np
import subprocess
import TEGT_io
import os
from datetime import datetime
import json
import shutil
import shlex
from subprocess import Popen, PIPE, TimeoutExpired
from threading import Thread
from re import compile as re_compile, IGNORECASE
from tempfile import mkdtemp, NamedTemporaryFile, mktemp as uns_mktemp
import inspect
import warnings
from typing import Dict, Any
import numpy as np
from ase.parallel import paropen

#build ase calculator objects that calculates classical forces in lammps
#and tight binding forces from latte in parallel

class TEGT_Calc():
    
    def __init__(self,model_dict,lammps_command="${HOME}/lammps/src/lmp_serial ", 
                 latte_command="${HOME}/LATTE/LATTE_MPI_DOUBLE "):
        self.model_dict=model_dict
        self.create_model(self.model_dict)
        
    def get_forces(self):
        return self.forces
    def get_potential_energy(self):
        return self.potential_energy
        
    def run(self):
        #run lammps part on a single processor
        ase.io.write("datafile",atoms,format="lammps-data",atom_style="full")
        TEGT_io.set_lammps_data_setting('datafile',"atom types",2)
        TEGT_io.set_lammps_setting(self.lammps_input, "read_data", "datafile")
        TEGT_io.set_lammps_setting(self.lammps_input,"log","log.output")
        
        subprocess.call(self.lammps_command+" <"+self.lammps_input,shell=True)
        self.Lammps_results = TEGT_io.read_lammps_log("log.output")
        
        dump_file=self.get_lammps_setting("dump", self.lammps_input)
        
        atoms_dump = TEGT_io.read(dump_file,format="lammps-dump-text",index=":")
        self.Lammps_forces = atoms_dump.get_forces()
        
        #run lammps part first then run latte part. Sum the two
        
        TEGT_io.write_latte_dat(atoms,'latte.dat',electron_file='')
        subprocess.call(self.latte_command+" <"+self.latte_input,shell=True)
        self.Latte_results = self.read_latte_log('log.latte')
        self.Latte_forces = self.read_latte_forces('latte.forces') #? figure out how to output forces from late
        
        self.forces = self.Lammps_forces + self.Latte_forces
        self.potential_energy = self.Lammps_results['PotEng'] + self.Latte_results['Potential Energy']
        
    ##########################################################################
    
    #creating total energy tight binding model, performing calculations w/ model
    
    ##########################################################################
    def create_model(self,input_dict):
        """setup total energy model based on input dictionary parameters """
        cwd=os.getcwd()
        model_dict={"tight binding parameters":None,
             "orbitals":None,
             "intralayer potential":None,
             "interlayer potential":None,
             "calc type": None,
             "label":"",
             "nkp":1
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
        
        if self.model_dict['nkp']>1:
            if self.model_dict["intralayer potential"]:
                self.option_to_file[self.model_dict["intralayer potential"]] = \
                self.option_to_file[self.model_dict["intralayer potential"]]+'_nkp64'
            if self.model_dict["interlayer potential"]:
                self.option_to_file[self.model_dict["interlayer potential"]] = \
                                        self.option_to_file[self.model_dict["interlayer potential"]]+'_nkp64'
            
        if self.model_dict["calc type"]!="band structure":
            #write lammps input file based on given potentials and whether or not to use latte
            self.lammps_input="bilayer_graphene.md"
            self.write_lammps_input(self.lammps_input,self.model_dict["intralayer potential"],self.model_dict["interlayer potential"],use_latte)
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
            latte_file = os.path.join(self.repo_root,"parameters_potentials/latte/latte.in")
            subprocess.call("cp "+latte_file+" .",shell=True)
            
            if self.model_dict["tight binding parameters"]!=None:
                subprocess.call("cp -r "+self.option_to_file[self.model_dict["tight binding parameters"]]+" TBparam",shell=True)
            
            if self.model_dict['nkp']>1:
                self.set_latte_setting("latte.in", "KON", "1")
                kdict = {'NKX':int(np.sqrt(self.model_dict['nkp'])),
                         'NKY':int(np.sqrt(self.model_dict['nkp'])),
                         'NKZ':1,
                         'KSHIFTX':   0,
                         'KSHIFTY':   0,
                         'KSHIFTZ':   0}
                self.set_latte_kpoints('latte.in',kdict)
                
            if 'kdict' in self.model_dict.keys():
                self.set_latte_kpoints('latte.in',kdict)
                
            self.set_latte_setting("latte.in", "PARAMPATH", "TBparam")
            self.write_electron_file(self.model_dict["orbitals"],"TBparam/electrons.dat")
           
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
    a=2.46
    d=3.35
    atoms = fg.shift.make_graphene(stacking=['A','B'],cell_type='rect',
                        n_layer=2,n_1=5,n_2=5,lat_con=2.46,
                        sep=d,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)   
    
    model_dict = dict({"tight binding parameters":"Popov Van Alsenoy",
                     "orbitals":"pz",
                     "nkp":225,
                     "intralayer potential":"Pz rebo",
                     "interlayer potential":"Pz KC inspired"})
    
    calc = TEGT_Calc(atoms,model_dict)
    
    #if no tight binding parameters given, just run full relaxation in lammps
    dyn = TEGT_FIRE(atoms, restart='qn.pckl',logfile='log.output')
    dyn.run(fmax=1e-5)
    