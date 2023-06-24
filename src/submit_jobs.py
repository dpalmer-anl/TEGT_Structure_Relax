# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:45:42 2023

@author: danpa
"""
import ase.io
import os
import glob
import subprocess
from datetime import datetime
import numpy as np

def submit_batch_file(executable,batch_options):

    sbatch_file="job"+str(hash(datetime.now()) )+".qsub"
    batch_copy = batch_options.copy()

    prefix="#BSUB "
    with open(sbatch_file,"w+") as f:
        #f.write("#!/bin/csh -vm \n \n")
        f.write("#!/bin/bash\n")
        f.write("# Begin LSF Directives\n")

        modules=batch_copy["modules"]

        for key in batch_copy:
            if key == "modules":
                continue
            f.write(prefix+key+str(batch_copy[key])+"\n")

        for m in modules:
            f.write("module load "+m+"\n")
        
        f.write("\nsource activate /ccs/home/dpalmer3/.conda/envs/my_env\n")
        f.write("export LD_LIBRARY_PATH=\"${HOME}/magma/lib:$LD_LIBRARY_PATH\"\n")
        f.write(executable)
    subprocess.call("bsub -L $SHELL "+sbatch_file,shell=True)
    
if __name__=="__main__":
    batch_options = {
                    "-P":" MAT221",
                    "-W":" 24:00",
                    "-nnodes":" 1",
                    "-alloc_flags": " gpumps",
                    "-J":" ase_relax",
                    "-o":" log.%J",
                    "-e":" error.%J",
                    "-q":" batch-hm",
                    "-N":" dpalmer3@illinois.edu",
                    "modules": ["gcc", "netlib-lapack","python","cuda"]}
    theta_vals = np.array([0.88,0.99,1.05,1.08,1.12,1.16,1.20,1.47,1.89,2.88])
    theta_vals = np.array([2.88])
    for t in theta_vals:
        executable = "python run_relax -t "+str(t)
        if theta_vals > 1.19:
            batch_options["-nnodes"] = 19
        else:
            batch_options["-nnodes"] = 38
        submit_batch_file(executable,batch_options)