# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:39:37 2023

@author: danpa
"""
import warnings
import numpy as np
import glob
from ase.optimize.optimize import Optimizer

class TEGT_FIRE(Optimizer):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 dt=0.1, maxstep=None, maxmove=None, dtmax=1.0, Nmin=5,
                 finc=1.1, fdec=0.5,
                 astart=0.1, fa=0.99, a=0.1, master=None, downhill_check=False,
                 position_reset_callback=None, force_consistent=None):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        downhill_check: boolean
            Downhill check directly compares potential energies of subsequent
            steps of the FIRE algorithm rather than relying on the current
            product v*f that is positive if the FIRE dynamics moves downhill.
            This can detect numerical issues where at large time steps the step
            is uphill in energy even though locally v*f is positive, i.e. the
            algorithm jumps over a valley because of a too large time step.

        position_reset_callback: function(atoms, r, e, e_last)
            Function that takes current *atoms* object, an array of position
            *r* that the optimizer will revert to, current energy *e* and
            energy of last step *e_last*. This is only called if e > e_last.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.  Only meaningful
            when downhill_check is True.
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                            master)

        self.dt = dt
        self.Nsteps = 0

        if maxstep is not None:
            self.maxstep = maxstep
        elif maxmove is not None:
            self.maxstep = maxmove
            warnings.warn('maxmove is deprecated; please use maxstep',
                          np.VisibleDeprecationWarning)
        else:
            self.maxstep = self.defaults['maxstep']

        self.dtmax = dtmax
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = a
        self.downhill_check = downhill_check
        self.position_reset_callback = position_reset_callback

    def initialize(self):
        self.v = None

    def read(self):
        self.v, self.dt = self.load()

    def step(self, f=None):
        atoms = self.atoms
        calc.run(atoms)
        if f is None:
            f = atoms.get_forces()

        if self.v is None:
            self.v = np.zeros((len(atoms), 3))
            if self.downhill_check:
                self.e_last = atoms.get_potential_energy()
                self.r_last = atoms.get_positions().copy()
                self.v_last = self.v.copy()
        else:
            is_uphill = False
            if self.downhill_check:
                e = atoms.get_potential_energy(
                    force_consistent=self.force_consistent)
                # Check if the energy actually decreased
                if e > self.e_last:
                    # If not, reset to old positions...
                    if self.position_reset_callback is not None:
                        self.position_reset_callback(atoms, self.r_last, e,
                                                     self.e_last)
                    atoms.set_positions(self.r_last)
                    is_uphill = True
                self.e_last = atoms.get_potential_energy()
                self.r_last = atoms.get_positions().copy()
                self.v_last = self.v.copy()

            vf = np.vdot(f, self.v)
            if vf > 0.0 and not is_uphill:
                self.v = (1.0 - self.a) * self.v + self.a * f / np.sqrt(
                    np.vdot(f, f)) * np.sqrt(np.vdot(self.v, self.v))
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.Nsteps += 1
            else:
                self.v[:] *= 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.Nsteps = 0

        self.v += self.dt * f
        dr = self.dt * self.v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxstep:
            dr = self.maxstep * dr / normdr
        r = atoms.get_positions()
        atoms.set_positions(r + dr)
        self.dump((self.v, self.dt))

def restart_relax(calc_dir):
    traj_file = glob.glob(os.path.join(calc_dir,'*.traj'),recursive=True)[0]
    try:
        atoms = ase.io.read(traj_file)
    except:
        atoms = ase.io.read(os.path.join(calc_dir,'dump.latte'),format='lammps-dump-text')
        
    calc = TEGT_calculator.TEGT_Calc(restart_file = os.path.join(calc_dir,'model.json'))
    atoms.calc = calc
    dyn = TEGT_FIRE(atoms, restart=os.path.join(calc_dir,'qn.pckl')
                    ,logfile=os.path.join(calc_dir,'ase_opt.output'),
                     trajectory=traj_file)
    return dyn    
if __name__=="__main__":
    #if no tight binding parameters given, just run full relaxation in lammps
    import flatgraphene as fg
    import TEGT_calculator
    import os
    import ase.io
    run=False
    restart=True
    if run:
        a=2.46
        d=3.35
        atoms = fg.shift.make_graphene(stacking=['A','B'],cell_type='rect',
                            n_layer=2,n_1=5,n_2=5,lat_con=2.46,
                            sep=d,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)   
        
        model_dict = dict({"tight binding parameters":"Popov Van Alsenoy",
                         "orbitals":"pz",
                         "nkp":225,
                         "intralayer potential":"Pz rebo",
                         "interlayer potential":"Pz KC inspired",
                         'use mpi':True})
        model_dict = dict({
                         "intralayer potential":"Rebo",
                         "interlayer potential":"kolmogorov crespi",
                         "use mpi":True,
                         'NGPU':0,
                         'calc type':'static',
                         'label':""})
        
        calc = TEGT_calculator.TEGT_Calc(model_dict)
        atoms.calc = calc
        calc_dir = calc.calc_dir
        dyn = TEGT_FIRE(atoms, restart=os.path.join(calc_dir,'qn.pckl')
                        ,logfile=os.path.join(calc_dir,'ase_opt.output'),
                         trajectory=os.path.join(calc_dir,'ab_bilayer.traj'))
        dyn.run(fmax=1e-5)
        
    if restart:
        calc_dir = 'calc__116969195546821886'
        atoms = ase.io.read(os.path.join(calc_dir,'dump.latte'),format='lammps-dump-text')
        
        calc = TEGT_calculator.TEGT_Calc(restart_file = os.path.join(calc_dir,'model.json'))
        atoms.calc = calc
        dyn = TEGT_FIRE(atoms, restart=os.path.join(calc_dir,'qn.pckl')
                        ,logfile=os.path.join(calc_dir,'ase_opt.output'),
                         trajectory=os.path.join(calc_dir,'ab_bilayer.traj'))
        dyn.run(fmax=1e-6)