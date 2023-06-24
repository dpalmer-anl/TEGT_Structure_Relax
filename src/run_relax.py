# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 13:29:11 2023

@author: danpa
"""

import numpy as np
import argparse
import flatgraphene as fg
import optimizer
import TEGT_calculator
import os

def get_top_layer(atoms):
    xyz=atoms.positions
    
    tags=np.array(atoms.get_chemical_symbols())
    top_pos_ind=np.where(tags!=tags[0])[0]
    top_pos=xyz[top_pos_ind,:]
    
    return top_pos,top_pos_ind

def get_bottom_layer(atoms):
    xyz=atoms.positions
    
    tags=np.array(atoms.get_chemical_symbols())
    bot_pos_ind=np.where(tags==tags[0])[0]
    bot_pos=xyz[bot_pos_ind,:]
    
    return bot_pos,bot_pos_ind

def get_recip_cell(cell):
    periodicR1 = cell[0,:]
    periodicR2 = cell[1,:]
    periodicR3 = cell[2,:]
    V = np.dot(periodicR1,np.cross(periodicR2,periodicR3))
    b1 = 2*np.pi*np.cross(periodicR2,periodicR3)/V
    b2 = 2*np.pi*np.cross(periodicR3,periodicR1)/V
    b3 = 2*np.pi*np.cross(periodicR1,periodicR2)/V
    return np.stack((b1,b2,b3),axis=0)

def get_twist_geom(t,a=2.46):
    sep=3.35
    p_found, q_found, theta_comp = fg.twist.find_p_q(t)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=a,sym=["B","Ti"],
                                        mass=[12.01,12.02],sep=sep,h_vac=3)
    pos = atoms.positions
    recip_cell = get_recip_cell(atoms.get_cell())
    vecs = np.array([recip_cell[0,:2],recip_cell[1,:2],recip_cell[0,:2]-recip_cell[1,:2]])
    top_pos,top_pos_ind =  get_top_layer(atoms)
    bot_pos,bot_pos_ind = get_bottom_layer(atoms)
    for i in top_pos_ind:
        pos[i,2] += np.sum(np.cos(np.dot(vecs, pos[i,:2])))
    for j in bot_pos_ind:
        pos[j,2] -= np.sum(np.cos(np.dot(vecs, pos[j,:2])))

    atoms.set_positions(pos)
        
    return atoms,top_pos_ind
if __name__=="__main__":

    #import matplotlib.pyplot as plt
    # atoms,top_pos_ind = get_twist_geom(2.88)
    # pos = atoms.positions[top_pos_ind,:]
    # cell = atoms.get_cell()
    # plt.scatter(pos[:,0],pos[:,1],c=pos[:,2])
    # line = np.linspace(0,1,10)
    # plt.plot(cell[0,0]*line,cell[0,1]*line)
    # plt.plot(cell[1,0]*line,cell[1,1]*line)
    # plt.colorbar()
    # plt.show()
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--theta',  type=str,default=False)
    args = parser.parse_args()
    
    theta_str = args.theta.replace(".","_")
    atoms = get_twist_geom(args.theta)
    model_dict = dict({"tight binding parameters":"Popov Van Alsenoy",
                          "orbitals":"pz",
                          "nkp":225,
                          "intralayer potential":"Pz rebo",
                          "interlayer potential":"Pz KC inspired",
                          'use mpi':True,
                          'label':theta_str})

        
    calc = TEGT_calculator.TEGT_Calc(model_dict)
    atoms.calc = calc
    calc_dir = calc.calc_dir
    dyn = optimizer.TEGT_FIRE(atoms, restart=os.path.join(calc_dir,'qn.pckl')
                    ,logfile=os.path.join(calc_dir,'ase_opt.output'),
                      trajectory=os.path.join(calc_dir,'tblg'+theta_str+'.traj'))
    dyn.run(fmax=1e-4)