# To be used with the latte-lib input file.  

units		metal
atom_style	full
atom_modify    sort 0 0.0  # This is to avoid sorting the coordinates
box tilt large
read_data AA.data
group top type 1
group bottom type 2

mass 1 12.0100
mass 2 12.0100

velocity	all create 0.0 87287 loop geom
# Interaction potential for carbon atoms
######################## Potential defition ########################
pair_style       hybrid/overlay reg/dep/poly 10.0 0 table linear 500
#pair_coeff	1 1 table ../nam_koshino_c-c.table NamKoshinoPair_C
pair_coeff 	1 1 table ../../porezag_correction/porezag_c-c.table POREZAG_C
pair_coeff       * *   reg/dep/poly  KC_insp.txt   C C # long-range #need to add in KC correction here
#pair_coeff      2 2 table ../nam_koshino_c-c.table NamKoshinoPair_C
pair_coeff	2 2 table ../../porezag_correction/porezag_c-c.table POREZAG_C
####################################################################

timestep 0.00025

compute intra all pair table
compute 0 all pair reg/dep/poly
variable Evdw  equal c_0[1]
variable Erep  equal c_0[2]

variable latteE equal "(ke + f_2)"
variable kinE equal "ke"
variable potE equal "f_2"

thermo 1
thermo_style   custom step pe ke etotal temp epair v_Erep v_Evdw c_intra v_latteE
log log.test
fix		1 all nve

fix   2 all latte NULL
dump           mydump2 all custom 1 dump.aa_latte id type x y z fx fy fz


run_style verlet
run 10
#min_style fire
#minimize       1e-13 1e-5 3000 10000000
