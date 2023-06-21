/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(reg/dep/poly,PairRegDepPoly);
// clang-format on
#else

#ifndef LMP_PAIR_REG_DEP_POLY_H
#define LMP_PAIR_REG_DEP_POLY_H

#include "pair.h"

namespace LAMMPS_NS {

class PairRegDepPoly : public Pair {
 public:
  PairRegDepPoly(class LAMMPS *);
  //virtual ~PairRegDepPoly();
  ~PairRegDepPoly() override;
  //virtual void compute(int, int);
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  void KC_neigh();
  void calc_normal();
  void calc_FvdW(int, int);
  double single(int, int, int, int, double, double, double, double &) override;

 protected:
  //int me;
  int maxlocal;           // size of numneigh, firstneigh arrays
  int pgsize;             // size of neighbor page
  int oneatom;            // max # of neighbors for one atom
  MyPage<int> *ipage;     // neighbor list pages
  int *KC_numneigh;       // # of pair neighbors for each atom
  int **KC_firstneigh;    // ptr to 1st neighbor of each atom
  int tap_flag;           // flag to turn on/off taper function

  struct Param {
    double delta, C, C0, C2, C4; //constants for registry dependent modification term
    double z0, A6, A8, A10; //constants for polynomial energy portion
    double S; //constant for conveniently scaling 
    double rcut; //cutoff radius
    double delta2inv, z06, z08, z010; //precomputed quantities
    int ielement, jelement;
  };
  Param *params;    // parameter set for I-J interactions
  int nmax;         // max # of atoms

  double cut_global;
  double cut_normal;
  double **cut;
  double **cutKCsq;
  double **offset;
  double **normal;
  double ***dnormdri;
  double ****dnormal;

  void read_file(char *);
  void allocate();

  /* ----Calculate the long-range cutoff term */
  inline double calc_Tap(double r_ij, double Rcut)
  {
    double Tap, r;
    double Tap_coeff[8] = {1.0, 0.0, 0.0, 0.0, -35.0, 84.0, -70.0, 20.0};

    r = r_ij / Rcut;
    if (r >= 1.0) {
      Tap = 0.0;
    } else {
      Tap = Tap_coeff[7] * r + Tap_coeff[6];
      Tap = Tap * r + Tap_coeff[5];
      Tap = Tap * r + Tap_coeff[4];
      Tap = Tap * r + Tap_coeff[3];
      Tap = Tap * r + Tap_coeff[2];
      Tap = Tap * r + Tap_coeff[1];
      Tap = Tap * r + Tap_coeff[0];
    }

    return (Tap);
  }

  /* ----Calculate the derivatives of long-range cutoff term */
  inline double calc_dTap(double r_ij, double Rcut)
  {
    double dTap, r;
    double Tap_coeff[8] = {1.0, 0.0, 0.0, 0.0, -35.0, 84.0, -70.0, 20.0};

    r = r_ij / Rcut;
    if (r >= 1.0) {
      dTap = 0.0;
    } else {
      dTap = 7.0 * Tap_coeff[7] * r + 6.0 * Tap_coeff[6];
      dTap = dTap * r + 5.0 * Tap_coeff[5];
      dTap = dTap * r + 4.0 * Tap_coeff[4];
      dTap = dTap * r + 3.0 * Tap_coeff[3];
      dTap = dTap * r + 2.0 * Tap_coeff[2];
      dTap = dTap * r + Tap_coeff[1];
      dTap = dTap / Rcut;
    }

    return (dTap);
  }
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/