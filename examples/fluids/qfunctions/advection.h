// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Advection initial condition and operator for Navier-Stokes example using PETSc

#ifndef advection_h
#define advection_h

#include <ceed.h>
#include <math.h>

#include "../qfunctions/advection_types.h"
#include "../qfunctions/stabilization_types.h"
#include "utils.h"

typedef struct SetupContextAdv_ *SetupContextAdv;
struct SetupContextAdv_ {
  CeedScalar           rc;
  CeedScalar           lx;
  CeedScalar           ly;
  CeedScalar           lz;
  CeedScalar           wind[3];
  CeedScalar           time;
  WindType             wind_type;
  BubbleType           bubble_type;
  BubbleContinuityType bubble_continuity_type;
};

// *****************************************************************************
// This QFunction sets the initial conditions and the boundary conditions
//   for two test cases: ROTATION and TRANSLATION
//
// -- ROTATION (default)
//      Initial Conditions:
//        Mass Density:
//          Constant mass density of 1.0
//        Momentum Density:
//          Rotational field in x,y
//        Energy Density:
//          Maximum of 1. x0 decreasing linearly to 0. as radial distance
//            increases to (1.-r/rc), then 0. everywhere else
//
//      Boundary Conditions:
//        Mass Density:
//          0.0 flux
//        Momentum Density:
//          0.0
//        Energy Density:
//          0.0 flux
//
// -- TRANSLATION
//      Initial Conditions:
//        Mass Density:
//          Constant mass density of 1.0
//        Momentum Density:
//           Constant rectilinear field in x,y
//        Energy Density:
//          Maximum of 1. x0 decreasing linearly to 0. as radial distance
//            increases to (1.-r/rc), then 0. everywhere else
//
//      Boundary Conditions:
//        Mass Density:
//          0.0 flux
//        Momentum Density:
//          0.0
//        Energy Density:
//          Inflow BCs:
//            E = E_wind
//          Outflow BCs:
//            E = E(boundary)
//          Both In/Outflow BCs for E are applied weakly in the
//            QFunction "Advection_Sur"
//
// *****************************************************************************

// *****************************************************************************
// This helper function provides support for the exact, time-dependent solution (currently not implemented) and IC formulation for 3D advection
// *****************************************************************************
CEED_QFUNCTION_HELPER CeedInt Exact_Advection(CeedInt dim, CeedScalar time, const CeedScalar X[], CeedInt Nf, CeedScalar q[], void *ctx) {
  const SetupContextAdv context = (SetupContextAdv)ctx;
  const CeedScalar      rc      = context->rc;
  const CeedScalar      lx      = context->lx;
  const CeedScalar      ly      = context->ly;
  const CeedScalar      lz      = context->lz;
  const CeedScalar     *wind    = context->wind;

  // Setup
  const CeedScalar x0[3]     = {0.25 * lx, 0.5 * ly, 0.5 * lz};
  const CeedScalar center[3] = {0.5 * lx, 0.5 * ly, 0.5 * lz};

  // -- Coordinates
  const CeedScalar x = X[0];
  const CeedScalar y = X[1];
  const CeedScalar z = X[2];

  // -- Energy
  CeedScalar r = 0.;
  switch (context->bubble_type) {
    case BUBBLE_SPHERE: {  // (dim=3)
      r = sqrt(Square(x - x0[0]) + Square(y - x0[1]) + Square(z - x0[2]));
    } break;
    case BUBBLE_CYLINDER: {  // (dim=2)
      r = sqrt(Square(x - x0[0]) + Square(y - x0[1]));
    } break;
  }

  // Initial Conditions
  switch (context->wind_type) {
    case WIND_ROTATION:
      q[0] = 1.;
      q[1] = -(y - center[1]);
      q[2] = (x - center[0]);
      q[3] = 0;
      break;
    case WIND_TRANSLATION:
      q[0] = 1.;
      q[1] = wind[0];
      q[2] = wind[1];
      q[3] = wind[2];
      break;
  }

  switch (context->bubble_continuity_type) {
    // original continuous, smooth shape
    case BUBBLE_CONTINUITY_SMOOTH: {
      q[4] = r <= rc ? (1. - r / rc) : 0.;
    } break;
    // discontinuous, sharp back half shape
    case BUBBLE_CONTINUITY_BACK_SHARP: {
      q[4] = ((r <= rc) && (y < center[1])) ? (1. - r / rc) : 0.;
    } break;
    // attempt to define a finite thickness that will get resolved under grid refinement
    case BUBBLE_CONTINUITY_THICK: {
      q[4] = ((r <= rc) && (y < center[1])) ? (1. - r / rc) * fmin(1.0, (center[1] - y) / 1.25) : 0.;
    } break;
  }
  return 0;
}

// *****************************************************************************
// This QFunction sets the initial conditions for 3D advection
// *****************************************************************************
CEED_QFUNCTION(ICsAdvection)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  // Outputs
  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[]  = {X[0][i], X[1][i], X[2][i]};
    CeedScalar       q[5] = {0.};

    Exact_Advection(3, 0., x, 5, q, ctx);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }  // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of the advection equation
//
// This is 3D advection given in two formulations based upon the weak form.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density    ,  Ui = rho ui
//   E   - Total Energy Density
//
// Advection Equation:
//   dE/dt + div( E u ) = 0
// *****************************************************************************
CEED_QFUNCTION(Advection)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1];
  const CeedScalar(*q_data)            = in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  // Context
  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar CtauS       = context->CtauS;
  const CeedScalar strong_form = context->strong_form;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho  = q[0][i];
    const CeedScalar u[3] = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E    = q[4][i];
    // -- Grad in
    const CeedScalar drho[3]  = {dq[0][0][i], dq[1][0][i], dq[2][0][i]};
    const CeedScalar du[3][3] = {
        {(dq[0][1][i] - drho[0] * u[0]) / rho, (dq[1][1][i] - drho[1] * u[0]) / rho, (dq[2][1][i] - drho[2] * u[0]) / rho},
        {(dq[0][2][i] - drho[0] * u[1]) / rho, (dq[1][2][i] - drho[1] * u[1]) / rho, (dq[2][2][i] - drho[2] * u[1]) / rho},
        {(dq[0][3][i] - drho[0] * u[2]) / rho, (dq[1][3][i] - drho[1] * u[2]) / rho, (dq[2][3][i] - drho[2] * u[2]) / rho}
    };
    const CeedScalar dE[3] = {dq[0][4][i], dq[1][4][i], dq[2][4][i]};
    CeedScalar       wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    // The Physics
    // Note with the order that du was filled and the order that dXdx was filled
    //   du[j][k]= du_j / dX_K    (note cap K to be clear this is u_{j,xi_k})
    //   dXdx[k][j] = dX_K / dx_j
    //   X_K=Kth reference element coordinate (note cap X and K instead of xi_k}
    //   x_j and u_j are jth  physical position and velocity components

    // No Change in density or momentum
    for (CeedInt f = 0; f < 4; f++) {
      for (CeedInt j = 0; j < 3; j++) dv[j][f][i] = 0;
      v[f][i] = 0;
    }

    // -- Total Energy
    // Evaluate the strong form using div(E u) = u . grad(E) + E div(u)
    // or in index notation: (u_j E)_{,j} = u_j E_j + E u_{j,j}
    CeedScalar div_u = 0, u_dot_grad_E = 0;
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dEdx_j = 0;
      for (CeedInt k = 0; k < 3; k++) {
        div_u += du[j][k] * dXdx[k][j];  // u_{j,j} = u_{j,K} X_{K,j}
        dEdx_j += dE[k] * dXdx[k][j];
      }
      u_dot_grad_E += u[j] * dEdx_j;
    }
    CeedScalar strong_conv = E * div_u + u_dot_grad_E;

    // Weak Galerkin convection term: dv \cdot (E u)
    for (CeedInt j = 0; j < 3; j++) dv[j][4][i] = (1 - strong_form) * wdetJ * E * (u[0] * dXdx[j][0] + u[1] * dXdx[j][1] + u[2] * dXdx[j][2]);
    v[4][i] = 0;

    // Strong Galerkin convection term: - v div(E u)
    v[4][i] = -strong_form * wdetJ * strong_conv;

    // Stabilization requires a measure of element transit time in the velocity
    //   field u.
    CeedScalar uX[3];
    for (CeedInt j = 0; j < 3; j++) uX[j] = dXdx[j][0] * u[0] + dXdx[j][1] * u[1] + dXdx[j][2] * u[2];
    const CeedScalar TauS = CtauS / sqrt(uX[0] * uX[0] + uX[1] * uX[1] + uX[2] * uX[2]);
    for (CeedInt j = 0; j < 3; j++) dv[j][4][i] -= wdetJ * TauS * strong_conv * uX[j];
  }  // End Quadrature Point Loop

  return 0;
}

// *****************************************************************************
// This QFunction implements 3D (mentioned above) with implicit time stepping method
// *****************************************************************************
CEED_QFUNCTION(IFunction_Advection)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1];
  const CeedScalar(*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*q_data)            = in[3];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  CeedScalar *jac_data           = out[2];

  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar CtauS       = context->CtauS;
  const CeedScalar strong_form = context->strong_form;
  const CeedScalar zeros[14]   = {0.};

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho  = q[0][i];
    const CeedScalar u[3] = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E    = q[4][i];
    // -- Grad in
    const CeedScalar drho[3]  = {dq[0][0][i], dq[1][0][i], dq[2][0][i]};
    const CeedScalar du[3][3] = {
        {(dq[0][1][i] - drho[0] * u[0]) / rho, (dq[1][1][i] - drho[1] * u[0]) / rho, (dq[2][1][i] - drho[2] * u[0]) / rho},
        {(dq[0][2][i] - drho[0] * u[1]) / rho, (dq[1][2][i] - drho[1] * u[1]) / rho, (dq[2][2][i] - drho[2] * u[1]) / rho},
        {(dq[0][3][i] - drho[0] * u[2]) / rho, (dq[1][3][i] - drho[1] * u[2]) / rho, (dq[2][3][i] - drho[2] * u[2]) / rho}
    };
    const CeedScalar dE[3] = {dq[0][4][i], dq[1][4][i], dq[2][4][i]};
    CeedScalar       wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    // The Physics
    // Note with the order that du was filled and the order that dXdx was filled
    //   du[j][k]= du_j / dX_K    (note cap K to be clear this is u_{j,xi_k} )
    //   dXdx[k][j] = dX_K / dx_j
    //   X_K=Kth reference element coordinate (note cap X and K instead of xi_k}
    //   x_j and u_j are jth  physical position and velocity components

    // No Change in density or momentum
    for (CeedInt f = 0; f < 4; f++) {
      for (CeedInt j = 0; j < 3; j++) dv[j][f][i] = 0;
      v[f][i] = wdetJ * q_dot[f][i];  // K Mass/transient term
    }

    // -- Total Energy
    // Evaluate the strong form using div(E u) = u . grad(E) + E div(u)
    //   or in index notation: (u_j E)_{,j} = u_j E_j + E u_{j,j}
    CeedScalar div_u = 0, u_dot_grad_E = 0;
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dEdx_j = 0;
      for (CeedInt k = 0; k < 3; k++) {
        div_u += du[j][k] * dXdx[k][j];  // u_{j,j} = u_{j,K} X_{K,j}
        dEdx_j += dE[k] * dXdx[k][j];
      }
      u_dot_grad_E += u[j] * dEdx_j;
    }
    CeedScalar strong_conv = E * div_u + u_dot_grad_E;
    CeedScalar strong_res  = q_dot[4][i] + strong_conv;

    v[4][i] = wdetJ * q_dot[4][i];  // transient part (ALWAYS)

    // Weak Galerkin convection term: -dv \cdot (E u)
    for (CeedInt j = 0; j < 3; j++) dv[j][4][i] = -wdetJ * (1 - strong_form) * E * (u[0] * dXdx[j][0] + u[1] * dXdx[j][1] + u[2] * dXdx[j][2]);

    // Strong Galerkin convection term: v div(E u)
    v[4][i] += wdetJ * strong_form * strong_conv;

    // Stabilization requires a measure of element transit time in the velocity
    //   field u.
    CeedScalar uX[3];
    for (CeedInt j = 0; j < 3; j++) uX[j] = dXdx[j][0] * u[0] + dXdx[j][1] * u[1] + dXdx[j][2] * u[2];
    const CeedScalar TauS = CtauS / sqrt(uX[0] * uX[0] + uX[1] * uX[1] + uX[2] * uX[2]);

    for (CeedInt j = 0; j < 3; j++) switch (context->stabilization) {
        case STAB_NONE:
          break;
        case STAB_SU:
          dv[j][4][i] += wdetJ * TauS * strong_conv * uX[j];
          break;
        case STAB_SUPG:
          dv[j][4][i] += wdetJ * TauS * strong_res * uX[j];
          break;
      }
    StoredValuesPack(Q, i, 0, 14, zeros, jac_data);
  }  // End Quadrature Point Loop

  return 0;
}

// *****************************************************************************
// This QFunction implements consistent outflow and inflow BCs
//      for 3D advection
//
//  Inflow and outflow faces are determined based on sign(dot(wind, normal)):
//    sign(dot(wind, normal)) > 0 : outflow BCs
//    sign(dot(wind, normal)) < 0 : inflow BCs
//
//  Outflow BCs:
//    The validity of the weak form of the governing equations is extended to the outflow and the current values of E are applied.
//
//  Inflow BCs:
//    A prescribed Total Energy (E_wind) is applied weakly.
// *****************************************************************************
CEED_QFUNCTION(Advection_InOutFlow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)    = in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]   = (CeedScalar(*)[CEED_Q_VLA])out[0];
  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar E_wind      = context->E_wind;
  const CeedScalar strong_form = context->strong_form;
  const bool       is_implicit = context->implicit;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho  = q[0][i];
    const CeedScalar u[3] = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E    = q[4][i];

    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    // Normal velocity
    const CeedScalar u_normal = norm[0] * u[0] + norm[1] * u[1] + norm[2] * u[2];

    // No Change in density or momentum
    for (CeedInt j = 0; j < 4; j++) {
      v[j][i] = 0;
    }
    // Implementing in/outflow BCs
    if (u_normal > 0) {  // outflow
      v[4][i] = -(1 - strong_form) * wdetJb * E * u_normal;
    } else {  // inflow
      v[4][i] = -(1 - strong_form) * wdetJb * E_wind * u_normal;
    }
  }  // End Quadrature Point Loop
  return 0;
}
// *****************************************************************************

#endif  // advection_h
