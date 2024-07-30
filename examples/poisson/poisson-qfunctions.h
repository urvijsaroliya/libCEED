// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
//  Authors: Peter Munch, Martin Kronbichler
//
// ---------------------------------------------------------------------

#include <ceed.h>
#include <deal.II/base/tensor.h>

/**
 * Context passed to libCEED Q-function.
 */
struct BuildContext
{
  CeedInt dim, space_dim;
};

template <int dim>
class Evaluator
{
public:
  Evaluator(const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out)
    :
    Q(Q),
    in(in),
    out(out)
  {}

  void compute(const CeedInt i)
  {
    dealii::Tensor<1, dim> grad = get_gradient(i);
    submit_gradient(grad, i);
    //CeedScalar val = get_value(i);
    //submit_value(val, i);
  }

private:
  dealii::Tensor<1, dim> get_gradient(const CeedInt i) const
  {
    const CeedScalar *ug = in[0];
    dealii::Tensor<1, dim> output;
    for (int d = 0; d < dim; ++d)
      output[d] = ug[i + Q * d];
    return output;
  }

  void submit_gradient(const dealii::Tensor<1, dim> input, const CeedInt i)
  {
    const CeedScalar *gdata = in[1];
    CeedScalar *vg = out[0];
    for (unsigned int d = 0; d < dim; ++d)
      vg[i + d * Q] = input[d] * gdata[i + Q * d];
    for (unsigned int d = 0, c = (dim * (dim + 1)) / 2 - 1; d < dim; ++d)
      for (unsigned int e = d + 1; e < dim; ++e, --c)
        {
          vg[i + e * Q] += input[d] * gdata[i + Q * c];
          vg[i + d * Q] += input[e] * gdata[i + Q * c];
        }
  }

  CeedScalar get_value(const CeedInt i) const
  {
    const CeedScalar *uv = in[1];
    return uv[i];
  }

  void submit_value(const CeedScalar input, const CeedInt i)
  {
    const CeedScalar *gdata = in[2];
    CeedScalar *vv = out[1];
    vv[i] = input * gdata[i + Q * dim * (dim + 1) / 2];
  }

  const CeedInt Q;
  const CeedScalar *const * in;
  CeedScalar *const *out;
};

/**
 * libCEED Q-function for building quadrature data for a Poisson operator
 */
CEED_QFUNCTION(f_build_poisson)
(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out)
{
  BuildContext     *bc = (BuildContext *)ctx;
  const CeedScalar *J = in[0], *w = in[1];
  CeedScalar       *qdata = out[0];

  switch (bc->dim + 10 * bc->space_dim)
    {
      case 11:
        CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
        {
          qdata[i + Q * 0] = w[i] / J[i];
          //qdata[i + Q * 1] = w[i] * J[i];
        }
        break;
      case 22:
        CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
        {
          const CeedScalar J11 = J[i + Q * 0];
          const CeedScalar J21 = J[i + Q * 1];
          const CeedScalar J12 = J[i + Q * 2];
          const CeedScalar J22 = J[i + Q * 3];
          const CeedScalar qw  = w[i] / (J11 * J22 - J21 * J12);
          qdata[i + Q * 0]     = qw * (J12 * J12 + J22 * J22);
          qdata[i + Q * 1]     = qw * (J11 * J11 + J21 * J21);
          qdata[i + Q * 2]     = -qw * (J11 * J12 + J21 * J22);
          //qdata[i + Q * 3]     = w[i] * (J11 * J22 - J21 * J12);
        }
        break;
      case 33:
        CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
        {
          const CeedScalar J11 = J[i + Q * 0];
          const CeedScalar J21 = J[i + Q * 1];
          const CeedScalar J31 = J[i + Q * 2];
          const CeedScalar J12 = J[i + Q * 3];
          const CeedScalar J22 = J[i + Q * 4];
          const CeedScalar J32 = J[i + Q * 5];
          const CeedScalar J13 = J[i + Q * 6];
          const CeedScalar J23 = J[i + Q * 7];
          const CeedScalar J33 = J[i + Q * 8];
          const CeedScalar A11 = J22 * J33 - J23 * J32;
          const CeedScalar A12 = J13 * J32 - J12 * J33;
          const CeedScalar A13 = J12 * J23 - J13 * J22;
          const CeedScalar A21 = J23 * J31 - J21 * J33;
          const CeedScalar A22 = J11 * J33 - J13 * J31;
          const CeedScalar A23 = J13 * J21 - J11 * J23;
          const CeedScalar A31 = J21 * J32 - J22 * J31;
          const CeedScalar A32 = J12 * J31 - J11 * J32;
          const CeedScalar A33 = J11 * J22 - J12 * J21;
          const CeedScalar qw  = w[i] / (J11 * A11 + J21 * A12 + J31 * A13);
          qdata[i + Q * 0]     = qw * (A11 * A11 + A12 * A12 + A13 * A13);
          qdata[i + Q * 1]     = qw * (A21 * A21 + A22 * A22 + A23 * A23);
          qdata[i + Q * 2]     = qw * (A31 * A31 + A32 * A32 + A33 * A33);
          qdata[i + Q * 3]     = qw * (A21 * A31 + A22 * A32 + A23 * A33);
          qdata[i + Q * 4]     = qw * (A11 * A31 + A12 * A32 + A13 * A33);
          qdata[i + Q * 5]     = qw * (A11 * A21 + A12 * A22 + A13 * A23);
          //qdata[i + Q * 6]     = w[i] * (J11 * A11 + J21 * A12 + J31 * A13);
        }
        break;
    }
  return 0;
}



/**
 * libCEED Q-function for applying a Poisson operator
 */
CEED_QFUNCTION(f_apply_poisson)
(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out)
{
  if (0)
    {
    BuildContext     *bc = (BuildContext *)ctx;
    switch (bc->dim + 10 * bc->space_dim)
      {
        case 11:
         {
            Evaluator<1> evaluator(Q, in, out);
            CeedPragmaSIMD for (CeedInt i = 0; i < Q; ++i)
              evaluator.compute(i);
         }
         break;
        case 22:
         {
             Evaluator<2> evaluator(Q, in, out);
             CeedPragmaSIMD for (CeedInt i = 0; i < Q; ++i)
                 evaluator.compute(i);
         }
         break;
        case 33:
         {
             Evaluator<3> evaluator(Q, in, out);
             CeedPragmaSIMD for (CeedInt i = 0; i < Q; ++i)
                 evaluator.compute(i);
         }
         break;
      default:
        AssertThrow(false, dealii::ExcNotImplemented());
      }
    }
  else
    {
      BuildContext     *bc = (BuildContext *)ctx;
      const CeedScalar *ug = in[0], *qdata = in[1];
      CeedScalar       *vg = out[0];

      switch (bc->dim)
        {
        case 1:
          CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
            {
              vg[i] = ug[i] * qdata[i + Q * 0];
            }
          break;
        case 2:
          CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
            {
              const CeedScalar ug0 = ug[i + Q * 0];
              const CeedScalar ug1 = ug[i + Q * 1];
              vg[i + Q * 0]        = qdata[i + Q * 0] * ug0 + qdata[i + Q * 2] * ug1;
              vg[i + Q * 1]        = qdata[i + Q * 2] * ug0 + qdata[i + Q * 1] * ug1;
            }
          break;
        case 3:
          CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
            {
              const CeedScalar ug0 = ug[i + Q * 0];
              const CeedScalar ug1 = ug[i + Q * 1];
              const CeedScalar ug2 = ug[i + Q * 2];
              vg[i + Q * 0] = qdata[i + Q * 0] * ug0 + qdata[i + Q * 5] * ug1 + qdata[i + Q * 4] * ug2;
              vg[i + Q * 1] = qdata[i + Q * 5] * ug0 + qdata[i + Q * 1] * ug1 + qdata[i + Q * 3] * ug2;
              vg[i + Q * 2] = qdata[i + Q * 4] * ug0 + qdata[i + Q * 3] * ug1 + qdata[i + Q * 2] * ug2;
            }
          break;
        }
    }
  return 0;
}



/**
 *libCEED Q-function for applying a vector Poisson operator
 */
CEED_QFUNCTION(f_apply_poisson_vec)
(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out)
{
  BuildContext     *bc = (BuildContext *)ctx;
  const CeedScalar *ug = in[0], *qdata = in[1];
  CeedScalar       *vg = out[0];

  switch (bc->dim)
    {
      case 1:
        CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
        {
          vg[i] = ug[i] * qdata[i];
        }
        break;
      case 2:
        CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
        {
          {
            const CeedScalar ug0      = ug[i + Q * 0 + Q * 2 * 0];
            const CeedScalar ug1      = ug[i + Q * 1 + Q * 2 * 0];
            vg[i + Q * 0 + Q * 2 * 0] = qdata[i + Q * 0] * ug0 + qdata[i + Q * 2] * ug1;
            vg[i + Q * 1 + Q * 2 * 0] = qdata[i + Q * 2] * ug0 + qdata[i + Q * 1] * ug1;
          }
          {
            const CeedScalar ug0      = ug[i + Q * 0 + Q * 2 * 1];
            const CeedScalar ug1      = ug[i + Q * 1 + Q * 2 * 1];
            vg[i + Q * 0 + Q * 2 * 1] = qdata[i + Q * 0] * ug0 + qdata[i + Q * 2] * ug1;
            vg[i + Q * 1 + Q * 2 * 1] = qdata[i + Q * 2] * ug0 + qdata[i + Q * 1] * ug1;
          }
        }
        break;
      case 3:
        CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
        {
          {
            const CeedScalar ug0 = ug[i + Q * 0 + Q * 3 * 0];
            const CeedScalar ug1 = ug[i + Q * 1 + Q * 3 * 0];
            const CeedScalar ug2 = ug[i + Q * 2 + Q * 3 * 0];
            vg[i + Q * 0 + Q * 3 * 0] =
              qdata[i + Q * 0] * ug0 + qdata[i + Q * 5] * ug1 + qdata[i + Q * 4] * ug2;
            vg[i + Q * 1 + Q * 3 * 0] =
              qdata[i + Q * 5] * ug0 + qdata[i + Q * 1] * ug1 + qdata[i + Q * 3] * ug2;
            vg[i + Q * 2 + Q * 3 * 0] =
              qdata[i + Q * 4] * ug0 + qdata[i + Q * 3] * ug1 + qdata[i + Q * 2] * ug2;
          }
          {
            const CeedScalar ug0 = ug[i + Q * 0 + Q * 3 * 1];
            const CeedScalar ug1 = ug[i + Q * 1 + Q * 3 * 1];
            const CeedScalar ug2 = ug[i + Q * 2 + Q * 3 * 1];
            vg[i + Q * 0 + Q * 3 * 1] =
              qdata[i + Q * 0] * ug0 + qdata[i + Q * 5] * ug1 + qdata[i + Q * 4] * ug2;
            vg[i + Q * 1 + Q * 3 * 1] =
              qdata[i + Q * 5] * ug0 + qdata[i + Q * 1] * ug1 + qdata[i + Q * 3] * ug2;
            vg[i + Q * 2 + Q * 3 * 1] =
              qdata[i + Q * 4] * ug0 + qdata[i + Q * 3] * ug1 + qdata[i + Q * 2] * ug2;
          }
          {
            const CeedScalar ug0 = ug[i + Q * 0 + Q * 3 * 2];
            const CeedScalar ug1 = ug[i + Q * 1 + Q * 3 * 2];
            const CeedScalar ug2 = ug[i + Q * 2 + Q * 3 * 2];
            vg[i + Q * 0 + Q * 3 * 2] =
              qdata[i + Q * 0] * ug0 + qdata[i + Q * 5] * ug1 + qdata[i + Q * 4] * ug2;
            vg[i + Q * 1 + Q * 3 * 2] =
              qdata[i + Q * 5] * ug0 + qdata[i + Q * 1] * ug1 + qdata[i + Q * 3] * ug2;
            vg[i + Q * 2 + Q * 3 * 2] =
              qdata[i + Q * 4] * ug0 + qdata[i + Q * 3] * ug1 + qdata[i + Q * 2] * ug2;
          }
        }
        break;
    }
  return 0;
}
