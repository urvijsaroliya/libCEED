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

// deal.II includes
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

// boost
#include <boost/algorithm/string.hpp>

#include <sstream>

// include operators
#include "poisson.h"

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // Init parameters
  using Number                     = double;
  using VectorType                 = LinearAlgebra::distributed::Vector<Number>;
  const unsigned int dim           = 3;
  const unsigned int fe_degree     = 2;
  const unsigned int n_q_points    = fe_degree + 1;
  const unsigned int n_components  = 1;
  std::string  libCEED_resource    = "/cpu/self/avx/blocked";

#ifdef DEAL_II_WITH_P4EST
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
#else
  parallel::shared::Triangulation<dim> tria(MPI_COMM_WORLD, ::Triangulation<dim>::none, true);
#endif

  // create mapping, quadrature, fe, mesh, ...
  MappingQ1<dim> mapping;
  QGauss<dim>    quadrature(n_q_points);
  FESystem<dim>  fe(FE_Q<dim>(fe_degree), n_components);
  DoFHandler<dim> dof_handler(tria);
  AffineConstraints<Number> constraints;

  for (unsigned int cycle = 0; cycle < 9 - dim; ++cycle)
  {
    pout << "Cycle " << cycle << std::endl;
    if (cycle == 0)
    {
      GridGenerator::hyper_cube(tria, 0., 1.);
      tria.refine_global(3 - dim);
    }
    tria.refine_global(1);
    dof_handler.distribute_dofs(fe);
    pout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    constraints.clear();
    constraints.reinit(dof_handler.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
    constraints.close();
    DoFRenumbering::support_point_wise(dof_handler);
    
    const auto test = [&](const std::string &label, const auto &op) {
      (void)label;

      // initialize vector
      VectorType u, v;
      op.initialize_dof_vector(u);
      op.initialize_dof_vector(v);
      u = 1.0;

      constraints.set_zero(u);

      // perform matrix-vector product
      op.vmult(v, u);

      // create solver
      ReductionControl reduction_control(100, 1e-20, 1e-6);
      // Modify the first argument for n_iter

      // create preconditioner
      DiagonalMatrix<VectorType> diagonal_matrix;
      op.compute_inverse_diagonal(diagonal_matrix.get_vector());

      std::chrono::time_point<std::chrono::system_clock> now;
      try
      {
        // solve problem
        SolverCG<VectorType> solver(reduction_control);
        now = std::chrono::system_clock::now();
        solver.solve(op, v, u, diagonal_matrix);
      }
      catch (const SolverControl::NoConvergence &)
      {
        pout << "Error: solver failed to converge with" << std::endl;
      }
      const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - now).count() / 1e9;

      pout << label << ": " << reduction_control.last_step() << " " << v.l2_norm() << " " 
            << time << std::endl;
    };

    // create and test the libCEED operator
    OperatorCeed<dim, Number> op_ceed(mapping, dof_handler, constraints, quadrature, libCEED_resource);
    test("ceed", op_ceed);

    // create and test a native deal.II operator
    OperatorDealii<dim, Number> op_dealii(mapping, dof_handler, constraints, quadrature);
    test("dealii", op_dealii);
    
  };
}
