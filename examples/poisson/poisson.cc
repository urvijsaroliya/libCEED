
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "poisson.h"

// #define USE_STD_SIMD

#ifdef USE_STD_SIMD
#  include <experimental/simd>
#endif

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

#include "common_code/curved_manifold.h"

using namespace dealii;

#include "common_code/create_triangulation.h"

// VERSION:
//   0: p:d:t + vectorized over elements;
//   1: p:f:t + vectorized within element (not optimized)
#define VERSION 0

#if VERSION == 0

#  ifdef USE_STD_SIMD
typedef std::experimental::native_simd<double> VectorizedArrayType;
#  else
typedef dealii::VectorizedArray<double> VectorizedArrayType;
#  endif

#elif VERSION == 1
typedef dealii::VectorizedArray<double, 1> VectorizedArrayType;
#endif

#define USE_SHMEM
#define SHOW_VARIANTS


template <int dim>
class QuadraticFunction : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int) const override
  {
    double result = 0;
    for (unsigned int d = 0; d < dim; ++d)
      result += p[d] * p[d];
    return result;
  }
};


template <int dim>
void
test(const unsigned int fe_degree,
     const unsigned int s,
     const MPI_Comm    &comm_shmem)
{
#ifndef USE_SHMEM
  (void)comm_shmem;
#endif

  warmup_code();
  const unsigned int n_q_points = fe_degree + 1;

  deallog.depth_console(0);

  Timer           time;
  MyManifold<dim> manifold;

  const auto tria = create_triangulation(s, manifold, false);

  using Number = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  // std::string  libCEED_resource = "/gpu/cuda/ref";
  std::string  libCEED_resource = "/cpu/self/ref/blocked";
  FE_Q<dim>            fe_q(fe_degree);
  MappingQGeneric<dim> mapping(1); // tri-linear mapping
  DoFHandler<dim>      dof_handler(*tria);
  QGauss<dim>    quadrature(n_q_points);
  AffineConstraints<Number> constraints;

  dof_handler.distribute_dofs(fe_q);
  constraints.clear();
  constraints.reinit(dof_handler.locally_owned_dofs(), DoFTools::extract_locally_relevant_dofs(dof_handler));
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();
  DoFRenumbering::support_point_wise(dof_handler);

  const auto test_ceed = [&](const std::string &label, const auto &op) {
      (void)label;

      // initialize vector
      VectorType u_exact, rhs, u_iterative;
      op.initialize_dof_vector(u_exact);
      op.initialize_dof_vector(rhs);
      op.initialize_dof_vector(u_iterative);
      VectorTools::interpolate(mapping, dof_handler, QuadraticFunction<dim>(), u_exact);
      u_exact.update_ghost_values();

      {
        FEValues<dim> fe_values(mapping, fe_q, QGauss<dim>(fe_q.degree + 1),
                                update_values | update_quadrature_points |
                                update_JxW_values | update_gradients);
        std::vector<types::global_dof_index> dof_indices(fe_q.dofs_per_cell);
        Vector<double> local_vector(fe_q.dofs_per_cell);
        std::vector<Tensor<1, dim>> grads_q_points(fe_values.n_quadrature_points);
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dof_indices);
              fe_values.reinit(cell);
              std::fill(grads_q_points.begin(), grads_q_points.end(), Tensor<1, dim>());
              for (unsigned int i = 0; i < fe_q.dofs_per_cell; ++i)
                {
                  double local_value = constraints.is_constrained(dof_indices[i]) ?
                    u_exact(dof_indices[i]) : 0;
                  for (const unsigned int q : fe_values.quadrature_point_indices())
                    grads_q_points[q] += local_value * fe_values.shape_grad(i, q);
                }
              for (unsigned int i = 0; i < fe_q.dofs_per_cell; ++i)
                {
                  double sum = 0;
                  for (const unsigned int q : fe_values.quadrature_point_indices())
                    sum += (-fe_values.shape_grad(i, q) * grads_q_points[q]
                            -
                            fe_values.shape_value(i, q) * 2 * dim) * fe_values.JxW(q);
                  local_vector[i] = sum;
                }
              constraints.distribute_local_to_global(local_vector, dof_indices, rhs);
            }
        rhs.compress(VectorOperation::add);
      }

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
        solver.solve(op, u_iterative, rhs, diagonal_matrix);
      }
      catch (const SolverControl::NoConvergence &)
      {
        // std::cout << "Error: solver failed to converge with" << std::endl;
        std::cout << "";
      }

      const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - now).count() / 1e9;
      for (unsigned int i = 0; i < u_iterative.locally_owned_size(); ++i)
        if (constraints.is_constrained(u_iterative.get_partitioner()->local_to_global(i)))
          u_iterative.local_element(i) = u_exact.local_element(i);

      if (false)
        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          data_out.add_data_vector(u_exact, "exact");
          data_out.add_data_vector(u_iterative, "iterative");
          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells = true;
          data_out.set_flags(flags);
          data_out.build_patches(mapping, fe_q.degree);
          std::ofstream file("output-" + std::to_string(s) + ".vtu");
          data_out.write_vtu(file);
        }

      u_iterative -= u_exact;

      std::cout << std::setw(2) << fe_degree << " | " << std::setw(2) << n_q_points   //
              << " |" << std::setw(10) << tria->n_global_active_cells()             //
              << " |" << std::setw(11) << dof_handler.n_dofs()                      //
              << " |" << std::setw(4) << reduction_control.last_step()
              << " | " << std::setw(11) << time / reduction_control.last_step() //
              << " | " << std::setw(11)
              << dof_handler.n_dofs() / time * reduction_control.last_step()
              << " | " << std::setw(11)
              << u_iterative.l2_norm() / u_exact.l2_norm() << std::endl;
      // std::cout
      //     << " p |  q | n_element |     n_dofs |     time/it |   dofs/s/it | mer_time/it | opt_time/it | itCG | time/matvec"
      //     << std::endl;
    };

    // create and test the libCEED operator
    OperatorCeed<dim, Number> op_ceed(mapping, dof_handler, constraints, quadrature, libCEED_resource);
    test_ceed("ceed", op_ceed);
}


template <int dim>
void
do_test(const unsigned int fe_degree, const int s_in)
{
  MPI_Comm comm_shmem = MPI_COMM_SELF;

#ifdef USE_SHMEM
  MPI_Comm_split_type(MPI_COMM_WORLD,
                      MPI_COMM_TYPE_SHARED,
                      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                      MPI_INFO_NULL,
                      &comm_shmem);
#endif

  if (s_in < 1)
    {
      unsigned int s =
        std::max(3U,
                 static_cast<unsigned int>(std::log2(1024 / fe_degree / fe_degree / fe_degree)));
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout
          << " p |  q | n_element |     n_dofs | #it |     time/it |   dofs/s/it |   l2_error"
          << std::endl;
      while ((2 + Utilities::fixed_power<dim>(fe_degree + 1)) * (1UL << (s / 4)) <
             6000000ULL * Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        {
          test<dim>(fe_degree, s, comm_shmem);
          ++s;
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << std::endl << std::endl;
    }
  else
    test<dim>(fe_degree, s_in, comm_shmem);

#ifdef USE_SHMEM
  MPI_Comm_free(&comm_shmem);
#endif
}


int
main(int argc, char **argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  unsigned int degree         = 1;
  unsigned int s              = -1;
  bool         compact_output = true;
  if (argc > 1)
    degree = std::atoi(argv[1]);
  if (argc > 2)
    s = std::atoi(argv[2]);

  do_test<3>(degree, s);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
