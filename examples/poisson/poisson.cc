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

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

// boost
#include <boost/algorithm/string.hpp>

#include <sstream>

template <int dim> LaplaceProblem<dim>::LaplaceProblem()
#ifdef DEAL_II_WITH_P4EST
    : triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim>::construct_multigrid_hierarchy)
#else
    : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
    , fe(degree_finite_element)
    , dof_handler(triangulation)
    , setup_time(0.)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    // The LaplaceProblem class holds an additional output stream that
    // collects detailed timings about the setup phase. This stream, called
    // time_details, is disabled by default through the @p false argument
    // specified here. For detailed timings, removing the @p false argument
    // prints all the details.
    , time_details(std::cout,
                   false &&
                     Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}


int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_proces (MPI_COMM_WORLD) == 0);
    using Number                     = double;
    using VectorType                 = LinearAlgebra::distributed::Vector<Number>;
    
    // !To set
    const unsigned int dim           = 3;
    const unsigned int fe_degree     = 2;
    // To set!

    const unsigned int n_vect_doubles = LinearAlgebra::distributed::Vector<Number>::size();
    const unsigned int n_vect_bits    = 8 * sizeof(Number) * n_vect_doubles;

    pcout << "Vectorization over " << n_vect_doubles << " doubles = " << n_vect_bits << " bits (" << Utilities::System::get_current_vectorization_level() << ')' << std::endl;

    const unsigned int n_q_points    = (bp <= BPType::BP4) ? (fe_degree + 2) : (fe_degree + 1);
    const unsigned int n_refinements = 9 - dim;
    const unsigned int n_components = 1;

    #ifdef DEAL_II_WITH_P4EST
        parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    #else
        parallel::shared::Triangulation<dim> tria(MPI_COMM_WORLD, ::Triangulation<dim>::none, true);
    #endif
    MappingQ1<dim> mapping;
    QGauss<dim>    quadrature(n_q_points);
    FESystem<dim>  fe(FE_Q<dim>(fe_degree), n_components);
    DoFHandler<dim> dof_handler(tria);
    AffineConstraints<Number> constraints;

    for (unsigned int cycle = 0; cycle < 9 - dim; ++cycle)
    {
        pcout << "Cycle " << cycle << std::endl;

        if (cycle == 0)
        {
            GridGenerator::hyper_cube(triangulation, 0., 1.);
            triangulation.refine_global(3 - dim);
        }
        triangulation.refine_global(1);
        // setup_system();
        // assemble_rhs();
        // solve();
        // output_results(cycle);
        pcout << std::endl;
    };
}
