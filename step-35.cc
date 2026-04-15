/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2009 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Author: Abner Salgado, Texas A&M University 2009
 */


// @sect3{Include files}

// We start by including all the necessary deal.II header files and some C++
// related ones. Each one of them has been discussed in previous tutorial
// programs, so we will not get into details here.
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>

// FSI: Particle handling (from step-70)
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/generators.h>
#include <deal.II/particles/data_out.h>

// Finally this is as in all previous programs:
namespace Step35
{
  using namespace dealii;

  // @sect3{Run time parameters}
  //
  // Since our method has several parameters that can be fine-tuned we put them
  // into an external file, so that they can be determined at run-time.
  //
  // This includes, in particular, the formulation of the equation for the
  // auxiliary variable $\phi$, for which we declare an <code>enum</code>. Next,
  // we declare a class that is going to read and store all the parameters that
  // our program needs to run.
  namespace RunTimeParameters
  {
    enum class Method
    {
      standard,
      rotational
    };

    class Data_Storage
    {
    public:
      Data_Storage();

      void read_data(const std::string &filename);

      Method form;

      double dt;
      double initial_time;
      double final_time;

      double Reynolds;

      unsigned int n_global_refines;

      unsigned int pressure_degree;

      unsigned int vel_max_iterations;
      unsigned int vel_Krylov_size;
      unsigned int vel_off_diagonals;
      unsigned int vel_update_prec;
      double       vel_eps;
      double       vel_diag_strength;

      bool         verbose;
      unsigned int output_interval;

      // === Additional FSI parameters ===
      std::string  mesh_filename;         // Mesh file name (default: nsbench2.inp)
      bool         use_nitsche;           // Enable/disable Nitsche method
      double       nitsche_beta;          // Nitsche penalty coefficient β
      unsigned int solid_refinement;      // Solid mesh refinement level
      double       solid_angular_velocity;// Solid angular velocity ω
      Point<2>     solid_center;          // Solid center coordinates
      double       solid_radius;          // Solid radius
      
      // Additional solid geometry and motion parameters (from step-35-ibm.cc)
      std::string  solid_geometry_type;   // "circle" or "rectangle"
      double       solid_width;           // Rectangle width
      double       solid_height;          // Rectangle height
      std::string  motion_type;           // "static", "prescribed", "fsi_coupled"
      double       amplitude_x;           // Prescribed motion amplitude X
      double       amplitude_y;           // Prescribed motion amplitude Y
      double       frequency;             // Prescribed motion frequency
      double       rotation_speed;        // Prescribed rotation speed
      double       solid_density;         // Solid density for FSI coupling
      double       relaxation_factor;     // IBM relaxation factor

      // === Mesh generation parameters ===
      std::string  mesh_generation_type;  // "file" or "channel"
      double       channel_length;        // Channel length (for 2D/3D)
      double       channel_width;         // Channel width (for 2D/3D)
      double       channel_height;        // Channel height (for 3D only)
      unsigned int channel_refinement;    // Initial refinement for channel
      unsigned int channel_cells_x;       // Number of cells in x-direction
      unsigned int channel_cells_y;       // Number of cells in y-direction
      unsigned int channel_cells_z;       // Number of cells in z-direction (3D only)

      // === Boundary condition parameters ===
      // For each boundary ID (1-6), we store boundary condition type
      // Types: "zero" (default), "constant", "parabolic", "function", "free_slip", "symmetry", "outflow"
      std::vector<std::string> boundary_types;      // Type for each boundary ID (1-6)
      std::vector<double>      boundary_values_x;   // X-component value for constant BC
      std::vector<double>      boundary_values_y;   // Y-component value for constant BC
      std::vector<double>      boundary_max_vel;    // Max velocity for parabolic inlet

      // === Initial condition parameters ===
      std::string  initial_velocity_type;  // "zero", "constant", or "parabolic"
      double       initial_velocity_x;     // X-component for constant initial velocity
      double       initial_velocity_y;     // Y-component for constant initial velocity
      double       initial_max_velocity;   // Max velocity for parabolic initial profile

      // === Sponge layer parameters ===
      bool         use_sponge_layer;       // Enable/disable sponge layer damping
      double       sponge_x_start;         // X-coordinate where sponge layer starts
      double       sponge_strength;        // Maximum damping coefficient σ_max
      unsigned int sponge_order;           // Order of damping profile (1=linear, 2=quadratic, 3=cubic)

    protected:
      ParameterHandler prm;
    };

    // In the constructor of this class we declare all the parameters. The
    // details of how this works have been discussed elsewhere, for example in
    // step-29.
    Data_Storage::Data_Storage()
      : form(Method::rotational)
      , dt(5e-4)
      , initial_time(0.)
      , final_time(1.)
      , Reynolds(1.)
      , n_global_refines(0)
      , pressure_degree(1)
      , vel_max_iterations(1000)
      , vel_Krylov_size(30)
      , vel_off_diagonals(60)
      , vel_update_prec(15)
      , vel_eps(1e-12)
      , vel_diag_strength(0.01)
      , verbose(true)
      , output_interval(15)
      , mesh_filename("nsbench2.inp")
      , use_nitsche(false)
      , nitsche_beta(100.0)
      , solid_refinement(3)
      , solid_angular_velocity(6.2831853)
      , solid_center(Point<2>(0.0, 0.0))
      , solid_radius(0.1)
      , solid_geometry_type("circle")
      , solid_width(0.5)
      , solid_height(0.3)
      , motion_type("static")
      , amplitude_x(0.0)
      , amplitude_y(0.0)
      , frequency(1.0)
      , rotation_speed(0.0)
      , solid_density(1.0)
      , relaxation_factor(1.0)
      , mesh_generation_type("file")
      , channel_length(10.0)
      , channel_width(4.1)
      , channel_height(1.0)
      , channel_refinement(0)
      , channel_cells_x(1)
      , channel_cells_y(1)
      , channel_cells_z(1)
      , initial_velocity_type("parabolic")
      , initial_velocity_x(0.0)
      , initial_velocity_y(0.0)
      , initial_max_velocity(1.5)
      , use_sponge_layer(false)
      , sponge_x_start(20.0)
      , sponge_strength(10.0)
      , sponge_order(2)
    {
      prm.declare_entry("Method_Form",
                        "rotational",
                        Patterns::Selection("rotational|standard"),
                        " Used to select the type of method that we are going "
                        "to use. ");
      prm.enter_subsection("Physical data");
      {
        prm.declare_entry("initial_time",
                          "0.",
                          Patterns::Double(0.),
                          " The initial time of the simulation. ");
        prm.declare_entry("final_time",
                          "1.",
                          Patterns::Double(0.),
                          " The final time of the simulation. ");
        prm.declare_entry("Reynolds",
                          "1.",
                          Patterns::Double(0.),
                          " The Reynolds number. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time step data");
      {
        prm.declare_entry("dt",
                          "5e-4",
                          Patterns::Double(0.),
                          " The time step size. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization");
      {
        prm.declare_entry("n_of_refines",
                          "0",
                          Patterns::Integer(0, 15),
                          " The number of global refines we do on the mesh. ");
        prm.declare_entry("pressure_fe_degree",
                          "1",
                          Patterns::Integer(1, 5),
                          " The polynomial degree for the pressure space. ");
        prm.declare_entry("mesh_filename",
                          "nsbench2.inp",
                          Patterns::FileName(),
                          " The mesh file for the fluid domain. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Data solve velocity");
      {
        prm.declare_entry(
          "max_iterations",
          "1000",
          Patterns::Integer(1, 1000),
          " The maximal number of iterations GMRES must make. ");
        prm.declare_entry("eps",
                          "1e-12",
                          Patterns::Double(0.),
                          " The stopping criterion. ");
        prm.declare_entry("Krylov_size",
                          "30",
                          Patterns::Integer(1),
                          " The size of the Krylov subspace to be used. ");
        prm.declare_entry("off_diagonals",
                          "60",
                          Patterns::Integer(0),
                          " The number of off-diagonal elements ILU must "
                          "compute. ");
        prm.declare_entry("diag_strength",
                          "0.01",
                          Patterns::Double(0.),
                          " Diagonal strengthening coefficient. ");
        prm.declare_entry("update_prec",
                          "15",
                          Patterns::Integer(1),
                          " This number indicates how often we need to "
                          "update the preconditioner");
      }
      prm.leave_subsection();

      prm.declare_entry("verbose",
                        "true",
                        Patterns::Bool(),
                        " This indicates whether the output of the solution "
                        "process should be verbose. ");

      prm.declare_entry("output_interval",
                        "1",
                        Patterns::Integer(1),
                        " This indicates between how many time steps we print "
                        "the solution. ");

      // Append at the end of Data_Storage::Data_Storage():
      prm.enter_subsection("Immersed solid");
      {
        prm.declare_entry("Use Nitsche method",
                          "false",
                          Patterns::Bool(),
                          "Enable/disable Nitsche method for immersed solids");
        prm.declare_entry("Nitsche beta",
                          "100.0",
                          Patterns::Double(0.),
                          "Nitsche penalty coefficient beta.");
        prm.declare_entry("Solid refinement",
                          "3",
                          Patterns::Integer(0, 10),
                          "Global refinement level for solid mesh.");
        prm.declare_entry("Solid angular velocity",
                          "6.2831853",  // 2*pi
                          Patterns::Double(),
                          "Angular velocity of solid rotation (rad/s).");
        prm.declare_entry("Solid center",
                          "0.0, 0.0",
                          Patterns::List(Patterns::Double(), 2, 2),
                          "Center of the solid disk.");
        prm.declare_entry("Solid radius",
                          "0.1",
                          Patterns::Double(0.),
                          "Radius of the solid disk.");
        
        // Additional parameters from step-35-ibm.cc
        prm.declare_entry("Solid geometry type",
                          "circle",
                          Patterns::Selection("circle|rectangle"),
                          "Type of solid geometry (circle or rectangle)");
        prm.declare_entry("Solid width",
                          "0.5",
                          Patterns::Double(0.),
                          "Width for rectangle geometry");
        prm.declare_entry("Solid height",
                          "0.3",
                          Patterns::Double(0.),
                          "Height for rectangle geometry");
        prm.declare_entry("Motion type",
                          "static",
                          Patterns::Selection("static|prescribed|fsi_coupled"),
                          "Type of motion model");
        prm.declare_entry("Amplitude X",
                          "0.0",
                          Patterns::Double(),
                          "Prescribed motion amplitude in X");
        prm.declare_entry("Amplitude Y",
                          "0.0",
                          Patterns::Double(),
                          "Prescribed motion amplitude in Y");
        prm.declare_entry("Frequency",
                          "1.0",
                          Patterns::Double(0.),
                          "Prescribed motion frequency");
        prm.declare_entry("Rotation speed",
                          "0.0",
                          Patterns::Double(),
                          "Prescribed rotation speed (rad/s)");
        prm.declare_entry("Solid density",
                          "1.0",
                          Patterns::Double(0.),
                          "Solid density for FSI coupling");
        prm.declare_entry("Relaxation factor",
                          "1.0",
                          Patterns::Double(0.),
                          "IBM relaxation factor");
      }
      prm.leave_subsection();

      // Mesh generation parameters
      prm.enter_subsection("Mesh generation");
      {
        prm.declare_entry("Mesh generation type",
                          "file",
                          Patterns::Selection("file|channel"),
                          "Type of mesh generation: 'file' to read from file, 'channel' to generate channel geometry");
        prm.declare_entry("Channel length",
                          "10.0",
                          Patterns::Double(0.),
                          "Length of the channel (for 2D/3D)");
        prm.declare_entry("Channel width",
                          "4.1",
                          Patterns::Double(0.),
                          "Width of the channel (for 2D/3D)");
        prm.declare_entry("Channel height",
                          "1.0",
                          Patterns::Double(0.),
                          "Height of the channel (for 3D only)");
        prm.declare_entry("Channel refinement",
                          "0",
                          Patterns::Integer(0, 10),
                          "Initial refinement level for channel mesh");
        prm.declare_entry("Channel cells X",
                          "1",
                          Patterns::Integer(1, 1000),
                          "Number of cells in x-direction (for anisotropic refinement)");
        prm.declare_entry("Channel cells Y",
                          "1",
                          Patterns::Integer(1, 1000),
                          "Number of cells in y-direction (for anisotropic refinement)");
        prm.declare_entry("Channel cells Z",
                          "1",
                          Patterns::Integer(1, 1000),
                          "Number of cells in z-direction (for 3D anisotropic refinement)");
      }
      prm.leave_subsection();

      // Append boundary condition parameters
      prm.enter_subsection("Boundary conditions");
      {
        // Boundary 1: left wall
        prm.declare_entry("Boundary 1 type",
                          "zero",
                          Patterns::Selection("zero|constant|parabolic|outflow|outflow_normal"),
                          "Boundary condition type for left wall (boundary ID 1)");
        prm.declare_entry("Boundary 1 velocity X",
                          "0.0",
                          Patterns::Double(),
                          "X-component velocity for left wall (if type is constant)");
        prm.declare_entry("Boundary 1 velocity Y",
                          "0.0",
                          Patterns::Double(),
                          "Y-component velocity for left wall (if type is constant)");
        prm.declare_entry("Boundary 1 max velocity",
                          "1.5",
                          Patterns::Double(0.),
                          "Maximum velocity for parabolic profile (if type is parabolic)");
        
        // Boundary 2: right wall (inlet/outlet)
        prm.declare_entry("Boundary 2 type",
                          "parabolic",
                          Patterns::Selection("zero|constant|parabolic|outflow|outflow_normal"),
                          "Boundary condition type for right wall (boundary ID 2)");
        prm.declare_entry("Boundary 2 velocity X",
                          "0.0",
                          Patterns::Double(),
                          "X-component velocity for right wall (if type is constant)");
        prm.declare_entry("Boundary 2 velocity Y",
                          "0.0",
                          Patterns::Double(),
                          "Y-component velocity for right wall (if type is constant)");
        prm.declare_entry("Boundary 2 max velocity",
                          "1.5",
                          Patterns::Double(0.),
                          "Maximum velocity for parabolic profile (if type is parabolic)");
        
        // Boundary 3: bottom wall
        prm.declare_entry("Boundary 3 type",
                          "zero",
                          Patterns::Selection("zero|constant|parabolic|outflow|outflow_normal"),
                          "Boundary condition type for bottom wall (boundary ID 3)");
        prm.declare_entry("Boundary 3 velocity X",
                          "0.0",
                          Patterns::Double(),
                          "X-component velocity for bottom wall (if type is constant)");
        prm.declare_entry("Boundary 3 velocity Y",
                          "0.0",
                          Patterns::Double(),
                          "Y-component velocity for bottom wall (if type is constant)");
        prm.declare_entry("Boundary 3 max velocity",
                          "1.5",
                          Patterns::Double(0.),
                          "Maximum velocity for parabolic profile (if type is parabolic)");
        
        // Boundary 4: top wall
        prm.declare_entry("Boundary 4 type",
                          "zero",
                          Patterns::Selection("zero|constant|parabolic|outflow|outflow_normal"),
                          "Boundary condition type for top wall (boundary ID 4)");
        prm.declare_entry("Boundary 4 velocity X",
                          "0.0",
                          Patterns::Double(),
                          "X-component velocity for top wall (if type is constant)");
        prm.declare_entry("Boundary 4 velocity Y",
                          "0.0",
                          Patterns::Double(),
                          "Y-component velocity for top wall (if type is constant)");
        prm.declare_entry("Boundary 4 max velocity",
                          "1.5",
                          Patterns::Double(0.),
                          "Maximum velocity for parabolic profile (if type is parabolic)");
      }
      prm.leave_subsection();

      // Append initial condition parameters
      prm.enter_subsection("Initial conditions");
      {
        prm.declare_entry("Initial velocity type",
                          "parabolic",
                          Patterns::Selection("zero|constant|parabolic"),
                          "Type of initial velocity field: 'zero' for zero velocity, 'constant' for constant velocity, 'parabolic' for parabolic profile");
        prm.declare_entry("Initial velocity X",
                          "0.0",
                          Patterns::Double(),
                          "X-component for constant initial velocity");
        prm.declare_entry("Initial velocity Y",
                          "0.0",
                          Patterns::Double(),
                          "Y-component for constant initial velocity");
        prm.declare_entry("Initial max velocity",
                          "1.5",
                          Patterns::Double(0.),
                          "Maximum velocity for parabolic initial profile");
      }
      prm.leave_subsection();

      // Sponge layer parameters
      prm.enter_subsection("Sponge layer");
      {
        prm.declare_entry("Use sponge layer",
                          "false",
                          Patterns::Bool(),
                          "Enable/disable sponge layer damping at outlet");
        prm.declare_entry("Sponge x start",
                          "20.0",
                          Patterns::Double(),
                          "X-coordinate where sponge layer starts");
        prm.declare_entry("Sponge strength",
                          "10.0",
                          Patterns::Double(0.),
                          "Maximum damping coefficient sigma_max");
        prm.declare_entry("Sponge order",
                          "2",
                          Patterns::Integer(1, 3),
                          "Order of damping profile: 1=linear, 2=quadratic, 3=cubic");
      }
      prm.leave_subsection();
    }



    void Data_Storage::read_data(const std::string &filename)
    {
      std::ifstream file(filename);
      AssertThrow(file, ExcFileNotOpen(filename));

      prm.parse_input(file);

      if (prm.get("Method_Form") == "rotational")
        form = Method::rotational;
      else
        form = Method::standard;

      prm.enter_subsection("Physical data");
      {
        initial_time = prm.get_double("initial_time");
        final_time   = prm.get_double("final_time");
        Reynolds     = prm.get_double("Reynolds");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time step data");
      {
        dt = prm.get_double("dt");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization");
      {
        n_global_refines = prm.get_integer("n_of_refines");
        pressure_degree  = prm.get_integer("pressure_fe_degree");
        mesh_filename    = prm.get("mesh_filename");
      }
      prm.leave_subsection();

      prm.enter_subsection("Data solve velocity");
      {
        vel_max_iterations = prm.get_integer("max_iterations");
        vel_eps            = prm.get_double("eps");
        vel_Krylov_size    = prm.get_integer("Krylov_size");
        vel_off_diagonals  = prm.get_integer("off_diagonals");
        vel_diag_strength  = prm.get_double("diag_strength");
        vel_update_prec    = prm.get_integer("update_prec");
      }
      prm.leave_subsection();

      verbose = prm.get_bool("verbose");

      output_interval = prm.get_integer("output_interval");

      // Append at the end of Data_Storage::read_data():
      prm.enter_subsection("Immersed solid");
      {
        use_nitsche           = prm.get_bool("Use Nitsche method");
        nitsche_beta          = prm.get_double("Nitsche beta");
        solid_refinement      = prm.get_integer("Solid refinement");
        solid_angular_velocity = prm.get_double("Solid angular velocity");
        solid_radius          = prm.get_double("Solid radius");
        solid_geometry_type   = prm.get("Solid geometry type");
        solid_width           = prm.get_double("Solid width");
        solid_height          = prm.get_double("Solid height");
        motion_type           = prm.get("Motion type");
        amplitude_x           = prm.get_double("Amplitude X");
        amplitude_y           = prm.get_double("Amplitude Y");
        frequency             = prm.get_double("Frequency");
        rotation_speed        = prm.get_double("Rotation speed");
        solid_density         = prm.get_double("Solid density");
        relaxation_factor     = prm.get_double("Relaxation factor");

        // Parse solid center coordinates
        const std::vector<std::string> coords =
          Utilities::split_string_list(prm.get("Solid center"));
        solid_center[0] = Utilities::string_to_double(coords[0]);
        solid_center[1] = Utilities::string_to_double(coords[1]);
      }
      prm.leave_subsection();

      // Read mesh generation parameters
      prm.enter_subsection("Mesh generation");
      {
        mesh_generation_type = prm.get("Mesh generation type");
        channel_length = prm.get_double("Channel length");
        channel_width = prm.get_double("Channel width");
        channel_height = prm.get_double("Channel height");
        channel_refinement = prm.get_integer("Channel refinement");
        channel_cells_x = prm.get_integer("Channel cells X");
        channel_cells_y = prm.get_integer("Channel cells Y");
        channel_cells_z = prm.get_integer("Channel cells Z");
      }
      prm.leave_subsection();

      // Read boundary condition parameters
      prm.enter_subsection("Boundary conditions");
      {
        // Initialize vectors for 6 boundaries (though we only use 1-4 for 2D)
        boundary_types.resize(7);      // Index 1-6
        boundary_values_x.resize(7);
        boundary_values_y.resize(7);
        boundary_max_vel.resize(7);
        
        // Boundary 1
        boundary_types[1] = prm.get("Boundary 1 type");
        boundary_values_x[1] = prm.get_double("Boundary 1 velocity X");
        boundary_values_y[1] = prm.get_double("Boundary 1 velocity Y");
        boundary_max_vel[1] = prm.get_double("Boundary 1 max velocity");
        
        // Boundary 2
        boundary_types[2] = prm.get("Boundary 2 type");
        boundary_values_x[2] = prm.get_double("Boundary 2 velocity X");
        boundary_values_y[2] = prm.get_double("Boundary 2 velocity Y");
        boundary_max_vel[2] = prm.get_double("Boundary 2 max velocity");
        
        // Boundary 3
        boundary_types[3] = prm.get("Boundary 3 type");
        boundary_values_x[3] = prm.get_double("Boundary 3 velocity X");
        boundary_values_y[3] = prm.get_double("Boundary 3 velocity Y");
        boundary_max_vel[3] = prm.get_double("Boundary 3 max velocity");
        
        // Boundary 4
        boundary_types[4] = prm.get("Boundary 4 type");
        boundary_values_x[4] = prm.get_double("Boundary 4 velocity X");
        boundary_values_y[4] = prm.get_double("Boundary 4 velocity Y");
        boundary_max_vel[4] = prm.get_double("Boundary 4 max velocity");
        
        // Boundaries 5 and 6 (for 3D) - set defaults
        boundary_types[5] = "zero";
        boundary_values_x[5] = 0.0;
        boundary_values_y[5] = 0.0;
        boundary_max_vel[5] = 1.5;
        
        boundary_types[6] = "zero";
        boundary_values_x[6] = 0.0;
        boundary_values_y[6] = 0.0;
        boundary_max_vel[6] = 1.5;
      }
      prm.leave_subsection();

      // Read initial condition parameters
      prm.enter_subsection("Initial conditions");
      {
        initial_velocity_type = prm.get("Initial velocity type");
        initial_velocity_x = prm.get_double("Initial velocity X");
        initial_velocity_y = prm.get_double("Initial velocity Y");
        initial_max_velocity = prm.get_double("Initial max velocity");
      }
      prm.leave_subsection();

      // Read sponge layer parameters
      prm.enter_subsection("Sponge layer");
      {
        use_sponge_layer = prm.get_bool("Use sponge layer");
        sponge_x_start = prm.get_double("Sponge x start");
        sponge_strength = prm.get_double("Sponge strength");
        sponge_order = prm.get_integer("Sponge order");
      }
      prm.leave_subsection();
    }
  } // namespace RunTimeParameters



  // @sect3{Equation data}

  // In the next namespace, we declare the initial and boundary conditions:
  namespace EquationData
  {
    // As we have chosen a completely decoupled formulation, we will not take
    // advantage of deal.II's capabilities to handle vector valued problems. We
    // do, however, want to use an interface for the equation data that is
    // somehow dimension independent. To be able to do that, our functions
    // should be able to know on which spatial component we are currently
    // working, and we should be able to have a common interface to do that. The
    // following class is an attempt in that direction.
    template <int dim>
    class MultiComponentFunction : public Function<dim>
    {
    public:
      MultiComponentFunction(const double initial_time = 0.);
      void set_component(const unsigned int d);

    protected:
      unsigned int comp;
    };

    template <int dim>
    MultiComponentFunction<dim>::MultiComponentFunction(
      const double initial_time)
      : Function<dim>(1, initial_time)
      , comp(0)
    {}


    template <int dim>
    void MultiComponentFunction<dim>::set_component(const unsigned int d)
    {
      Assert(d < dim, ExcIndexRange(d, 0, dim));
      comp = d;
    }


    // With this class defined, we declare classes that describe the boundary
    // conditions for velocity and pressure:
    template <int dim>
    class Velocity : public MultiComponentFunction<dim>
    {
    public:
      Velocity(const double initial_time = 0.0,
               const std::string &type = "parabolic",
               const double vx = 0.0,
               const double vy = 0.0,
               const double max_vel = 1.5,
               const double channel_width = 4.1);

      virtual double value(const Point<dim>  &p,
                           const unsigned int component = 0) const override;

      virtual void value_list(const std::vector<Point<dim>> &points,
                              std::vector<double>           &values,
                              const unsigned int component = 0) const override;

    private:
      std::string velocity_type;
      double velocity_x;
      double velocity_y;
      double max_velocity;
      double channel_H;
    };


    template <int dim>
    Velocity<dim>::Velocity(const double initial_time,
                            const std::string &type,
                            const double vx,
                            const double vy,
                            const double max_vel,
                            const double channel_width)
      : MultiComponentFunction<dim>(initial_time)
      , velocity_type(type)
      , velocity_x(vx)
      , velocity_y(vy)
      , max_velocity(max_vel)
      , channel_H(channel_width)
    {}


    template <int dim>
    void Velocity<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double>           &values,
                                   const unsigned int) const
    {
      const unsigned int n_points = points.size();
      AssertDimension(values.size(), n_points);
      for (unsigned int i = 0; i < n_points; ++i)
        values[i] = Velocity<dim>::value(points[i]);
    }


    template <int dim>
    double Velocity<dim>::value(const Point<dim> &p, const unsigned int) const
    {
      if (velocity_type == "zero")
        {
          // Zero velocity field
          return 0.0;
        }
      else if (velocity_type == "constant")
        {
          // Constant velocity field
          if (this->comp == 0)
            return velocity_x;
          else if (this->comp == 1)
            return velocity_y;
          else
            return 0.0;
        }
      else if (velocity_type == "parabolic")
        {
          // Parabolic velocity profile (only for x-component)
          if (this->comp == 0)
            {
              return 4. * max_velocity * p[1] * (channel_H - p[1]) / (channel_H * channel_H);
            }
          else
            return 0.;
        }
      else
        {
          // Default to zero velocity for unknown types
          return 0.0;
        }
    }



    template <int dim>
    class Pressure : public Function<dim>
    {
    public:
      Pressure(const double initial_time = 0.0);

      virtual double value(const Point<dim>  &p,
                           const unsigned int component = 0) const override;

      virtual void value_list(const std::vector<Point<dim>> &points,
                              std::vector<double>           &values,
                              const unsigned int component = 0) const override;
    };

    template <int dim>
    Pressure<dim>::Pressure(const double initial_time)
      : Function<dim>(1, initial_time)
    {}


    template <int dim>
    double Pressure<dim>::value(const Point<dim>  &p,
                                const unsigned int component) const
    {
      (void)component;
      AssertIndexRange(component, 1);
      return 25. - p[0];
    }

    template <int dim>
    void Pressure<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double>           &values,
                                   const unsigned int component) const
    {
      (void)component;
      AssertIndexRange(component, 1);
      const unsigned int n_points = points.size();
      AssertDimension(values.size(), n_points);
      for (unsigned int i = 0; i < n_points; ++i)
        values[i] = Pressure<dim>::value(points[i]);
    }

    // ============================================================
    // Solid rotational velocity field (adapted from step-70's SolidVelocity)
    // ============================================================
    template <int dim>
    class SolidVelocity : public Function<dim>
    {
    public:
      SolidVelocity(const double omega      = 2.0 * numbers::PI,
                    const double initial_time = 0.0)
        : Function<dim>(dim, initial_time)
        , omega(omega)
      {}

      // Rotational velocity field around origin: u_x = -ω*y, u_y = ω*x
      virtual double value(const Point<dim> &p,
                           unsigned int      component = 0) const override
      {
        if (component == 0)
          return -omega * p[1];
        else if (component == 1)
          return omega * p[0];
        return 0.0;
      }

    private:
      double omega; // Angular velocity
    };

  } // namespace EquationData



  // @sect3{The <code>NavierStokesProjection</code> class}

  // Now for the main class of the program. It implements the various versions
  // of the projection method for Navier-Stokes equations. The names for all the
  // methods and member variables should be self-explanatory, taking into
  // account the implementation details given in the introduction.
  template <int dim>
  class NavierStokesProjection
  {
  public:
    NavierStokesProjection(const RunTimeParameters::Data_Storage &data);

    void run(const bool verbose = false, const unsigned int n_plots = 10);

  protected:
    RunTimeParameters::Method type;

    const unsigned int deg;
    const double       dt;
    const double       t_0;
    const double       T;
    const double       Re;

    EquationData::Velocity<dim>               vel_exact;
    std::map<types::global_dof_index, double> boundary_values;
    std::vector<types::boundary_id>           boundary_ids;

    Triangulation<dim> triangulation;

    const FE_Q<dim> fe_velocity;
    const FE_Q<dim> fe_pressure;

    DoFHandler<dim> dof_handler_velocity;
    DoFHandler<dim> dof_handler_pressure;

    const QGauss<dim> quadrature_pressure;
    const QGauss<dim> quadrature_velocity;

    SparsityPattern sparsity_pattern_velocity;
    SparsityPattern sparsity_pattern_pressure;
    SparsityPattern sparsity_pattern_pres_vel;

    SparseMatrix<double> vel_Laplace_plus_Mass;
    SparseMatrix<double> vel_it_matrix[dim];
    SparseMatrix<double> vel_Mass;
    SparseMatrix<double> vel_Laplace;
    SparseMatrix<double> vel_Advection;
    SparseMatrix<double> pres_Laplace;
    SparseMatrix<double> pres_Mass;
    SparseMatrix<double> pres_Diff[dim];
    SparseMatrix<double> pres_iterative;

    Vector<double> pres_n;
    Vector<double> pres_n_minus_1;
    Vector<double> phi_n;
    Vector<double> phi_n_minus_1;
    Vector<double> u_n[dim];
    Vector<double> u_n_minus_1[dim];
    Vector<double> u_star[dim];
    Vector<double> force[dim];
    Vector<double> v_tmp;
    Vector<double> pres_tmp;
    Vector<double> rot_u;

    SparseILU<double>   prec_velocity[dim];
    SparseILU<double>   prec_pres_Laplace;
    SparseDirectUMFPACK prec_mass;
    SparseDirectUMFPACK prec_vel_mass;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << ']');

    void create_triangulation_and_dofs(const unsigned int n_refines);

    void initialize();

    void interpolate_velocity();

    void diffusion_step(const bool reinit_prec);

    void projection_step(const bool reinit_prec);

    void update_pressure(const bool reinit_prec);

  private:
    unsigned int vel_max_its;
    unsigned int vel_Krylov_size;
    unsigned int vel_off_diagonals;
    unsigned int vel_update_prec;
    double       vel_eps;
    double       vel_diag_strength;

    void initialize_velocity_matrices();

    void initialize_pressure_matrices();

    // The next few structures and functions are for doing various things in
    // parallel. They follow the scheme laid out in @ref threads, using the
    // WorkStream class. As explained there, this requires us to declare two
    // structures for each of the assemblers, a per-task data and a scratch data
    // structure. These are then handed over to functions that assemble local
    // contributions and that copy these local contributions to the global
    // objects.
    //
    // One of the things that are specific to this program is that we don't just
    // have a single DoFHandler object that represents both the velocities and
    // the pressure, but we use individual DoFHandler objects for these two
    // kinds of variables. We pay for this optimization when we want to assemble
    // terms that involve both variables, such as the divergence of the velocity
    // and the gradient of the pressure, times the respective test functions.
    // When doing so, we can't just anymore use a single FEValues object, but
    // rather we need two, and they need to be initialized with cell iterators
    // that point to the same cell in the triangulation but different
    // DoFHandlers.
    //
    // To do this in practice, we declare a "synchronous" iterator -- an object
    // that internally consists of several (in our case two) iterators, and each
    // time the synchronous iteration is moved forward one step, each of the
    // iterators stored internally is moved forward one step as well, thereby
    // always staying in sync. As it so happens, there is a deal.II class that
    // facilitates this sort of thing. (What is important here is to know that
    // two DoFHandler objects built on the same triangulation will walk over the
    // cells of the triangulation in the same order.)
    using IteratorTuple =
      std::tuple<typename DoFHandler<dim>::active_cell_iterator,
                 typename DoFHandler<dim>::active_cell_iterator>;

    using IteratorPair = SynchronousIterators<IteratorTuple>;

    void initialize_gradient_operator();

    struct InitGradPerTaskData
    {
      unsigned int                         d;
      unsigned int                         vel_dpc;
      unsigned int                         pres_dpc;
      FullMatrix<double>                   local_grad;
      std::vector<types::global_dof_index> vel_local_dof_indices;
      std::vector<types::global_dof_index> pres_local_dof_indices;

      InitGradPerTaskData(const unsigned int dd,
                          const unsigned int vdpc,
                          const unsigned int pdpc)
        : d(dd)
        , vel_dpc(vdpc)
        , pres_dpc(pdpc)
        , local_grad(vdpc, pdpc)
        , vel_local_dof_indices(vdpc)
        , pres_local_dof_indices(pdpc)
      {}
    };

    struct InitGradScratchData
    {
      unsigned int  nqp;
      FEValues<dim> fe_val_vel;
      FEValues<dim> fe_val_pres;
      InitGradScratchData(const FE_Q<dim>   &fe_v,
                          const FE_Q<dim>   &fe_p,
                          const QGauss<dim> &quad,
                          const UpdateFlags  flags_v,
                          const UpdateFlags  flags_p)
        : nqp(quad.size())
        , fe_val_vel(fe_v, quad, flags_v)
        , fe_val_pres(fe_p, quad, flags_p)
      {}
      InitGradScratchData(const InitGradScratchData &data)
        : nqp(data.nqp)
        , fe_val_vel(data.fe_val_vel.get_fe(),
                     data.fe_val_vel.get_quadrature(),
                     data.fe_val_vel.get_update_flags())
        , fe_val_pres(data.fe_val_pres.get_fe(),
                      data.fe_val_pres.get_quadrature(),
                      data.fe_val_pres.get_update_flags())
      {}
    };

    void assemble_one_cell_of_gradient(const IteratorPair  &SI,
                                       InitGradScratchData &scratch,
                                       InitGradPerTaskData &data);

    void copy_gradient_local_to_global(const InitGradPerTaskData &data);

    // The same general layout also applies to the following classes and
    // functions implementing the assembly of the advection term:
    void assemble_advection_term();

    struct AdvectionPerTaskData
    {
      FullMatrix<double>                   local_advection;
      std::vector<types::global_dof_index> local_dof_indices;
      AdvectionPerTaskData(const unsigned int dpc)
        : local_advection(dpc, dpc)
        , local_dof_indices(dpc)
      {}
    };

    struct AdvectionScratchData
    {
      unsigned int                nqp;
      unsigned int                dpc;
      std::vector<Point<dim>>     u_star_local;
      std::vector<Tensor<1, dim>> grad_u_star;
      std::vector<double>         u_star_tmp;
      FEValues<dim>               fe_val;
      AdvectionScratchData(const FE_Q<dim>   &fe,
                           const QGauss<dim> &quad,
                           const UpdateFlags  flags)
        : nqp(quad.size())
        , dpc(fe.n_dofs_per_cell())
        , u_star_local(nqp)
        , grad_u_star(nqp)
        , u_star_tmp(nqp)
        , fe_val(fe, quad, flags)
      {}

      AdvectionScratchData(const AdvectionScratchData &data)
        : nqp(data.nqp)
        , dpc(data.dpc)
        , u_star_local(nqp)
        , grad_u_star(nqp)
        , u_star_tmp(nqp)
        , fe_val(data.fe_val.get_fe(),
                 data.fe_val.get_quadrature(),
                 data.fe_val.get_update_flags())
      {}
    };

    void assemble_one_cell_of_advection(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      AdvectionScratchData                                 &scratch,
      AdvectionPerTaskData                                 &data);

    void copy_advection_local_to_global(const AdvectionPerTaskData &data);

    // The final few functions implement the diffusion solve as well as
    // postprocessing the output, including computing the curl of the velocity:
    void diffusion_component_solve(const unsigned int d);

    void output_results(const unsigned int step);

    void assemble_vorticity(const bool reinit_prec);

    // ================================================================
    // FSI additional members (adapted from step-70 for serial structure)
    // ================================================================

    // Solid mesh and DoFs (only for generating integration points, not solving solid equations)
    Triangulation<dim>     solid_triangulation;
    FE_Q<dim>              solid_fe_q;
    DoFHandler<dim>        solid_dof_handler;
    QGauss<dim>            solid_quadrature;

    // Particle handler: stores solid integration point positions and JxW weights
    // Each particle carries 1 property: JxW integration weight
    Particles::ParticleHandler<dim> solid_particle_handler;

    // Nitsche penalty parameters
    bool   use_nitsche;          // Enable/disable Nitsche method
    double nitsche_beta;         // Input parameter β
    double nitsche_penalty_param;// Actual β/h (h is minimum cell diameter)
    double solid_omega;          // Solid angular velocity

    // Solid center and radius (for initializing solid mesh)
    Point<dim> solid_center;
    double     solid_radius;

    // Mesh filename
    std::string mesh_filename;

    // ================================================================
    // Mesh generation parameters
    // ================================================================
    std::string  mesh_generation_type;  // "file" or "channel"
    double       channel_length;        // Channel length (for 2D/3D)
    double       channel_width;         // Channel width (for 2D/3D)
    double       channel_height;        // Channel height (for 3D only)
    unsigned int channel_refinement;    // Initial refinement for channel
    unsigned int channel_cells_x;       // Number of cells in x-direction
    unsigned int channel_cells_y;       // Number of cells in y-direction
    unsigned int channel_cells_z;       // Number of cells in z-direction (3D only)

    // ================================================================
    // FSI additional methods
    // ================================================================

    // Initialize solid particles (called only once)
    void setup_solid_particles();

    // Move solid particles each time step
    void move_solid_particles(const double dt);

    // Assemble Nitsche penalty term (component d) into given matrix and vector
    void assemble_nitsche_restriction(unsigned int      d,
                                      SparseMatrix<double> &vel_matrix,
                                      Vector<double>       &rhs_vector);

    // Output solid particle positions (for visualization)
    void output_solid_particles(const unsigned int step);

    // ================================================================
    // Boundary condition parameters
    // ================================================================
    std::vector<std::string> boundary_types;      // Type for each boundary ID (1-6)
    std::vector<double>      boundary_values_x;   // X-component value for constant BC
    std::vector<double>      boundary_values_y;   // Y-component value for constant BC
    std::vector<double>      boundary_max_vel;    // Max velocity for parabolic inlet

    // ================================================================
    // Sponge layer parameters
    // ================================================================
    bool         use_sponge_layer;       // Enable/disable sponge layer damping
    double       sponge_x_start;         // X-coordinate where sponge layer starts
    double       sponge_x_end;           // X-coordinate where sponge layer ends (computed as channel_length)
    double       sponge_strength;        // Maximum damping coefficient σ_max
    unsigned int sponge_order;           // Order of damping profile (1=linear, 2=quadratic, 3=cubic)

    // Helper function to compute sponge damping coefficient σ(x)
    double compute_sponge_coefficient(const Point<dim> &p) const;
  };



  // @sect4{ <code>NavierStokesProjection::NavierStokesProjection</code> }

  // In the constructor, we just read all the data from the
  // <code>Data_Storage</code> object that is passed as an argument, verify that
  // the data we read is reasonable and, finally, create the triangulation and
  // load the initial data.
  template <int dim>
  NavierStokesProjection<dim>::NavierStokesProjection(
    const RunTimeParameters::Data_Storage &data)
    : type(data.form)
    , deg(data.pressure_degree)
    , dt(data.dt)
    , t_0(data.initial_time)
    , T(data.final_time)
    , Re(data.Reynolds)
    , vel_exact(data.initial_time,
                data.initial_velocity_type,
                data.initial_velocity_x,
                data.initial_velocity_y,
                data.initial_max_velocity,
                data.channel_width)
    , fe_velocity(deg + 1)
    , fe_pressure(deg)
    , dof_handler_velocity(triangulation)
    , dof_handler_pressure(triangulation)
    , quadrature_pressure(deg + 1)
    , quadrature_velocity(deg + 2)
    , vel_max_its(data.vel_max_iterations)
    , vel_Krylov_size(data.vel_Krylov_size)
    , vel_off_diagonals(data.vel_off_diagonals)
    , vel_update_prec(data.vel_update_prec)
    , vel_eps(data.vel_eps)
    , vel_diag_strength(data.vel_diag_strength)
    // ===== Additional FSI initialization =====
    , solid_fe_q(deg + 1)          // Solid FE order matches velocity
    , solid_dof_handler(solid_triangulation)
    , solid_quadrature(deg + 2)    // Solid integration accuracy
    , solid_particle_handler(triangulation,
                            StaticMappingQ1<dim>::mapping,
                            1 /* n_properties: JxW */)
    , use_nitsche(data.use_nitsche)
    , nitsche_beta(data.nitsche_beta)
    , nitsche_penalty_param(0.0)   // Will be computed in setup_solid_particles()
    , solid_omega(data.solid_angular_velocity)
    , solid_center(data.solid_center)
    , solid_radius(data.solid_radius)
    , mesh_filename(data.mesh_filename)
    , mesh_generation_type(data.mesh_generation_type)
    , channel_length(data.channel_length)
    , channel_width(data.channel_width)
    , channel_height(data.channel_height)
    , channel_refinement(data.channel_refinement)
    , channel_cells_x(data.channel_cells_x)
    , channel_cells_y(data.channel_cells_y)
    , channel_cells_z(data.channel_cells_z)
    // Boundary condition parameters
    , boundary_types(data.boundary_types)
    , boundary_values_x(data.boundary_values_x)
    , boundary_values_y(data.boundary_values_y)
    , boundary_max_vel(data.boundary_max_vel)
    , use_sponge_layer(data.use_sponge_layer)
    , sponge_x_start(data.sponge_x_start)
    , sponge_x_end(data.channel_length)
    , sponge_strength(data.sponge_strength)
    , sponge_order(data.sponge_order)
  {
    if (deg < 1)
      std::cout
        << " WARNING: The chosen pair of finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;

    AssertThrow(!((dt <= 0.) || (dt > .5 * T)), ExcInvalidTimeStep(dt, .5 * T));

    create_triangulation_and_dofs(data.n_global_refines);
    initialize();
    // ===== Additional: Initialize solid particles after fluid mesh and DoFs are set up =====
    if (use_nitsche)
      setup_solid_particles();
  }


  // @sect4{<code>NavierStokesProjection::create_triangulation_and_dofs</code>}

  // The method that creates the triangulation and refines it the needed number
  // of times. After creating the triangulation, it creates the mesh dependent
  // data, i.e. it distributes degrees of freedom and renumbers them, and
  // initializes the matrices and vectors that we will use.
  template <int dim>
  void NavierStokesProjection<dim>::create_triangulation_and_dofs(
    const unsigned int n_refines)
  {
    if (mesh_generation_type == "file")
      {
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);

        std::string   filename = mesh_filename;
        std::ifstream file(filename);
        Assert(file, ExcFileNotOpen(filename));
        grid_in.read_ucd(file);

        std::cout << "Mesh file: " << mesh_filename << std::endl;
      }
    else if (mesh_generation_type == "channel")
      {
        // Generate channel geometry
        std::cout << "Generating channel geometry:" << std::endl;
        std::cout << "  Length: " << channel_length << std::endl;
        std::cout << "  Width: " << channel_width << std::endl;
        
        if (dim == 2)
          {
            // 2D channel: rectangle [0, length] x [0, width]
            Point<dim> p1(0.0, 0.0);
            Point<dim> p2(channel_length, channel_width);
            
            // Use subdivided_hyper_rectangle for anisotropic refinement
            std::vector<unsigned int> repetitions(dim);
            repetitions[0] = channel_cells_x; // cells in x-direction
            repetitions[1] = channel_cells_y; // cells in y-direction
            
            GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      repetitions,
                                                      p1,
                                                      p2);
            
            // Set boundary IDs:
            // 1: left wall (x=0)
            // 2: right wall (x=length)
            // 3: bottom wall (y=0)
            // 4: top wall (y=width)
            for (auto &cell : triangulation.active_cell_iterators())
              for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                  {
                    const Point<dim> face_center = cell->face(f)->center();
                    if (std::abs(face_center[0] - 0.0) < 1e-10)
                      cell->face(f)->set_boundary_id(1); // left wall
                    else if (std::abs(face_center[0] - channel_length) < 1e-10)
                      cell->face(f)->set_boundary_id(2); // right wall
                    else if (std::abs(face_center[1] - 0.0) < 1e-10)
                      cell->face(f)->set_boundary_id(3); // bottom wall
                    else if (std::abs(face_center[1] - channel_width) < 1e-10)
                      cell->face(f)->set_boundary_id(4); // top wall
                  }
            
            std::cout << "  Cells in x-direction: " << channel_cells_x << std::endl;
            std::cout << "  Cells in y-direction: " << channel_cells_y << std::endl;
          }
        else if (dim == 3)
          {
            // 3D channel: box [0, length] x [0, width] x [0, height]
            std::cout << "  Height: " << channel_height << std::endl;
            Point<dim> p1(0.0, 0.0, 0.0);
            Point<dim> p2(channel_length, channel_width, channel_height);
            
            // Use subdivided_hyper_rectangle for anisotropic refinement
            std::vector<unsigned int> repetitions(dim);
            repetitions[0] = channel_cells_x; // cells in x-direction
            repetitions[1] = channel_cells_y; // cells in y-direction
            repetitions[2] = channel_cells_z; // cells in z-direction
            
            GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      repetitions,
                                                      p1,
                                                      p2);
            
            // Set boundary IDs:
            // 1: left wall (x=0)
            // 2: right wall (x=length)
            // 3: bottom wall (y=0)
            // 4: top wall (y=width)
            // 5: front wall (z=0)
            // 6: back wall (z=height)
            for (auto &cell : triangulation.active_cell_iterators())
              for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                  {
                    const Point<dim> face_center = cell->face(f)->center();
                    if (std::abs(face_center[0] - 0.0) < 1e-10)
                      cell->face(f)->set_boundary_id(1); // left wall
                    else if (std::abs(face_center[0] - channel_length) < 1e-10)
                      cell->face(f)->set_boundary_id(2); // right wall
                    else if (std::abs(face_center[1] - 0.0) < 1e-10)
                      cell->face(f)->set_boundary_id(3); // bottom wall
                    else if (std::abs(face_center[1] - channel_width) < 1e-10)
                      cell->face(f)->set_boundary_id(4); // top wall
                    else if (std::abs(face_center[2] - 0.0) < 1e-10)
                      cell->face(f)->set_boundary_id(5); // front wall
                    else if (std::abs(face_center[2] - channel_height) < 1e-10)
                      cell->face(f)->set_boundary_id(6); // back wall
                  }
            
            std::cout << "  Cells in x-direction: " << channel_cells_x << std::endl;
            std::cout << "  Cells in y-direction: " << channel_cells_y << std::endl;
            std::cout << "  Cells in z-direction: " << channel_cells_z << std::endl;
          }
        else
          {
            AssertThrow(false, ExcNotImplemented());
          }
        
        // Apply initial refinement for channel
        triangulation.refine_global(channel_refinement);
        std::cout << "Channel refinement level: " << channel_refinement << std::endl;
      }
    else
      {
        AssertThrow(false, ExcMessage("Unknown mesh generation type: " + mesh_generation_type));
      }

    std::cout << "Number of global refines = " << n_refines << std::endl;
    triangulation.refine_global(n_refines);
    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

    boundary_ids = triangulation.get_boundary_ids();

    dof_handler_velocity.distribute_dofs(fe_velocity);
    DoFRenumbering::boost::Cuthill_McKee(dof_handler_velocity);
    dof_handler_pressure.distribute_dofs(fe_pressure);
    DoFRenumbering::boost::Cuthill_McKee(dof_handler_pressure);

    initialize_velocity_matrices();
    initialize_pressure_matrices();
    initialize_gradient_operator();

    pres_n.reinit(dof_handler_pressure.n_dofs());
    pres_n_minus_1.reinit(dof_handler_pressure.n_dofs());
    phi_n.reinit(dof_handler_pressure.n_dofs());
    phi_n_minus_1.reinit(dof_handler_pressure.n_dofs());
    pres_tmp.reinit(dof_handler_pressure.n_dofs());
    for (unsigned int d = 0; d < dim; ++d)
      {
        u_n[d].reinit(dof_handler_velocity.n_dofs());
        u_n_minus_1[d].reinit(dof_handler_velocity.n_dofs());
        u_star[d].reinit(dof_handler_velocity.n_dofs());
        force[d].reinit(dof_handler_velocity.n_dofs());
      }
    v_tmp.reinit(dof_handler_velocity.n_dofs());
    rot_u.reinit(dof_handler_velocity.n_dofs());

    std::cout << "dim (X_h) = " << (dof_handler_velocity.n_dofs() * dim) //
              << std::endl                                               //
              << "dim (M_h) = " << dof_handler_pressure.n_dofs()         //
              << std::endl                                               //
              << "Re        = " << Re << std::endl                       //
              << std::endl;
  }


  // @sect4{ <code>NavierStokesProjection::initialize</code> }

  // This method creates the constant matrices and loads the initial data
  template <int dim>
  void NavierStokesProjection<dim>::initialize()
  {
    vel_Laplace_plus_Mass = 0.;
    vel_Laplace_plus_Mass.add(1. / Re, vel_Laplace);
    vel_Laplace_plus_Mass.add(1.5 / dt, vel_Mass);

    EquationData::Pressure<dim> pres(t_0);
    VectorTools::interpolate(dof_handler_pressure, pres, pres_n_minus_1);
    pres.advance_time(dt);
    VectorTools::interpolate(dof_handler_pressure, pres, pres_n);
    phi_n         = 0.;
    phi_n_minus_1 = 0.;
    for (unsigned int d = 0; d < dim; ++d)
      {
        vel_exact.set_time(t_0);
        vel_exact.set_component(d);
        VectorTools::interpolate(dof_handler_velocity,
                                 vel_exact,
                                 u_n_minus_1[d]);
        vel_exact.advance_time(dt);
        VectorTools::interpolate(dof_handler_velocity, vel_exact, u_n[d]);
      }
  }


  // @sect4{ <code>NavierStokesProjection::initialize_*_matrices</code> }

  // In this set of methods we initialize the sparsity patterns, the constraints
  // (if any) and assemble the matrices that do not depend on the timestep
  // <code>dt</code>. Note that for the Laplace and mass matrices, we can use
  // functions in the library that do this. Because the expensive operations of
  // this function -- creating the two matrices -- are entirely independent, we
  // could in principle mark them as tasks that can be worked on in %parallel
  // using the Threads::new_task functions. We won't do that here since these
  // functions internally already are parallelized, and in particular because
  // the current function is only called once per program run and so does not
  // incur a cost in each time step. The necessary modifications would be quite
  // straightforward, however.
  template <int dim>
  void NavierStokesProjection<dim>::initialize_velocity_matrices()
  {
    {
      DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs(),
                                 dof_handler_velocity.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler_velocity, dsp);
      sparsity_pattern_velocity.copy_from(dsp);
    }
    vel_Laplace_plus_Mass.reinit(sparsity_pattern_velocity);
    for (unsigned int d = 0; d < dim; ++d)
      vel_it_matrix[d].reinit(sparsity_pattern_velocity);
    vel_Mass.reinit(sparsity_pattern_velocity);
    vel_Laplace.reinit(sparsity_pattern_velocity);
    vel_Advection.reinit(sparsity_pattern_velocity);

    MatrixCreator::create_mass_matrix(dof_handler_velocity,
                                      quadrature_velocity,
                                      vel_Mass);
    MatrixCreator::create_laplace_matrix(dof_handler_velocity,
                                         quadrature_velocity,
                                         vel_Laplace);
  }

  // The initialization of the matrices that act on the pressure space is
  // similar to the ones that act on the velocity space.
  template <int dim>
  void NavierStokesProjection<dim>::initialize_pressure_matrices()
  {
    {
      DynamicSparsityPattern dsp(dof_handler_pressure.n_dofs(),
                                 dof_handler_pressure.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler_pressure, dsp);
      sparsity_pattern_pressure.copy_from(dsp);
    }

    pres_Laplace.reinit(sparsity_pattern_pressure);
    pres_iterative.reinit(sparsity_pattern_pressure);
    pres_Mass.reinit(sparsity_pattern_pressure);

    MatrixCreator::create_laplace_matrix(dof_handler_pressure,
                                         quadrature_pressure,
                                         pres_Laplace);
    MatrixCreator::create_mass_matrix(dof_handler_pressure,
                                      quadrature_pressure,
                                      pres_Mass);
  }


  // For the gradient operator, we start by initializing the sparsity pattern
  // and compressing it. It is important to notice here that the gradient
  // operator acts from the pressure space into the velocity space, so we have
  // to deal with two different finite element spaces. To keep the loops
  // synchronized, we use the alias that we have defined before, namely
  // <code>PairedIterators</code> and <code>IteratorPair</code>.
  template <int dim>
  void NavierStokesProjection<dim>::initialize_gradient_operator()
  {
    {
      DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs(),
                                 dof_handler_pressure.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler_velocity,
                                      dof_handler_pressure,
                                      dsp);
      sparsity_pattern_pres_vel.copy_from(dsp);
    }

    InitGradPerTaskData per_task_data(0,
                                      fe_velocity.n_dofs_per_cell(),
                                      fe_pressure.n_dofs_per_cell());
    InitGradScratchData scratch_data(fe_velocity,
                                     fe_pressure,
                                     quadrature_velocity,
                                     update_gradients | update_JxW_values,
                                     update_values);

    for (unsigned int d = 0; d < dim; ++d)
      {
        pres_Diff[d].reinit(sparsity_pattern_pres_vel);
        per_task_data.d = d;
        WorkStream::run(
          IteratorPair(IteratorTuple(dof_handler_velocity.begin_active(),
                                     dof_handler_pressure.begin_active())),
          IteratorPair(IteratorTuple(dof_handler_velocity.end(),
                                     dof_handler_pressure.end())),
          *this,
          &NavierStokesProjection<dim>::assemble_one_cell_of_gradient,
          &NavierStokesProjection<dim>::copy_gradient_local_to_global,
          scratch_data,
          per_task_data);
      }
  }

  template <int dim>
  void NavierStokesProjection<dim>::assemble_one_cell_of_gradient(
    const IteratorPair  &SI,
    InitGradScratchData &scratch,
    InitGradPerTaskData &data)
  {
    scratch.fe_val_vel.reinit(std::get<0>(*SI));
    scratch.fe_val_pres.reinit(std::get<1>(*SI));

    std::get<0>(*SI)->get_dof_indices(data.vel_local_dof_indices);
    std::get<1>(*SI)->get_dof_indices(data.pres_local_dof_indices);

    data.local_grad = 0.;
    for (unsigned int q = 0; q < scratch.nqp; ++q)
      {
        for (unsigned int i = 0; i < data.vel_dpc; ++i)
          for (unsigned int j = 0; j < data.pres_dpc; ++j)
            data.local_grad(i, j) +=
              -scratch.fe_val_vel.JxW(q) *
              scratch.fe_val_vel.shape_grad(i, q)[data.d] *
              scratch.fe_val_pres.shape_value(j, q);
      }
  }


  template <int dim>
  void NavierStokesProjection<dim>::copy_gradient_local_to_global(
    const InitGradPerTaskData &data)
  {
    for (unsigned int i = 0; i < data.vel_dpc; ++i)
      for (unsigned int j = 0; j < data.pres_dpc; ++j)
        pres_Diff[data.d].add(data.vel_local_dof_indices[i],
                              data.pres_local_dof_indices[j],
                              data.local_grad(i, j));
  }


  // @sect4{ <code>NavierStokesProjection::run</code> }

  // This is the time marching function, which starting at <code>t_0</code>
  // advances in time using the projection method with time step <code>dt</code>
  // until <code>T</code>.
  //
  // Its second parameter, <code>verbose</code> indicates whether the function
  // should output information what it is doing at any given moment: for
  // example, it will say whether we are working on the diffusion, projection
  // substep; updating preconditioners etc. Rather than implementing this
  // output using code like
  // @code
  //   if (verbose) std::cout << "something";
  // @endcode
  // we use the ConditionalOStream class to do that for us. That
  // class takes an output stream and a condition that indicates whether the
  // things you pass to it should be passed through to the given output
  // stream, or should just be ignored. This way, above code simply becomes
  // @code
  //   verbose_cout << "something";
  // @endcode
  // and does the right thing in either case.
  template <int dim>
  void NavierStokesProjection<dim>::run(const bool         verbose,
                                        const unsigned int output_interval)
  {
    ConditionalOStream verbose_cout(std::cout, verbose);

    const auto n_steps = static_cast<unsigned int>((T - t_0) / dt);
    vel_exact.set_time(2. * dt);
    output_results(1);
    // ===========================================================
    // Additional: Output initial solid particle positions
    // ===========================================================
    if (use_nitsche)
      output_solid_particles(1);

    for (unsigned int n = 2; n <= n_steps; ++n)
      {
        if (n % output_interval == 0)
          {
            verbose_cout << "Plotting Solution" << std::endl;
            output_results(n);
            if (use_nitsche)
              output_solid_particles(n); // Additional
          }

        std::cout << "Step = " << n << " Time = " << (n * dt) << std::endl;

        // ===========================================================
        // Additional: At the beginning of each time step, move solid particles from (n-1)*dt to n*dt
        // (following step-70's set_particle_positions approach)
        // ===========================================================
        if (use_nitsche)
          {
            verbose_cout << "  Moving solid particles" << std::endl;
            move_solid_particles(dt);
          }
        // ===========================================================

        verbose_cout << "  Interpolating the velocity" << std::endl;
        interpolate_velocity();

        verbose_cout << "  Diffusion Step" << std::endl;
        if (n % vel_update_prec == 0)
          verbose_cout << "    With reinitialization of the preconditioner"
                       << std::endl;
        diffusion_step((n % vel_update_prec == 0) || (n == 2));

        verbose_cout << "  Projection Step" << std::endl;
        projection_step((n == 2));

        verbose_cout << "  Updating the Pressure" << std::endl;
        update_pressure((n == 2));

        vel_exact.advance_time(dt);
      }
    output_results(n_steps);
  }



  template <int dim>
  void NavierStokesProjection<dim>::interpolate_velocity()
  {
    for (unsigned int d = 0; d < dim; ++d)
      {
        u_star[d].equ(2., u_n[d]);
        u_star[d] -= u_n_minus_1[d];
      }
  }


  // @sect4{<code>NavierStokesProjection::diffusion_step</code>}

  // The implementation of a diffusion step. Note that the expensive operation
  // is the diffusion solve at the end of the function, which we have to do once
  // for each velocity component. To accelerate things a bit, we allow to do
  // this in %parallel, using the Threads::new_task function which makes sure
  // that the <code>dim</code> solves are all taken care of and are scheduled to
  // available processors: if your machine has more than one processor core and
  // no other parts of this program are using resources currently, then the
  // diffusion solves will run in %parallel. On the other hand, if your system
  // has only one processor core then running things in %parallel would be
  // inefficient (since it leads, for example, to cache congestion) and things
  // will be executed sequentially.
  template <int dim>
  void NavierStokesProjection<dim>::diffusion_step(const bool reinit_prec)
  {
    pres_tmp.equ(-1., pres_n);
    pres_tmp.add(-4. / 3., phi_n, 1. / 3., phi_n_minus_1);

    assemble_advection_term();

    for (unsigned int d = 0; d < dim; ++d)
      {
        force[d] = 0.;
        v_tmp.equ(2. / dt, u_n[d]);
        v_tmp.add(-.5 / dt, u_n_minus_1[d]);
        vel_Mass.vmult_add(force[d], v_tmp);

        pres_Diff[d].vmult_add(force[d], pres_tmp);
        u_n_minus_1[d] = u_n[d];

      vel_it_matrix[d].copy_from(vel_Laplace_plus_Mass);
      vel_it_matrix[d].add(1., vel_Advection);

      // ===========================================================
      // Sponge layer: Add damping term -σ(x)u to RHS and σ(x) to matrix
      // This absorbs outgoing disturbances near the outlet
      // ===========================================================
      if (use_sponge_layer)
        {
          FEValues<dim> fe_values(fe_velocity,
                                  quadrature_velocity,
                                  update_values | update_quadrature_points |
                                  update_JxW_values);

          const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();
          const unsigned int n_q_points    = quadrature_velocity.size();

          FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
          Vector<double>     cell_rhs(dofs_per_cell);
          std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
          std::vector<double> u_n_values(n_q_points);

          for (const auto &cell : dof_handler_velocity.active_cell_iterators())
            {
              fe_values.reinit(cell);
              cell_matrix = 0.0;
              cell_rhs    = 0.0;
              cell->get_dof_indices(local_dof_indices);

              // Get current velocity values at quadrature points
              fe_values.get_function_values(u_n[d], u_n_values);

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const Point<dim> &q_point = fe_values.quadrature_point(q);
                  const double sigma = compute_sponge_coefficient(q_point);

                  if (sigma > 0.0)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          const double phi_i = fe_values.shape_value(i, q);

                          // RHS: -σ(x) * u^n * φ_i * JxW
                          cell_rhs(i) -= sigma * u_n_values[q] * phi_i * fe_values.JxW(q);

                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              const double phi_j = fe_values.shape_value(j, q);
                              // Matrix: σ(x) * φ_i * φ_j * JxW
                              cell_matrix(i, j) += sigma * phi_i * phi_j * fe_values.JxW(q);
                            }
                        }
                    }
                }

              // Add local contributions to global system
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  force[d](local_dof_indices[i]) += cell_rhs(i);
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    vel_it_matrix[d].add(local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix(i, j));
                }
            }
        }
      // ===========================================================

      // ===========================================================
      // Additional: Add Nitsche penalty term (from step-70)
      // Must be called before apply_boundary_values()!
      // ===========================================================
      if (use_nitsche)
        assemble_nitsche_restriction(d, vel_it_matrix[d], force[d]);
      // ===========================================================

      // Set the component for vel_exact function (important for boundary condition application)
      vel_exact.set_component(d);

      boundary_values.clear();
      for (const auto &boundary_id : boundary_ids)
        {
          if (boundary_id >= 1 && boundary_id <= 6)
            {
              const std::string &bc_type = boundary_types[boundary_id];
              
              if (bc_type == "outflow")
                {
                  // Convective Outflow Condition (Energy-stable outflow)
                  // Implement weak boundary term: ∫_Γ max(u·n, 0) u·v dS
                  // This provides energy stability while allowing natural outflow
                  
                  // We need to assemble this term for each velocity component d
                  // The term is: ∫_Γ max(u·n, 0) u_d * v_d dS
                  // where u_d is the d-th component of velocity
                  
                  // Since we're in the loop for component d, we assemble only for this component
                  // The full vector term would require coupling between components,
                  // but for simplicity we treat each component separately
                  
                  // Setup face quadrature and FE values
                  QGauss<dim-1> face_quadrature(fe_velocity.degree + 1);
                  FEFaceValues<dim> fe_face(fe_velocity,
                                            face_quadrature,
                                            update_values |
                                            update_normal_vectors |
                                            update_JxW_values);
                  
                  const unsigned int n_face_q_points = face_quadrature.size();
                  const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();
                  
                  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
                  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
                  
                  // Vector to store u_star values at quadrature points
                  std::vector<double> u_star_values(n_face_q_points);
                  
                  // Loop over all cells to find faces on this boundary
                  for (auto &cell : dof_handler_velocity.active_cell_iterators())
                    {
                      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                        {
                          if (cell->face(face)->at_boundary() && 
                              cell->face(face)->boundary_id() == boundary_id)
                            {
                              fe_face.reinit(cell, face);
                              cell->get_dof_indices(local_dof_indices);
                              
                              local_matrix = 0.0;
                              
                              // Get u_star values at quadrature points for component d
                              fe_face.get_function_values(u_star[d], u_star_values);
                              
                              for (unsigned int q = 0; q < n_face_q_points; ++q)
                                {
                                  const Tensor<1,dim> normal = fe_face.normal_vector(q);
                                  const double JxW = fe_face.JxW(q);
                                  
                                  // Calculate normal velocity component
                                  // Note: This is an approximation - we use only component d
                                  // A more accurate implementation would use the full velocity vector
                                  double un = u_star_values[q] * normal[d];
                                  
                                  // α = max(un, 0) - only penalize outflow
                                  double alpha = std::max(un, 0.0);
                                  
                                  // Skip if alpha is zero (inflow or tangential flow)
                                  if (alpha == 0.0)
                                    continue;
                                  
                                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                    {
                                      const double phi_i = fe_face.shape_value(i, q);
                                      
                                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                        {
                                          const double phi_j = fe_face.shape_value(j, q);
                                          local_matrix(i, j) += alpha * phi_i * phi_j * JxW;
                                        }
                                    }
                                }
                              
                              // Add local contributions to global matrix
                              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                  vel_it_matrix[d].add(local_dof_indices[i], 
                                                       local_dof_indices[j], 
                                                       local_matrix(i, j));
                            }
                        }
                    }
                  
                  // Continue to next boundary (don't apply Dirichlet BC)
                  continue;
                }
              else if (bc_type == "outflow_normal")
                {
                  // For outflow_normal boundary condition:
                  // - For left/right walls (boundary_id = 1 or 2): 
                  //   * Normal direction is x-direction (wall normal points in x-direction)
                  //   * Apply convective outflow condition for normal (x) component (d == 0)
                  //   * Free slip for tangential (y) component (d == 1)
                  // - For bottom/top walls (boundary_id = 3 or 4):
                  //   * Normal direction is y-direction (wall normal points in y-direction)
                  //   * Apply convective outflow condition for normal (y) component (d == 1)
                  //   * Free slip for tangential (x) component (d == 0)
                  
                  if (boundary_id == 1 || boundary_id == 2)
                    {
                      // Left or right wall: normal is x-direction
                      if (d == 0) // x-component (normal direction)
                        {
                          // Apply convective outflow condition for normal component
                          // Setup face quadrature and FE values
                          QGauss<dim-1> face_quadrature(fe_velocity.degree + 1);
                          FEFaceValues<dim> fe_face(fe_velocity,
                                                    face_quadrature,
                                                    update_values |
                                                    update_normal_vectors |
                                                    update_JxW_values);
                          
                          const unsigned int n_face_q_points = face_quadrature.size();
                          const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();
                          
                          std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
                          FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
                          
                          // Vector to store u_star values at quadrature points
                          std::vector<double> u_star_values(n_face_q_points);
                          
                          // Loop over all cells to find faces on this boundary
                          for (auto &cell : dof_handler_velocity.active_cell_iterators())
                            {
                              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                                {
                                  if (cell->face(face)->at_boundary() && 
                                      cell->face(face)->boundary_id() == boundary_id)
                                    {
                                      fe_face.reinit(cell, face);
                                      cell->get_dof_indices(local_dof_indices);
                                      
                                      local_matrix = 0.0;
                                      
                                      // Get u_star values at quadrature points for component d (x-component)
                                      fe_face.get_function_values(u_star[d], u_star_values);
                                      
                                      for (unsigned int q = 0; q < n_face_q_points; ++q)
                                        {
                                          const Tensor<1,dim> normal = fe_face.normal_vector(q);
                                          const double JxW = fe_face.JxW(q);
                                          
                                          // Calculate normal velocity component
                                          // For left/right walls, normal is in x-direction
                                          double un = u_star_values[q] * normal[0]; // normal[0] is x-component of normal
                                          
                                          // α = max(un, 0) - only penalize outflow
                                          double alpha = std::max(un, 0.0);
                                          
                                          // Skip if alpha is zero (inflow or tangential flow)
                                          if (alpha == 0.0)
                                            continue;
                                          
                                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                            {
                                              const double phi_i = fe_face.shape_value(i, q);
                                              
                                              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                                {
                                                  const double phi_j = fe_face.shape_value(j, q);
                                                  local_matrix(i, j) += alpha * phi_i * phi_j * JxW;
                                                }
                                            }
                                        }
                                      
                                      // Add local contributions to global matrix
                                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                          vel_it_matrix[d].add(local_dof_indices[i], 
                                                               local_dof_indices[j], 
                                                               local_matrix(i, j));
                                    }
                                }
                            }
                        }
                      // For y-component (d == 1), do nothing (free slip)
                    }
                  else if (boundary_id == 3 || boundary_id == 4)
                    {
                      // Bottom or top wall: normal is y-direction
                      if (d == 1) // y-component (normal direction)
                        {
                          // Apply convective outflow condition for normal component
                          // Setup face quadrature and FE values
                          QGauss<dim-1> face_quadrature(fe_velocity.degree + 1);
                          FEFaceValues<dim> fe_face(fe_velocity,
                                                    face_quadrature,
                                                    update_values |
                                                    update_normal_vectors |
                                                    update_JxW_values);
                          
                          const unsigned int n_face_q_points = face_quadrature.size();
                          const unsigned int dofs_per_cell = fe_velocity.n_dofs_per_cell();
                          
                          std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
                          FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
                          
                          // Vector to store u_star values at quadrature points
                          std::vector<double> u_star_values(n_face_q_points);
                          
                          // Loop over all cells to find faces on this boundary
                          for (auto &cell : dof_handler_velocity.active_cell_iterators())
                            {
                              for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                                {
                                  if (cell->face(face)->at_boundary() && 
                                      cell->face(face)->boundary_id() == boundary_id)
                                    {
                                      fe_face.reinit(cell, face);
                                      cell->get_dof_indices(local_dof_indices);
                                      
                                      local_matrix = 0.0;
                                      
                                      // Get u_star values at quadrature points for component d (y-component)
                                      fe_face.get_function_values(u_star[d], u_star_values);
                                      
                                      for (unsigned int q = 0; q < n_face_q_points; ++q)
                                        {
                                          const Tensor<1,dim> normal = fe_face.normal_vector(q);
                                          const double JxW = fe_face.JxW(q);
                                          
                                          // Calculate normal velocity component
                                          // For bottom/top walls, normal is in y-direction
                                          double un = u_star_values[q] * normal[1]; // normal[1] is y-component of normal
                                          
                                          // α = max(un, 0) - only penalize outflow
                                          double alpha = std::max(un, 0.0);
                                          
                                          // Skip if alpha is zero (inflow or tangential flow)
                                          if (alpha == 0.0)
                                            continue;
                                          
                                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                            {
                                              const double phi_i = fe_face.shape_value(i, q);
                                              
                                              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                                {
                                                  const double phi_j = fe_face.shape_value(j, q);
                                                  local_matrix(i, j) += alpha * phi_i * phi_j * JxW;
                                                }
                                            }
                                        }
                                      
                                      // Add local contributions to global matrix
                                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                                          vel_it_matrix[d].add(local_dof_indices[i], 
                                                               local_dof_indices[j], 
                                                               local_matrix(i, j));
                                    }
                                }
                            }
                        }
                      // For x-component (d == 0), do nothing (free slip)
                    }
                  continue;
                }
              else if (bc_type == "zero")
                {
                  // Zero velocity boundary condition
                  VectorTools::interpolate_boundary_values(
                    dof_handler_velocity,
                    boundary_id,
                    Functions::ZeroFunction<dim>(),
                    boundary_values);
                }
              else if (bc_type == "constant")
                {
                  // Constant velocity boundary condition
                  if (d == 0) // x-component
                    {
                      Functions::ConstantFunction<dim> constant_bc(boundary_values_x[boundary_id]);
                      VectorTools::interpolate_boundary_values(
                        dof_handler_velocity,
                        boundary_id,
                        constant_bc,
                        boundary_values);
                    }
                  else if (d == 1) // y-component
                    {
                      Functions::ConstantFunction<dim> constant_bc(boundary_values_y[boundary_id]);
                      VectorTools::interpolate_boundary_values(
                        dof_handler_velocity,
                        boundary_id,
                        constant_bc,
                        boundary_values);
                    }
                }
              else if (bc_type == "parabolic")
                {
                  // Parabolic velocity profile (only for x-component)
                  if (d == 0)
                    {
                      // Create a custom function for parabolic profile
                      class ParabolicFunction : public Function<dim>
                      {
                      public:
                        ParabolicFunction(double max_vel, double channel_width)
                          : Function<dim>(1)
                          , Um(max_vel)
                          , H(channel_width)
                        {}
                        
                        virtual double value(const Point<dim> &p,
                                             unsigned int component = 0) const override
                        {
                          (void)component;
                          return 4. * Um * p[1] * (H - p[1]) / (H * H);
                        }
                        
                      private:
                        double Um;
                        double H;
                      };
                      
                      ParabolicFunction parabolic_bc(boundary_max_vel[boundary_id], channel_width);
                      VectorTools::interpolate_boundary_values(
                        dof_handler_velocity,
                        boundary_id,
                        parabolic_bc,
                        boundary_values);
                    }
                  else if (d == 1)
                    {
                      // For y-component, use zero velocity
                      VectorTools::interpolate_boundary_values(
                        dof_handler_velocity,
                        boundary_id,
                        Functions::ZeroFunction<dim>(),
                        boundary_values);
                    }
                }
            }
        }
        MatrixTools::apply_boundary_values(boundary_values,
                                           vel_it_matrix[d],
                                           u_n[d],
                                           force[d]);
      }


    Threads::TaskGroup<void> tasks;
    for (unsigned int d = 0; d < dim; ++d)
      {
        if (reinit_prec)
          prec_velocity[d].initialize(vel_it_matrix[d],
                                      SparseILU<double>::AdditionalData(
                                        vel_diag_strength, vel_off_diagonals));
        tasks += Threads::new_task(
          &NavierStokesProjection<dim>::diffusion_component_solve, *this, d);
      }
    tasks.join_all();
  }



  template <int dim>
  void
  NavierStokesProjection<dim>::diffusion_component_solve(const unsigned int d)
  {
    SolverControl solver_control(vel_max_its, vel_eps * force[d].l2_norm());
    SolverGMRES<Vector<double>> gmres(
      solver_control,
      SolverGMRES<Vector<double>>::AdditionalData(vel_Krylov_size));
    gmres.solve(vel_it_matrix[d], u_n[d], force[d], prec_velocity[d]);
  }


  // @sect4{ <code>NavierStokesProjection::assemble_advection_term</code> }

  // The following few functions deal with assembling the advection terms, which
  // is the part of the system matrix for the diffusion step that changes at
  // every time step. As mentioned above, we will run the assembly loop over all
  // cells in %parallel, using the WorkStream class and other
  // facilities as described in the documentation topic on @ref threads.
  template <int dim>
  void NavierStokesProjection<dim>::assemble_advection_term()
  {
    vel_Advection = 0.;
    AdvectionPerTaskData data(fe_velocity.n_dofs_per_cell());
    AdvectionScratchData scratch(fe_velocity,
                                 quadrature_velocity,
                                 update_values | update_JxW_values |
                                   update_gradients);
    WorkStream::run(
      dof_handler_velocity.begin_active(),
      dof_handler_velocity.end(),
      *this,
      &NavierStokesProjection<dim>::assemble_one_cell_of_advection,
      &NavierStokesProjection<dim>::copy_advection_local_to_global,
      scratch,
      data);
  }



  template <int dim>
  void NavierStokesProjection<dim>::assemble_one_cell_of_advection(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    AdvectionScratchData                                 &scratch,
    AdvectionPerTaskData                                 &data)
  {
    scratch.fe_val.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);
    for (unsigned int d = 0; d < dim; ++d)
      {
        scratch.fe_val.get_function_values(u_star[d], scratch.u_star_tmp);
        for (unsigned int q = 0; q < scratch.nqp; ++q)
          scratch.u_star_local[q][d] = scratch.u_star_tmp[q];
      }

    for (unsigned int d = 0; d < dim; ++d)
      {
        scratch.fe_val.get_function_gradients(u_star[d], scratch.grad_u_star);
        for (unsigned int q = 0; q < scratch.nqp; ++q)
          {
            if (d == 0)
              scratch.u_star_tmp[q] = 0.;
            scratch.u_star_tmp[q] += scratch.grad_u_star[q][d];
          }
      }

    data.local_advection = 0.;
    for (unsigned int q = 0; q < scratch.nqp; ++q)
      for (unsigned int i = 0; i < scratch.dpc; ++i)
        for (unsigned int j = 0; j < scratch.dpc; ++j)
          data.local_advection(i, j) += (scratch.u_star_local[q] *            //
                                           scratch.fe_val.shape_grad(j, q) *  //
                                           scratch.fe_val.shape_value(i, q)   //
                                         +                                    //
                                         0.5 *                                //
                                           scratch.u_star_tmp[q] *            //
                                           scratch.fe_val.shape_value(i, q) * //
                                           scratch.fe_val.shape_value(j, q))  //
                                        * scratch.fe_val.JxW(q);
  }



  template <int dim>
  void NavierStokesProjection<dim>::copy_advection_local_to_global(
    const AdvectionPerTaskData &data)
  {
    for (unsigned int i = 0; i < fe_velocity.n_dofs_per_cell(); ++i)
      for (unsigned int j = 0; j < fe_velocity.n_dofs_per_cell(); ++j)
        vel_Advection.add(data.local_dof_indices[i],
                          data.local_dof_indices[j],
                          data.local_advection(i, j));
  }



  // @sect4{<code>NavierStokesProjection::projection_step</code>}

  // This implements the projection step:
  template <int dim>
  void NavierStokesProjection<dim>::projection_step(const bool reinit_prec)
  {
    pres_iterative.copy_from(pres_Laplace);

    pres_tmp = 0.;
    for (unsigned d = 0; d < dim; ++d)
      pres_Diff[d].Tvmult_add(pres_tmp, u_n[d]);

    phi_n_minus_1 = phi_n;

    static std::map<types::global_dof_index, double> bval;
    if (reinit_prec)
      {
        bval.clear();
        // Apply pressure boundary condition on right wall (boundary 2)
        // This is similar to step-35.cc.original which uses boundary 3
        // We always use boundary 2 regardless of its velocity boundary condition type
        VectorTools::interpolate_boundary_values(dof_handler_pressure,
                                                 2,
                                                 Functions::ZeroFunction<dim>(),
                                                 bval);
      }

    MatrixTools::apply_boundary_values(bval, pres_iterative, phi_n, pres_tmp);

    if (reinit_prec)
      prec_pres_Laplace.initialize(pres_iterative,
                                   SparseILU<double>::AdditionalData(
                                     vel_diag_strength, vel_off_diagonals));

    SolverControl solvercontrol(vel_max_its, vel_eps * pres_tmp.l2_norm());
    SolverCG<Vector<double>> cg(solvercontrol);
    cg.solve(pres_iterative, phi_n, pres_tmp, prec_pres_Laplace);

    phi_n *= 1.5 / dt;
  }


  // @sect4{ <code>NavierStokesProjection::update_pressure</code> }

  // This is the pressure update step of the projection method. It implements
  // the standard formulation of the method, that is @f[ p^{n+1} = p^n +
  // \phi^{n+1}, @f] or the rotational form, which is @f[ p^{n+1} = p^n +
  // \phi^{n+1} - \frac{1}{Re} \nabla\cdot u^{n+1}. @f]
  template <int dim>
  void NavierStokesProjection<dim>::update_pressure(const bool reinit_prec)
  {
    pres_n_minus_1 = pres_n;
    switch (type)
      {
        case RunTimeParameters::Method::standard:
          pres_n += phi_n;
          break;
        case RunTimeParameters::Method::rotational:
          if (reinit_prec)
            prec_mass.initialize(pres_Mass);
          pres_n = pres_tmp;
          prec_mass.solve(pres_n);
          pres_n.sadd(1. / Re, 1., pres_n_minus_1);
          pres_n += phi_n;
          break;
        default:
          DEAL_II_NOT_IMPLEMENTED();
      };
  }


  // @sect4{ <code>NavierStokesProjection::output_results</code> }

  // This method plots the current solution. The main difficulty is that we want
  // to create a single output file that contains the data for all velocity
  // components, the pressure, and also the vorticity of the flow. On the other
  // hand, velocities and the pressure live on separate DoFHandler objects, and
  // so can't be written to the same file using a single DataOut object. As a
  // consequence, we have to work a bit harder to get the various pieces of data
  // into a single DoFHandler object, and then use that to drive graphical
  // output.
  //
  // We will not elaborate on this process here, but rather refer to step-32,
  // where a similar procedure is used (and is documented) to create a joint
  // DoFHandler object for all variables.
  //
  // Let us also note that we here compute the vorticity as a scalar quantity in
  // a separate function, using the $L^2$ projection of the quantity
  // $\text{curl} u$ onto the finite element space used for the components of
  // the velocity. In principle, however, we could also have computed it as a
  // pointwise quantity from the velocity, and do so through the
  // DataPostprocessor mechanism discussed in step-29 and step-33.
  template <int dim>
  void NavierStokesProjection<dim>::output_results(const unsigned int step)
  {
    assemble_vorticity((step == 1));
    const FESystem<dim> joint_fe(fe_velocity ^ dim, fe_pressure, fe_velocity);
    DoFHandler<dim>     joint_dof_handler(triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);
    Assert(joint_dof_handler.n_dofs() ==
             ((dim + 1) * dof_handler_velocity.n_dofs() +
              dof_handler_pressure.n_dofs()),
           ExcInternalError());
    Vector<double> joint_solution(joint_dof_handler.n_dofs());
    std::vector<types::global_dof_index> loc_joint_dof_indices(
      joint_fe.n_dofs_per_cell()),
      loc_vel_dof_indices(fe_velocity.n_dofs_per_cell()),
      loc_pres_dof_indices(fe_pressure.n_dofs_per_cell());
    typename DoFHandler<dim>::active_cell_iterator
      joint_cell = joint_dof_handler.begin_active(),
      joint_endc = joint_dof_handler.end(),
      vel_cell   = dof_handler_velocity.begin_active(),
      pres_cell  = dof_handler_pressure.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell, ++pres_cell)
      {
        joint_cell->get_dof_indices(loc_joint_dof_indices);
        vel_cell->get_dof_indices(loc_vel_dof_indices);
        pres_cell->get_dof_indices(loc_pres_dof_indices);
        for (unsigned int i = 0; i < joint_fe.n_dofs_per_cell(); ++i)
          switch (joint_fe.system_to_base_index(i).first.first)
            {
              case 0:
                Assert(joint_fe.system_to_base_index(i).first.second < dim,
                       ExcInternalError());
                joint_solution(loc_joint_dof_indices[i]) =
                  u_n[joint_fe.system_to_base_index(i).first.second](
                    loc_vel_dof_indices[joint_fe.system_to_base_index(i)
                                          .second]);
                break;
              case 1:
                Assert(joint_fe.system_to_base_index(i).first.second == 0,
                       ExcInternalError());
                joint_solution(loc_joint_dof_indices[i]) =
                  pres_n(loc_pres_dof_indices[joint_fe.system_to_base_index(i)
                                                .second]);
                break;
              case 2:
                Assert(joint_fe.system_to_base_index(i).first.second == 0,
                       ExcInternalError());
                joint_solution(loc_joint_dof_indices[i]) = rot_u(
                  loc_vel_dof_indices[joint_fe.system_to_base_index(i).second]);
                break;
              default:
                DEAL_II_ASSERT_UNREACHABLE();
            }
      }
    std::vector<std::string> joint_solution_names(dim, "v");
    joint_solution_names.emplace_back("p");
    joint_solution_names.emplace_back("rot_u");
    DataOut<dim> data_out;
    data_out.attach_dof_handler(joint_dof_handler);
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        dim + 2, DataComponentInterpretation::component_is_part_of_vector);
    component_interpretation[dim] =
      DataComponentInterpretation::component_is_scalar;
    component_interpretation[dim + 1] =
      DataComponentInterpretation::component_is_scalar;
    data_out.add_data_vector(joint_solution,
                             joint_solution_names,
                             DataOut<dim>::type_dof_data,
                             component_interpretation);
    data_out.build_patches(deg + 1);
    std::ofstream output("solution-" + Utilities::int_to_string(step, 5) +
                         ".vtk");
    data_out.write_vtk(output);
  }



  // Following is the helper function that computes the vorticity by projecting
  // the term $\text{curl} u$ onto the finite element space used for the
  // components of the velocity. The function is only called whenever we
  // generate graphical output, so not very often, and as a consequence we
  // didn't bother parallelizing it using the WorkStream concept as we do for
  // the other assembly functions. That should not be overly complicated,
  // however, if needed. Moreover, the implementation that we have here only
  // works for 2d, so we bail if that is not the case.
  template <int dim>
  void NavierStokesProjection<dim>::assemble_vorticity(const bool reinit_prec)
  {
    Assert(dim == 2, ExcNotImplemented());
    if (reinit_prec)
      prec_vel_mass.initialize(vel_Mass);

    FEValues<dim>      fe_val_vel(fe_velocity,
                             quadrature_velocity,
                             update_gradients | update_JxW_values |
                               update_values);
    const unsigned int dpc = fe_velocity.n_dofs_per_cell(),
                       nqp = quadrature_velocity.size();
    std::vector<types::global_dof_index> ldi(dpc);
    Vector<double>                       loc_rot(dpc);

    std::vector<Tensor<1, dim>> grad_u1(nqp), grad_u2(nqp);
    rot_u = 0.;

    for (const auto &cell : dof_handler_velocity.active_cell_iterators())
      {
        fe_val_vel.reinit(cell);
        cell->get_dof_indices(ldi);
        fe_val_vel.get_function_gradients(u_n[0], grad_u1);
        fe_val_vel.get_function_gradients(u_n[1], grad_u2);
        loc_rot = 0.;
        for (unsigned int q = 0; q < nqp; ++q)
          for (unsigned int i = 0; i < dpc; ++i)
            loc_rot(i) += (grad_u2[q][0] - grad_u1[q][1]) * //
                          fe_val_vel.shape_value(i, q) *    //
                          fe_val_vel.JxW(q);

        for (unsigned int i = 0; i < dpc; ++i)
          rot_u(ldi[i]) += loc_rot(i);
      }

    prec_vel_mass.solve(rot_u);
  }

    // ================================================================
    // FSI additional method implementations
    // ================================================================

    template <int dim>
    void NavierStokesProjection<dim>::setup_solid_particles()
    {
      // ----------------------------------------------------------
      // 1. Create solid mesh (disk) and refine globally
      // ----------------------------------------------------------
      GridGenerator::hyper_ball(solid_triangulation,
                                 solid_center,
                                 solid_radius);
      solid_triangulation.refine_global(
        static_cast<unsigned int>(
          std::round(std::log2(solid_radius /
                               GridTools::minimal_cell_diameter(triangulation)))));

      solid_dof_handler.distribute_dofs(solid_fe_q);

      // ----------------------------------------------------------
      // 2. Compute Nitsche penalty parameter β/h (h is fluid mesh minimum cell diameter)
      // ----------------------------------------------------------
      nitsche_penalty_param =
        nitsche_beta / GridTools::minimal_cell_diameter(triangulation);

      std::cout << "Nitsche penalty parameter (beta/h) = "
                << nitsche_penalty_param << std::endl;

      // ----------------------------------------------------------
      // 3. Extract particle data from solid integration points
      //    Each particle stores: position (physical coordinates), JxW (property)
      // ----------------------------------------------------------
      std::vector<Point<dim>>          quadrature_points_vec;
      std::vector<std::vector<double>> properties;

      FEValues<dim> fe_v(solid_fe_q,
                         solid_quadrature,
                         update_quadrature_points | update_JxW_values);

      for (const auto &cell : solid_dof_handler.active_cell_iterators())
        {
          fe_v.reinit(cell);
          const auto &qp  = fe_v.get_quadrature_points();
          const auto &JxW = fe_v.get_JxW_values();

          for (unsigned int q = 0; q < solid_quadrature.size(); ++q)
            {
              quadrature_points_vec.push_back(qp[q]);
              properties.push_back({JxW[q]}); // Property: integration weight
            }
        }

      // ----------------------------------------------------------
      // 4. Insert particles into fluid triangulation
      //    Simplified serial version without MPI
      // ----------------------------------------------------------
      // For serial case, we can use a simpler approach
      // Create a map from cells to particles
      std::multimap<typename Triangulation<dim>::active_cell_iterator,
                    Particles::Particle<dim>>
        particles_map;

      // Find cells for each particle
      for (unsigned int i = 0; i < quadrature_points_vec.size(); ++i)
        {
          const auto cell = GridTools::find_active_cell_around_point(
            StaticMappingQ1<dim>::mapping, triangulation, quadrature_points_vec[i]);
          if (cell.first.state() == IteratorState::valid)
            {
              // Create particle
              Particles::Particle<dim> particle(quadrature_points_vec[i],
                                                cell.second,
                                                i);
              particles_map.insert(std::make_pair(cell.first, particle));
            }
        }

      // Insert particles using the map
      solid_particle_handler.insert_particles(particles_map);

      // Now set properties for each particle
      // We need to iterate through particles and set properties
      unsigned int particle_index = 0;
      for (auto particle = solid_particle_handler.begin();
           particle != solid_particle_handler.end();
           ++particle, ++particle_index)
        {
          if (particle_index < properties.size())
            {
              // Set properties for the particle
              particle->set_properties(properties[particle_index]);
            }
        }

      std::cout << "Solid particles inserted: "
                << solid_particle_handler.n_global_particles() << std::endl;
    }

  template <int dim>
  void NavierStokesProjection<dim>::move_solid_particles(const double current_dt)
  {
    const double dtheta = solid_omega * current_dt;
    const double cos_dt = std::cos(dtheta);
    const double sin_dt = std::sin(dtheta);

    // Move each particle's position (2D rotation transformation)
    for (auto &particle : solid_particle_handler)
      {
        Point<dim> p = particle.get_location();

        // Compute rotation relative to solid center
        const double rx = p[0] - solid_center[0];
        const double ry = p[1] - solid_center[1];

        Point<dim> new_pos;
        new_pos[0] = solid_center[0] + cos_dt * rx - sin_dt * ry;
        new_pos[1] = solid_center[1] + sin_dt * rx + cos_dt * ry;

        particle.set_location(new_pos);
      }

    // Re-sort particles into correct fluid cells (critical step!)
    solid_particle_handler.sort_particles_into_subdomains_and_cells();
  }

  template <int dim>
  void NavierStokesProjection<dim>::assemble_nitsche_restriction(
    const unsigned int    d,
    SparseMatrix<double> &vel_matrix,
    Vector<double>       &rhs_vector)
  {
    // Solid velocity field (rotational velocity)
    EquationData::SolidVelocity<dim> solid_vel(solid_omega);

    const unsigned int dpc = fe_velocity.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dpc);
    FullMatrix<double>                   local_matrix(dpc, dpc);
    Vector<double>                       local_rhs(dpc);

    // ----------------------------------------------------------------
    // Traverse all particles (step-70 pattern: find cell first, then traverse all particles in cell)
    // ----------------------------------------------------------------
    auto particle = solid_particle_handler.begin();
    while (particle != solid_particle_handler.end())
      {
        local_matrix = 0.0;
        local_rhs    = 0.0;

        // Get the fluid triangulation cell where the particle is located
        const auto &cell = particle->get_surrounding_cell();

        // Convert to velocity DoFHandler iterator, get global DoF indices
        const auto dh_cell =
          typename DoFHandler<dim>::cell_iterator(*cell, &dof_handler_velocity);
        dh_cell->get_dof_indices(local_dof_indices);

        // Traverse all solid particles in this cell
        const auto pic = solid_particle_handler.particles_in_cell(cell);
        Assert(pic.begin() == particle, ExcInternalError());

        for (const auto &p : pic)
          {
            // Reference coordinates (in fluid cell reference coordinate system)
            const Point<dim> ref_q  = p.get_reference_location();
            // Physical coordinates (in physical space)
            const Point<dim> real_q = p.get_location();
            // JxW weight (read from particle properties)
            const double     JxW    = p.get_properties()[0];

            // Current solid velocity value at component d
            const double us_d = solid_vel.value(real_q, d);

            // Assemble local contribution
            for (unsigned int i = 0; i < dpc; ++i)
              {
                // Compute basis function value at reference coordinates (FE_Q reference basis)
                const double phi_i = fe_velocity.shape_value(i, ref_q);

                // Right-hand side vector: β/h * u_s^d * φ_i * JxW
                local_rhs(i) += nitsche_penalty_param * us_d * phi_i * JxW;

                for (unsigned int j = 0; j < dpc; ++j)
                  {
                    const double phi_j = fe_velocity.shape_value(j, ref_q);
                    // Matrix term: β/h * φ_i * φ_j * JxW
                    local_matrix(i, j) +=
                      nitsche_penalty_param * phi_i * phi_j * JxW;
                  }
              }
          }

        // Accumulate local contributions to global matrix and vector
        for (unsigned int i = 0; i < dpc; ++i)
          {
            rhs_vector(local_dof_indices[i]) += local_rhs(i);
            for (unsigned int j = 0; j < dpc; ++j)
              vel_matrix.add(local_dof_indices[i],
                             local_dof_indices[j],
                             local_matrix(i, j));
          }

        // Jump to first particle of next cell
        particle = pic.end();
      }
  }

  template <int dim>
  void NavierStokesProjection<dim>::output_solid_particles(
    const unsigned int step)
  {
    Particles::DataOut<dim> particles_out;
    particles_out.build_patches(solid_particle_handler);

    std::ofstream output("solid-particles-" +
                         Utilities::int_to_string(step, 5) + ".vtu");
    particles_out.write_vtu(output);
  }

  // ================================================================
  // Sponge layer damping coefficient function
  // ================================================================
  template <int dim>
  double NavierStokesProjection<dim>::compute_sponge_coefficient(
    const Point<dim> &p) const
  {
    if (!use_sponge_layer)
      return 0.0;

    const double x = p[0];

    // If outside sponge zone, return 0
    if (x < sponge_x_start)
      return 0.0;

    // Normalized distance in sponge layer: ξ ∈ [0, 1]
    const double xi = (x - sponge_x_start) / (sponge_x_end - sponge_x_start);

    // Clamping to [0, 1]
    const double xi_clamped = std::min(1.0, std::max(0.0, xi));

    // Compute damping profile based on order
    double sigma = 0.0;
    switch (sponge_order)
      {
        case 1: // Linear profile
          sigma = sponge_strength * xi_clamped;
          break;
        case 2: // Quadratic profile
          sigma = sponge_strength * xi_clamped * xi_clamped;
          break;
        case 3: // Cubic profile
          sigma = sponge_strength * xi_clamped * xi_clamped * xi_clamped;
          break;
        default:
          sigma = sponge_strength * xi_clamped * xi_clamped; // Default to quadratic
          break;
      }

    return sigma;
  }
} // namespace Step35


// @sect3{ The main function }

// The main function looks very much like in all the other tutorial programs, so
// there is little to comment on here:
int main(int argc, char *argv[])
{
  try
    {
      using namespace Step35;

      // Determine parameter file name from command line or use default
      std::string parameter_filename = "parameter-file.prm";
      if (argc > 1)
        {
          parameter_filename = argv[1];
          std::cout << "Using parameter file: " << parameter_filename << std::endl;
        }
      else
        {
          std::cout << "Using default parameter file: " << parameter_filename << std::endl;
        }

      RunTimeParameters::Data_Storage data;
      data.read_data(parameter_filename);

      deallog.depth_console(data.verbose ? 2 : 0);

      NavierStokesProjection<2> test(data);
      test.run(data.verbose, data.output_interval);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  std::cout << "----------------------------------------------------"
            << std::endl
            << "Apparently everything went fine!" << std::endl
            << "Don't forget to brush your teeth :-)" << std::endl
            << std::endl;
  return 0;
}
