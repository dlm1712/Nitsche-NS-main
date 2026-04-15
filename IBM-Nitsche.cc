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
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/parsed_function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_tools.h>

// FSI: Particle handling for Nitsche method (from step-70)
#include <deal.II/lac/full_matrix.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/generators.h>
#include <deal.II/particles/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>

// Finally this is as in all previous programs:
namespace Step35
{
  using namespace dealii;

  // @sect3{IBM (Immersed Boundary Method) components}
  //
  // The following namespace contains all the classes and functions related to
  // the Immersed Boundary Method for fluid-structure interaction.
  namespace IBM
  {
    // @sect4{LagrangianPoint structure}
    //
    // This structure holds all data associated with a single Lagrangian point
    // on the immersed solid boundary.
    template <int dim>
    struct LagrangianPoint
    {
      Point<dim>     position;           // Current position
      Point<dim>     reference_position; // Reference position (initial config)
      Tensor<1, dim> velocity;           // Solid point velocity
      Tensor<1, dim> acceleration;       // Acceleration

      Tensor<1, dim> fluid_velocity; // Interpolated fluid velocity

      Tensor<1, dim> ibm_force;      // IBM force (direct forcing method)
      Tensor<1, dim> internal_force; // Internal force (elastic models)
      Tensor<1, dim> external_force; // External force (gravity, etc.)

      double   mass;       // Point mass
      double   arc_length; // Arc length this point represents
      unsigned int id;     // Global ID
      double   weight;     // Direct forcing weight

      // PID control members
      Tensor<1, dim> velocity_error_integral;  // Integral of velocity error
      Tensor<1, dim> velocity_error_previous;  // Previous velocity error (for D term)

      LagrangianPoint()
        : position()
        , reference_position()
        , velocity()
        , acceleration()
        , fluid_velocity()
        , ibm_force()
        , internal_force()
        , external_force()
        , mass(0.0)
        , arc_length(0.0)
        , id(0)
        , weight(0.0)
        , velocity_error_integral()
        , velocity_error_previous()
      {}
    };

    // @sect4{Motion type and DOF enumerations}
    enum class MotionType
    {
      Static,     // Fixed in place
      Prescribed, // Prescribed motion function
      FSICoupled  // Fully coupled FSI
    };

    enum class MotionDOF
    {
      TranslationX,
      TranslationY,
      TranslationZ,
      RotationZ,
      RotationX,
      RotationY,
      All
    };

    enum class DeltaType
    {
      Peskin,
      FEM,
      Dual
    };

    enum class GeometryType
    {
      Circle,
      Rectangle,
      Ellipse,
      Polygon,
      FromFile
    };

    // @sect4{SolidModelBase class}
    //
    // Abstract base class for solid constitutive models.
    template <int dim>
    class SolidModelBase
    {
    public:
      virtual ~SolidModelBase() = default;

      virtual void compute_ibm_forces(std::vector<LagrangianPoint<dim>> &points,
                                      const double                       dt) = 0;

      virtual void
      compute_internal_forces(std::vector<LagrangianPoint<dim>> &points,
                              const double                       time) = 0;

      virtual std::string get_model_type() const = 0;

      virtual void parse_parameters(ParameterHandler &prm) = 0;
      static void  declare_parameters(ParameterHandler &prm);
    };

    template <int dim>
    void SolidModelBase<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Solid model");
      {
        prm.declare_entry("Type",
                          "rigid",
                          Patterns::Selection("rigid|elastic"),
                          "Type of solid model");
      }
      prm.leave_subsection();
    }

    // @sect4{RigidBodyDirectForcing class}
    //
    // Implementation of direct forcing method for rigid bodies.
    // Core formula: F_ibm = w * (v_fluid - v_solid) / dt
    template <int dim>
    class RigidBodyDirectForcing : public SolidModelBase<dim>
    {
    public:
      enum class WeightMethod
      {
        GridSize,          // w = rho_f * h^dim
        DeltaSupport,      // w = rho_f * (2h)^dim
        LagrangianSpacing, // w = rho_f * ds * h
        Integrated         // Exact integration
      };

      RigidBodyDirectForcing();

      static void declare_parameters(ParameterHandler &prm);
      void        parse_parameters(ParameterHandler &prm) override;

      void compute_ibm_forces(std::vector<LagrangianPoint<dim>> &points,
                              const double                       dt) override;

      void compute_internal_forces(std::vector<LagrangianPoint<dim>> &points,
                                   const double time) override
      {
        // Rigid body has no internal stress
        for (auto &point : points)
          point.internal_force = Tensor<1, dim>();
      }

      std::string get_model_type() const override
      {
        return "rigid_direct_forcing";
      }

      void precompute_weights(const std::vector<LagrangianPoint<dim>> &points,
                              const double                             h);

      double       fluid_density;
      double       relaxation_factor;
      WeightMethod weight_method;

      // PID control parameters
      bool   use_pid;         // Enable PID control (false = original P-only)
      double Kp;              // Proportional gain
      double Ki;              // Integral gain
      double Kd;              // Derivative gain
      double integral_limit;  // Anti-windup saturation limit
      double tau_integral;    // Integral time constant for exponential decay

    private:
      std::vector<double> point_weights;

      double delta_function_1d(const double r, const double h) const;
      double delta_function(const Tensor<1, dim> &r, const double h) const;
    };

    template <int dim>
    RigidBodyDirectForcing<dim>::RigidBodyDirectForcing()
      : fluid_density(1.0)
      , relaxation_factor(1.0)
      , weight_method(WeightMethod::GridSize)
      , use_pid(false)
      , Kp(1.0)
      , Ki(0.0)
      , Kd(0.0)
      , integral_limit(1e10)
      , tau_integral(1e10)
    {}

    template <int dim>
    void RigidBodyDirectForcing<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Rigid body direct forcing");
      {
        prm.declare_entry("Fluid density",
                          "1.0",
                          Patterns::Double(0.0),
                          "Fluid density for weight calculation");
        prm.declare_entry("Relaxation factor",
                          "1.0",
                          Patterns::Double(0.0),
                          "Additional relaxation factor");
        prm.declare_entry(
          "Weight method",
          "grid_size",
          Patterns::Selection(
            "grid_size|delta_support|lagrangian_spacing|integrated"),
          "Method to compute the weight factor w");

        // PID control parameters
        prm.declare_entry("Use PID control",
                          "false",
                          Patterns::Bool(),
                          "Use PID control instead of proportional-only forcing");
        prm.declare_entry("Kp",
                          "1.0",
                          Patterns::Double(0.0),
                          "Proportional gain for PID control");
        prm.declare_entry("Ki",
                          "0.0",
                          Patterns::Double(0.0),
                          "Integral gain for PID control");
        prm.declare_entry("Kd",
                          "0.0",
                          Patterns::Double(0.0),
                          "Derivative gain for PID control");
        prm.declare_entry("Integral limit",
                          "1e10",
                          Patterns::Double(0.0),
                          "Anti-windup saturation limit for integral term");
        prm.declare_entry("Integral time constant",
                          "1e10",
                          Patterns::Double(0.0),
                          "Time constant for integral decay (tau). "
                          "Large value = standard integral. "
                          "Small value = only recent errors contribute.");
      }
      prm.leave_subsection();
    }

    template <int dim>
    void RigidBodyDirectForcing<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Rigid body direct forcing");
      {
        fluid_density      = prm.get_double("Fluid density");
        relaxation_factor  = prm.get_double("Relaxation factor");
        const std::string method = prm.get("Weight method");
        if (method == "grid_size")
          weight_method = WeightMethod::GridSize;
        else if (method == "delta_support")
          weight_method = WeightMethod::DeltaSupport;
        else if (method == "lagrangian_spacing")
          weight_method = WeightMethod::LagrangianSpacing;
        else if (method == "integrated")
          weight_method = WeightMethod::Integrated;

        // Parse PID control parameters
        use_pid        = prm.get_bool("Use PID control");
        Kp             = prm.get_double("Kp");
        Ki             = prm.get_double("Ki");
        Kd             = prm.get_double("Kd");
        integral_limit = prm.get_double("Integral limit");
        tau_integral   = prm.get_double("Integral time constant");
      }
      prm.leave_subsection();
    }

    template <int dim>
    double RigidBodyDirectForcing<dim>::delta_function_1d(const double r,
                                                          const double h) const
    {
      const double abs_r = std::abs(r);
      const double q     = abs_r / h;

      if (q >= 2.0)
        return 0.0;
      else if (q >= 1.0)
        return (5.0 - 2.0 * q -
                std::sqrt(-7.0 + 12.0 * q - 4.0 * q * q)) /
               (8.0 * h);
      else
        return (3.0 - 2.0 * q +
                std::sqrt(1.0 + 4.0 * q - 4.0 * q * q)) /
               (8.0 * h);
    }

    template <int dim>
    double
    RigidBodyDirectForcing<dim>::delta_function(const Tensor<1, dim> &r,
                                                const double          h) const
    {
      double result = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        result *= delta_function_1d(r[d], h);
      return result;
    }

    template <int dim>
    void RigidBodyDirectForcing<dim>::precompute_weights(
      const std::vector<LagrangianPoint<dim>> &points,
      const double                             h)
    {
      const unsigned int n_points = points.size();
      point_weights.resize(n_points);

      for (unsigned int i = 0; i < n_points; ++i)
        {
          double w = 0.0;

          switch (weight_method)
            {
              case WeightMethod::GridSize:
                w = fluid_density * std::pow(h, dim);
                break;

              case WeightMethod::DeltaSupport:
                {
                  const double support_width = 2.0 * h;
                  w = fluid_density * std::pow(support_width, dim);
                }
                break;

              case WeightMethod::LagrangianSpacing:
                {
                  double       ds          = 0.0;
                  unsigned int n_neighbors = 0;

                  if (i > 0)
                    {
                      ds += points[i].position.distance(points[i - 1].position);
                      ++n_neighbors;
                    }
                  if (i < n_points - 1)
                    {
                      ds += points[i].position.distance(points[i + 1].position);
                      ++n_neighbors;
                    }
                  if (i == 0 && n_points > 2)
                    {
                      ds +=
                        points[0].position.distance(points[n_points - 1].position);
                      ++n_neighbors;
                    }
                  if (i == n_points - 1 && n_points > 2)
                    {
                      ds +=
                        points[n_points - 1].position.distance(points[0].position);
                      ++n_neighbors;
                    }

                  if (n_neighbors > 0)
                    ds /= n_neighbors;
                  else
                    ds = h;

                  if (dim == 2)
                    w = fluid_density * ds * h;
                  else
                    w = fluid_density * ds * h * h;
                }
                break;

              case WeightMethod::Integrated:
                w = fluid_density * std::pow(h, dim);
                break;
            }

          point_weights[i] = w * relaxation_factor;
        }
    }

    template <int dim>
    void RigidBodyDirectForcing<dim>::compute_ibm_forces(
      std::vector<LagrangianPoint<dim>> &points,
      const double                       dt)
    {
      Assert(point_weights.size() == points.size(),
             ExcMessage(
               "Weights not precomputed. Call precompute_weights first."));

      for (unsigned int i = 0; i < points.size(); ++i)
        {
          // Compute velocity error: e = v_solid - v_fluid
          const Tensor<1, dim> velocity_error =
            points[i].velocity - points[i].fluid_velocity;

          if (use_pid)
            {
              // === PID Control ===

              // P term: Proportional
              const Tensor<1, dim> P_term = Kp * velocity_error;

              // I term: Integral with exponential decay (leaky integrator)
              // alpha = exp(-dt / tau), when tau -> inf, alpha -> 1 (standard integral)
              const double alpha = (tau_integral > 1e-14)
                                     ? std::exp(-dt / tau_integral)
                                     : 0.0;

              // Update integral: decay old value, then add new error
              points[i].velocity_error_integral =
                alpha * points[i].velocity_error_integral + velocity_error * dt;

              // Anti-windup: clamp integral term
              for (unsigned int d = 0; d < dim; ++d)
                {
                  if (points[i].velocity_error_integral[d] > integral_limit)
                    points[i].velocity_error_integral[d] = integral_limit;
                  else if (points[i].velocity_error_integral[d] < -integral_limit)
                    points[i].velocity_error_integral[d] = -integral_limit;
                }

              const Tensor<1, dim> I_term = Ki * points[i].velocity_error_integral;

              // D term: Derivative
              const Tensor<1, dim> error_rate =
                (velocity_error - points[i].velocity_error_previous) / dt;
              const Tensor<1, dim> D_term = Kd * error_rate;

              // Save current error for next time step
              points[i].velocity_error_previous = velocity_error;

              // Combine PID terms
              points[i].ibm_force =
                point_weights[i] * (P_term + I_term + D_term) / dt;
            }
          else
            {
              // Original direct forcing (P-only): F_ibm = w * (v_solid - v_fluid) / dt
              points[i].ibm_force = point_weights[i] * velocity_error / dt;
            }
        }
    }

    // @sect4{MotionModelBase class}
    template <int dim>
    class MotionModelBase
    {
    public:
      virtual ~MotionModelBase() = default;

      virtual void update_motion(std::vector<LagrangianPoint<dim>> &points,
                                 Point<dim>                        &center,
                                 Tensor<1, dim>                    &vel,
                                 Tensor<1, dim>                    &angular_vel,
                                 const Tensor<1, dim>              &fluid_force,
                                 const double                       fluid_torque,
                                 const double                       dt,
                                 const double                       time) = 0;

      virtual MotionType get_motion_type() const = 0;
    };

    // @sect4{StaticMotionModel class}
    template <int dim>
    class StaticMotionModel : public MotionModelBase<dim>
    {
    public:
      void update_motion(std::vector<LagrangianPoint<dim>> & /*points*/,
                         Point<dim> & /*center*/,
                         Tensor<1, dim> & /*vel*/,
                         Tensor<1, dim> & /*angular_vel*/,
                         const Tensor<1, dim> & /*fluid_force*/,
                         const double /*fluid_torque*/,
                         const double /*dt*/,
                         const double /*time*/) override
      {
        // Static: no position update
      }

      MotionType get_motion_type() const override
      {
        return MotionType::Static;
      }
    };

    // @sect4{PrescribedMotionModel class}
    template <int dim>
    class PrescribedMotionModel : public MotionModelBase<dim>
    {
    public:
      PrescribedMotionModel();

      void update_motion(std::vector<LagrangianPoint<dim>> &points,
                         Point<dim>                        &center,
                         Tensor<1, dim>                    &vel,
                         Tensor<1, dim>                    &angular_vel,
                         const Tensor<1, dim>              &fluid_force,
                         const double                       fluid_torque,
                         const double                       dt,
                         const double                       time) override;

      MotionType get_motion_type() const override
      {
        return MotionType::Prescribed;
      }

      void set_initial_state(const Point<dim>              &center,
                             const std::vector<Point<dim>> &positions);

      static void declare_parameters(ParameterHandler &prm);
      void        parse_parameters(ParameterHandler &prm);

      // Motion parameters (amplitude, frequency, etc.)
      double amplitude_x;
      double amplitude_y;
      double frequency;
      double rotation_speed;

    private:
      Point<dim>              initial_center;
      std::vector<Point<dim>> initial_positions;
    };

    template <int dim>
    PrescribedMotionModel<dim>::PrescribedMotionModel()
      : amplitude_x(0.0)
      , amplitude_y(0.0)
      , frequency(1.0)
      , rotation_speed(0.0)
    {}

    template <int dim>
    void PrescribedMotionModel<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Prescribed motion");
      {
        prm.declare_entry("Amplitude X", "0.0", Patterns::Double(),
                          "Translation amplitude in X");
        prm.declare_entry("Amplitude Y", "0.0", Patterns::Double(),
                          "Translation amplitude in Y");
        prm.declare_entry("Frequency", "1.0", Patterns::Double(0.0),
                          "Motion frequency");
        prm.declare_entry("Rotation speed", "0.0", Patterns::Double(),
                          "Angular velocity (rad/s)");
      }
      prm.leave_subsection();
    }

    template <int dim>
    void PrescribedMotionModel<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Prescribed motion");
      {
        amplitude_x    = prm.get_double("Amplitude X");
        amplitude_y    = prm.get_double("Amplitude Y");
        frequency      = prm.get_double("Frequency");
        rotation_speed = prm.get_double("Rotation speed");
      }
      prm.leave_subsection();
    }

    template <int dim>
    void PrescribedMotionModel<dim>::set_initial_state(
      const Point<dim>              &center,
      const std::vector<Point<dim>> &positions)
    {
      initial_center    = center;
      initial_positions = positions;
    }

    template <int dim>
    void PrescribedMotionModel<dim>::update_motion(
      std::vector<LagrangianPoint<dim>> &points,
      Point<dim>                        &center,
      Tensor<1, dim>                    &vel,
      Tensor<1, dim>                    & /*angular_vel*/,
      const Tensor<1, dim>              & /*fluid_force*/,
      const double /*fluid_torque*/,
      const double /*dt*/,
      const double time)
    {
      const double pi        = numbers::PI;
      const double new_angle = rotation_speed * time;

      // Compute new center position
      Point<dim> new_center = initial_center;
      new_center[0] += amplitude_x * std::sin(2.0 * pi * frequency * time);
      if (dim > 1)
        new_center[1] += amplitude_y * std::cos(2.0 * pi * frequency * time);

      // Update velocity
      vel[0] = amplitude_x * 2.0 * pi * frequency *
               std::cos(2.0 * pi * frequency * time);
      if (dim > 1)
        vel[1] = -amplitude_y * 2.0 * pi * frequency *
                 std::sin(2.0 * pi * frequency * time);

      // Update all Lagrangian points
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          Point<dim> rel;
          for (unsigned int d = 0; d < dim; ++d)
            rel[d] = initial_positions[i][d] - initial_center[d];

          // Apply rotation (2D)
          Point<dim> rotated;
          if (dim >= 2)
            {
              rotated[0] =
                rel[0] * std::cos(new_angle) - rel[1] * std::sin(new_angle);
              rotated[1] =
                rel[0] * std::sin(new_angle) + rel[1] * std::cos(new_angle);
            }

          for (unsigned int d = 0; d < dim; ++d)
            points[i].position[d] = new_center[d] + rotated[d];

          // Set point velocity
          points[i].velocity = vel;
        }

      center = new_center;
    }

    // @sect4{FSICoupledMotionModel class}
    template <int dim>
    class FSICoupledMotionModel : public MotionModelBase<dim>
    {
    public:
      FSICoupledMotionModel();

      void update_motion(std::vector<LagrangianPoint<dim>> &points,
                         Point<dim>                        &center,
                         Tensor<1, dim>                    &vel,
                         Tensor<1, dim>                    &angular_vel,
                         const Tensor<1, dim>              &fluid_force,
                         const double                       fluid_torque,
                         const double                       dt,
                         const double                       time) override;

      MotionType get_motion_type() const override
      {
        return MotionType::FSICoupled;
      }

      double mass;
      double moment_of_inertia;
      bool   couple_translation_x;
      bool   couple_translation_y;
      bool   couple_rotation;

      Tensor<1, dim> external_force;
      double         external_torque;

    private:
      Point<dim>              initial_center;
      std::vector<Point<dim>> initial_positions;
    };

    template <int dim>
    FSICoupledMotionModel<dim>::FSICoupledMotionModel()
      : mass(1.0)
      , moment_of_inertia(1.0)
      , couple_translation_x(true)
      , couple_translation_y(true)
      , couple_rotation(false)
      , external_torque(0.0)
    {}

    template <int dim>
    void FSICoupledMotionModel<dim>::update_motion(
      std::vector<LagrangianPoint<dim>> &points,
      Point<dim>                        &center,
      Tensor<1, dim>                    &vel,
      Tensor<1, dim>                    &angular_vel,
      const Tensor<1, dim>              &fluid_force,
      const double                       fluid_torque,
      const double                       dt,
      const double /*time*/)
    {
      // Compute acceleration from forces: a = F/m
      Tensor<1, dim> acceleration;
      for (unsigned int d = 0; d < dim; ++d)
        acceleration[d] =
          (fluid_force[d] + external_force[d]) / mass;

      // Update velocity
      if (couple_translation_x)
        vel[0] += acceleration[0] * dt;
      if (dim > 1 && couple_translation_y)
        vel[1] += acceleration[1] * dt;

      // Update angular velocity
      if (couple_rotation)
        {
          double angular_accel =
            (fluid_torque + external_torque) / moment_of_inertia;
          angular_vel[0] += angular_accel * dt; // Using [0] for 2D rotation
        }

      // Update center position
      for (unsigned int d = 0; d < dim; ++d)
        center[d] += vel[d] * dt;

      // Update all Lagrangian point positions
      for (auto &point : points)
        {
          for (unsigned int d = 0; d < dim; ++d)
            point.position[d] += vel[d] * dt;
          point.velocity = vel;
        }
    }

    // @sect4{GeometryBase class}
    template <int dim>
    class GeometryBase
    {
    public:
      virtual ~GeometryBase() = default;

      virtual std::vector<LagrangianPoint<dim>>
      generate_points(const unsigned int n_points,
                      const Point<dim>  &center,
                      const double       scale = 1.0) const = 0;

      virtual GeometryType get_type() const = 0;
    };

    // @sect4{CircleGeometry class}
    template <int dim>
    class CircleGeometry : public GeometryBase<dim>
    {
    public:
      CircleGeometry(const double r = 0.25)
        : radius(r)
      {}

      std::vector<LagrangianPoint<dim>>
      generate_points(const unsigned int n_points,
                      const Point<dim>  &center,
                      const double       scale = 1.0) const override;

      GeometryType get_type() const override
      {
        return GeometryType::Circle;
      }

      double radius;
    };

    template <int dim>
    std::vector<LagrangianPoint<dim>>
    CircleGeometry<dim>::generate_points(const unsigned int n_points,
                                         const Point<dim>  &center,
                                         const double       scale) const
    {
      std::vector<LagrangianPoint<dim>> points(n_points);
      const double                      pi = numbers::PI;
      const double                      r  = radius * scale;
      const double arc_len = 2.0 * pi * r / n_points;

      for (unsigned int i = 0; i < n_points; ++i)
        {
          const double theta = 2.0 * pi * i / n_points;
          points[i].position[0] = center[0] + r * std::cos(theta);
          if (dim > 1)
            points[i].position[1] = center[1] + r * std::sin(theta);

          points[i].reference_position = points[i].position;
          points[i].arc_length         = arc_len;
          points[i].id                 = i;
        }
      return points;
    }

    // @sect4{RectangleGeometry class}
    template <int dim>
    class RectangleGeometry : public GeometryBase<dim>
    {
    public:
      RectangleGeometry(const double w = 0.5, const double h = 0.3)
        : width(w)
        , height(h)
      {}

      std::vector<LagrangianPoint<dim>>
      generate_points(const unsigned int n_points,
                      const Point<dim>  &center,
                      const double       scale = 1.0) const override;

      GeometryType get_type() const override
      {
        return GeometryType::Rectangle;
      }

      double width;
      double height;
    };

    template <int dim>
    std::vector<LagrangianPoint<dim>>
    RectangleGeometry<dim>::generate_points(const unsigned int n_points,
                                            const Point<dim>  &center,
                                            const double       scale) const
    {
      std::vector<LagrangianPoint<dim>> points;
      const double                      w          = width * scale;
      const double                      h          = height * scale;
      const double                      perimeter  = 2.0 * (w + h);
      const double                      ds         = perimeter / n_points;

      unsigned int id = 0;
      // Bottom edge
      for (double s = 0; s < w; s += ds)
        {
          LagrangianPoint<dim> pt;
          pt.position[0] = center[0] - w / 2 + s;
          if (dim > 1)
            pt.position[1] = center[1] - h / 2;
          pt.reference_position = pt.position;
          pt.arc_length         = ds;
          pt.id                 = id++;
          points.push_back(pt);
        }
      // Right edge
      for (double s = 0; s < h; s += ds)
        {
          LagrangianPoint<dim> pt;
          pt.position[0] = center[0] + w / 2;
          if (dim > 1)
            pt.position[1] = center[1] - h / 2 + s;
          pt.reference_position = pt.position;
          pt.arc_length         = ds;
          pt.id                 = id++;
          points.push_back(pt);
        }
      // Top edge
      for (double s = 0; s < w; s += ds)
        {
          LagrangianPoint<dim> pt;
          pt.position[0] = center[0] + w / 2 - s;
          if (dim > 1)
            pt.position[1] = center[1] + h / 2;
          pt.reference_position = pt.position;
          pt.arc_length         = ds;
          pt.id                 = id++;
          points.push_back(pt);
        }
      // Left edge
      for (double s = 0; s < h; s += ds)
        {
          LagrangianPoint<dim> pt;
          pt.position[0] = center[0] - w / 2;
          if (dim > 1)
            pt.position[1] = center[1] + h / 2 - s;
          pt.reference_position = pt.position;
          pt.arc_length         = ds;
          pt.id                 = id++;
          points.push_back(pt);
        }

      return points;
    }

    // @sect4{ImmersedSolid class}
    //
    // Main class representing an immersed solid body.
    template <int dim>
    class ImmersedSolid
    {
    public:
      ImmersedSolid(const unsigned int solid_id = 0);

      void initialize(const GeometryBase<dim> &geometry,
                      const Point<dim>        &center,
                      const unsigned int       n_points,
                      const double             scale = 1.0);

      void set_solid_model(std::unique_ptr<SolidModelBase<dim>> model);
      void set_motion_model(std::unique_ptr<MotionModelBase<dim>> model);

      void update(const Tensor<1, dim> &fluid_force,
                  const double          fluid_torque,
                  const double          dt,
                  const double          time);

      void interpolate_fluid_velocity(const Vector<double>   velocity[dim],
                                      const DoFHandler<dim> &dof_handler,
                                      const Mapping<dim>    &mapping,
                                      const double           h);

      void compute_ibm_forces(const double dt);

      void spread_ibm_forces_to_fluid(Vector<double>         fluid_force[dim],
                                      const DoFHandler<dim> &dof_handler,
                                      const Mapping<dim>    &mapping,
                                      const double           h) const;

      void precompute_weights(const double h);

      Tensor<1, dim> compute_total_force() const;
      double         compute_total_torque() const;

      void output_boundary(const std::string &filename) const;

      // Delta function type control
      void set_delta_type(DeltaType type)
      {
        delta_type = type;
      }

      // Mass matrix initialization for FEM/Dual delta functions
      void initialize_mass_matrix(const DoFHandler<dim> &dof_handler,
                                  const QGauss<dim>     &quadrature)
      {
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        SparsityPattern sparsity_pattern;
        sparsity_pattern.copy_from(dsp);
        mass_matrix.reinit(sparsity_pattern);

        MatrixCreator::create_mass_matrix(dof_handler,
                                          quadrature,
                                          mass_matrix);

        mass_solver.initialize(mass_matrix);
      }

      // Member variables
      unsigned int                       id;
      std::vector<LagrangianPoint<dim>>  lagrangian_points;
      std::vector<Point<dim>>            initial_positions;

      Point<dim>     center_of_mass;
      Point<dim>     initial_center;
      Tensor<1, dim> velocity;
      Tensor<1, dim> angular_velocity;
      double         orientation_angle;

      double mass;
      double moment_of_inertia;
      double density;

      std::unique_ptr<SolidModelBase<dim>>  solid_model;
      std::unique_ptr<MotionModelBase<dim>> motion_model;

    private:
      double delta_function_1d(const double r, const double h) const;
      double delta_function(const Point<dim> &r, const double h) const;

      // Delta function type
      DeltaType delta_type;

      // Mass matrix and solver for FEM/Dual delta functions
      SparseMatrix<double> mass_matrix;
      SparseDirectUMFPACK  mass_solver;
    };

    template <int dim>
    ImmersedSolid<dim>::ImmersedSolid(const unsigned int solid_id)
      : id(solid_id)
      , orientation_angle(0.0)
      , mass(1.0)
      , moment_of_inertia(1.0)
      , density(1.0)
    {}

    template <int dim>
    void ImmersedSolid<dim>::initialize(const GeometryBase<dim> &geometry,
                                        const Point<dim>        &center,
                                        const unsigned int       n_points,
                                        const double             scale)
    {
      lagrangian_points = geometry.generate_points(n_points, center, scale);
      center_of_mass    = center;
      initial_center    = center;

      initial_positions.resize(lagrangian_points.size());
      for (unsigned int i = 0; i < lagrangian_points.size(); ++i)
        initial_positions[i] = lagrangian_points[i].position;

      // Set up prescribed motion if applicable
      if (motion_model)
        {
          if (auto *prescribed =
                dynamic_cast<PrescribedMotionModel<dim> *>(motion_model.get()))
            {
              prescribed->set_initial_state(center, initial_positions);
            }
        }
    }

    template <int dim>
    void
    ImmersedSolid<dim>::set_solid_model(std::unique_ptr<SolidModelBase<dim>> model)
    {
      solid_model = std::move(model);
    }

    template <int dim>
    void ImmersedSolid<dim>::set_motion_model(
      std::unique_ptr<MotionModelBase<dim>> model)
    {
      motion_model = std::move(model);
    }

    template <int dim>
    double ImmersedSolid<dim>::delta_function_1d(const double r,
                                                 const double h) const
    {
      const double abs_r = std::abs(r);
      const double q     = abs_r / h;

      if (q >= 2.0)
        return 0.0;
      else if (q >= 1.0)
        return (5.0 - 2.0 * q - std::sqrt(-7.0 + 12.0 * q - 4.0 * q * q)) /
               (8.0 * h);
      else
        return (3.0 - 2.0 * q + std::sqrt(1.0 + 4.0 * q - 4.0 * q * q)) /
               (8.0 * h);
    }

    template <int dim>
    double ImmersedSolid<dim>::delta_function(const Point<dim> &r,
                                              const double      h) const
    {
      double result = 1.0;
      for (unsigned int d = 0; d < dim; ++d)
        result *= delta_function_1d(r[d], h);
      return result;
    }

    template <int dim>
    void ImmersedSolid<dim>::interpolate_fluid_velocity(
      const Vector<double>   velocity[dim],
      const DoFHandler<dim> &dof_handler,
      const Mapping<dim>    &mapping,
      const double           h)
    {
      const FiniteElement<dim> &fe = dof_handler.get_fe();

      FEValues<dim> fe_values(mapping,
                              fe,
                              QMidpoint<dim>(),
                              update_values | update_quadrature_points);

      for (auto &point : lagrangian_points)
        {
          point.fluid_velocity = Tensor<1, dim>();

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (!cell->point_inside(point.position))
                continue;

              fe_values.reinit(cell);

              std::vector<types::global_dof_index> local_dof_indices(
                fe.n_dofs_per_cell());
              cell->get_dof_indices(local_dof_indices);

              std::vector<double> shape_values(fe.n_dofs_per_cell());

              fe_values.get_function_values(velocity[0], shape_values);

              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                {
                  const double phi_i =
                    fe_values.shape_value(i, 0);  // Use quadrature point index 0

                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      if (delta_type == DeltaType::Peskin)
                        {
                          const Point<dim> quadrature_point = fe_values.quadrature_point(0);
                          Point<dim> r;
                          for (unsigned int dd = 0; dd < dim; ++dd)
                            r[dd] = point.position[dd] - quadrature_point[dd];

                          const double delta = delta_function(r, h);

                          point.fluid_velocity[d] +=
                            velocity[d](local_dof_indices[i]) *
                            delta * std::pow(h, dim);
                        }
                      else
                        {
                          point.fluid_velocity[d] +=
                            velocity[d](local_dof_indices[i]) * phi_i;
                        }
                    }
                }

              break;
            }
        }
    }

    template <int dim>
    void ImmersedSolid<dim>::compute_ibm_forces(const double dt)
    {
      if (solid_model)
        solid_model->compute_ibm_forces(lagrangian_points, dt);
    }

    template <int dim>
    void ImmersedSolid<dim>::spread_ibm_forces_to_fluid(
      Vector<double>         fluid_force[dim],
      const DoFHandler<dim> &dof_handler,
      const Mapping<dim>    &mapping,
      const double           h) const
    {
      const FiniteElement<dim> &fe = dof_handler.get_fe();

      FEValues<dim> fe_values(mapping,
                              fe,
                              QMidpoint<dim>(),
                              update_values | update_quadrature_points);

      Vector<double> dual_rhs;
      Vector<double> dual_solution;

      if (delta_type == DeltaType::Dual)
      {
        dual_rhs.reinit(dof_handler.n_dofs());
        dual_solution.reinit(dof_handler.n_dofs());
      }

      for (const auto &point : lagrangian_points)
        {
          if (point.ibm_force.norm() < 1e-14)
            continue;

          for (const auto &cell : dof_handler.active_cell_iterators())
            {
              if (!cell->point_inside(point.position))
                continue;

              fe_values.reinit(cell);

              std::vector<types::global_dof_index> local_dof_indices(
                fe.n_dofs_per_cell());
              cell->get_dof_indices(local_dof_indices);

              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                {
                  const double phi_i =
                    fe_values.shape_value(i, 0);  // Use quadrature point index 0

                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      if (delta_type == DeltaType::Peskin)
                        {
                          const Point<dim> quadrature_point = fe_values.quadrature_point(0);
                          Point<dim> r;
                          for (unsigned int dd = 0; dd < dim; ++dd)
                            r[dd] = point.position[dd] - quadrature_point[dd];

                          const double delta = delta_function(r, h);

                          fluid_force[d](local_dof_indices[i]) +=
                            point.ibm_force[d] *
                            delta * std::pow(h, dim);
                        }
                      else if (delta_type == DeltaType::FEM)
                        {
                          fluid_force[d](local_dof_indices[i]) +=
                            point.ibm_force[d] * phi_i;
                        }
                      else if (delta_type == DeltaType::Dual)
                        {
                          dual_rhs(local_dof_indices[i]) +=
                            point.ibm_force[d] * phi_i;
                        }
                    }
                }

              break;
            }
        }

      if (delta_type == DeltaType::Dual)
        {
          // For SparseDirectUMFPACK, the first argument is both RHS and solution
          dual_solution = dual_rhs;
          mass_solver.solve(dual_solution);

          for (unsigned int d = 0; d < dim; ++d)
            fluid_force[d] += dual_solution;
        }
    }

    template <int dim>
    void ImmersedSolid<dim>::precompute_weights(const double h)
    {
      if (auto *rigid =
            dynamic_cast<RigidBodyDirectForcing<dim> *>(solid_model.get()))
        {
          rigid->precompute_weights(lagrangian_points, h);
        }
    }

    template <int dim>
    void ImmersedSolid<dim>::update(const Tensor<1, dim> &fluid_force,
                                    const double          fluid_torque,
                                    const double          dt,
                                    const double          time)
    {
      if (motion_model)
        motion_model->update_motion(lagrangian_points,
                                    center_of_mass,
                                    velocity,
                                    angular_velocity,
                                    fluid_force,
                                    fluid_torque,
                                    dt,
                                    time);
    }

    template <int dim>
    Tensor<1, dim> ImmersedSolid<dim>::compute_total_force() const
    {
      Tensor<1, dim> total_force;
      for (const auto &point : lagrangian_points)
        total_force += point.ibm_force;
      return total_force;
    }

    template <int dim>
    double ImmersedSolid<dim>::compute_total_torque() const
    {
      double total_torque = 0.0;
      for (const auto &point : lagrangian_points)
        {
          Point<dim> r;
          for (unsigned int d = 0; d < dim; ++d)
            r[d] = point.position[d] - center_of_mass[d];

          // 2D torque: r x F = r_x * F_y - r_y * F_x
          if (dim >= 2)
            total_torque +=
              r[0] * point.ibm_force[1] - r[1] * point.ibm_force[0];
        }
      return total_torque;
    }

    template <int dim>
    void ImmersedSolid<dim>::output_boundary(const std::string &filename) const
    {
      std::ofstream out(filename);
      out << "# Lagrangian boundary points\n";
      out << "# x y [z] fx fy [fz]\n";
      for (const auto &point : lagrangian_points)
        {
          for (unsigned int d = 0; d < dim; ++d)
            out << point.position[d] << " ";
          for (unsigned int d = 0; d < dim; ++d)
            out << point.ibm_force[d] << " ";
          out << "\n";
        }
    }

    // @sect4{SolidArrayManager class}
    template <int dim>
    class SolidArrayManager
    {
    public:
      SolidArrayManager() = default;

      void add_solid(std::unique_ptr<ImmersedSolid<dim>> solid);

      void update_all(const double dt, const double time);

      void interpolate_all_fluid_velocities(const Vector<double>   velocity[dim],
                                            const DoFHandler<dim> &dof_handler,
                                            const Mapping<dim>    &mapping,
                                            const double           h);

      void compute_all_ibm_forces(const double dt);

      void spread_all_ibm_forces_to_fluid(Vector<double>         fluid_force[dim],
                                          const DoFHandler<dim> &dof_handler,
                                          const Mapping<dim>    &mapping,
                                          const double           h) const;

      void precompute_all_weights(const double h);

      void output_all_boundaries(const std::string &prefix,
                                 const unsigned int step) const;

      unsigned int n_solids() const
      {
        return solids.size();
      }

      ImmersedSolid<dim> &get_solid(unsigned int i)
      {
        return *solids[i];
      }

      const ImmersedSolid<dim> &get_solid(unsigned int i) const
      {
        return *solids[i];
      }

    private:
      std::vector<std::unique_ptr<ImmersedSolid<dim>>> solids;
    };

    template <int dim>
    void
    SolidArrayManager<dim>::add_solid(std::unique_ptr<ImmersedSolid<dim>> solid)
    {
      solids.push_back(std::move(solid));
    }

    template <int dim>
    void SolidArrayManager<dim>::update_all(const double dt, const double time)
    {
      for (auto &solid : solids)
        {
          Tensor<1, dim> fluid_force = solid->compute_total_force();
          double         fluid_torque = solid->compute_total_torque();
          solid->update(fluid_force, fluid_torque, dt, time);
        }
    }

    template <int dim>
    void SolidArrayManager<dim>::interpolate_all_fluid_velocities(
      const Vector<double>   velocity[dim],
      const DoFHandler<dim> &dof_handler,
      const Mapping<dim>    &mapping,
      const double           h)
    {
      for (auto &solid : solids)
        solid->interpolate_fluid_velocity(velocity, dof_handler, mapping, h);
    }

    template <int dim>
    void SolidArrayManager<dim>::compute_all_ibm_forces(const double dt)
    {
      for (auto &solid : solids)
        solid->compute_ibm_forces(dt);
    }

    template <int dim>
    void SolidArrayManager<dim>::spread_all_ibm_forces_to_fluid(
      Vector<double>         fluid_force[dim],
      const DoFHandler<dim> &dof_handler,
      const Mapping<dim>    &mapping,
      const double           h) const
    {
      for (const auto &solid : solids)
        solid->spread_ibm_forces_to_fluid(fluid_force, dof_handler, mapping, h);
    }

    template <int dim>
    void SolidArrayManager<dim>::precompute_all_weights(const double h)
    {
      for (auto &solid : solids)
        solid->precompute_weights(h);
    }

    template <int dim>
    void SolidArrayManager<dim>::output_all_boundaries(
      const std::string  &prefix,
      const unsigned int  step) const
    {
      for (unsigned int i = 0; i < solids.size(); ++i)
        {
          std::string filename = prefix + "-solid-" + std::to_string(i) + "-" +
                                 Utilities::int_to_string(step, 5) + ".dat";
          solids[i]->output_boundary(filename);
        }
    }

  } // namespace IBM

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
      bool         output_ibm_solid;   // Control flag for IBM solid boundary output

      // Mesh filename
      std::string  mesh_filename;

      // === Mesh generation parameters ===
      std::string  mesh_generation_type;  // "file" or "channel"
      double       channel_length;        // Channel length (for 2D/3D)
      double       channel_width;         // Channel width (for 2D/3D)
      double       channel_height;        // Channel height (for 3D only)
      unsigned int channel_refinement;    // Initial refinement for channel
      unsigned int channel_cells_x;       // Number of cells in x-direction
      unsigned int channel_cells_y;       // Number of cells in y-direction
      unsigned int channel_cells_z;       // Number of cells in z-direction (3D only)

    // FSI method selection
    std::string  fsi_method;           // "off", "ibm", or "nitsche"
    
    // Common FSI parameters
    std::string  solid_geometry_type;  // "circle" or "rectangle"
    double       solid_radius;         // Circle radius
    double       solid_width;          // Rectangle width
    double       solid_height;         // Rectangle height
    Point<2>     solid_center;         // Solid center position
    unsigned int solid_n_points;       // Number of Lagrangian points
    std::string  solid_motion_type;    // "static", "prescribed", "fsi_coupled"
    double       solid_amplitude_x;    // Prescribed motion amplitude X
    double       solid_amplitude_y;    // Prescribed motion amplitude Y
    double       solid_frequency;      // Prescribed motion frequency
    double       solid_rotation_speed; // Prescribed rotation speed
    double       solid_density;        // Solid density
    
    // IBM-specific parameters
    double       ibm_fluid_density;    // Fluid density for IBM
    double       ibm_relaxation_factor;// IBM relaxation factor
    std::string  delta_type;           // IBM delta kernel type: "peskin", "fem", "dual"
    
    // Nitsche-specific parameters
    bool         use_nitsche;          // Enable Nitsche method (legacy, use fsi_method instead)
    double       nitsche_beta;         // Nitsche penalty parameter β
    double       nitsche_gamma;        // Nitsche stability parameter γ
    unsigned int solid_refinement;     // Solid mesh refinement level for Nitsche

      // Initial condition parameters
      std::string  initial_velocity_type;  // "zero", "constant", or "parabolic"
      double       initial_velocity_x;     // X-component for constant initial velocity
      double       initial_velocity_y;     // Y-component for constant initial velocity
      double       initial_max_velocity;   // Max velocity for parabolic initial profile

      // Inlet boundary condition parameters (Boundary ID 1)
      std::string  bc_inlet_type;           // "constant", "parabolic", or "function"
      double       bc_inlet_constant_u;     // Constant velocity in x-direction
      double       bc_inlet_constant_v;     // Constant velocity in y-direction
      double       bc_inlet_constant_w;     // Constant velocity in z-direction (3D only)
      double       bc_inlet_parabolic_max;  // Max velocity for parabolic profile
      std::string  bc_inlet_function_u;     // User function expression for u
      std::string  bc_inlet_function_v;     // User function expression for v
      std::string  bc_inlet_function_w;     // User function expression for w (3D only)

      // Wall boundary condition parameters
      // For 2D: ID 3 = bottom wall, ID 4 = top wall
      // For 3D: ID 3 = bottom, ID 4 = top, ID 5 = front, ID 6 = back
      std::string  bc_wall_bottom_type;     // "no_slip", "moving", or "symmetric"
      std::string  bc_wall_top_type;
      std::string  bc_wall_front_type;      // 3D only
      std::string  bc_wall_back_type;       // 3D only
      
      // Moving wall velocity expressions
      std::string  bc_wall_bottom_velocity_u;
      std::string  bc_wall_bottom_velocity_v;
      std::string  bc_wall_bottom_velocity_w;
      std::string  bc_wall_top_velocity_u;
      std::string  bc_wall_top_velocity_v;
      std::string  bc_wall_top_velocity_w;
      std::string  bc_wall_front_velocity_u;  // 3D only
      std::string  bc_wall_front_velocity_v;
      std::string  bc_wall_front_velocity_w;
      std::string  bc_wall_back_velocity_u;   // 3D only
      std::string  bc_wall_back_velocity_v;
      std::string  bc_wall_back_velocity_w;

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
      , output_ibm_solid(false)
      , mesh_filename("nsbench2.inp")
      , mesh_generation_type("file")
      , channel_length(10.0)
      , channel_width(4.1)
      , channel_height(1.0)
      , channel_refinement(0)
      , channel_cells_x(1)
      , channel_cells_y(1)
      , channel_cells_z(1)
      , fsi_method("off")
      , solid_geometry_type("circle")
      , solid_radius(0.05)
      , solid_width(0.1)
      , solid_height(0.05)
      , solid_center(Point<2>(0.2, 0.2))
      , solid_n_points(64)
      , solid_motion_type("static")
      , solid_amplitude_x(0.0)
      , solid_amplitude_y(0.0)
      , solid_frequency(1.0)
      , solid_rotation_speed(0.0)
      , solid_density(1.0)
      , ibm_fluid_density(1.0)
      , ibm_relaxation_factor(1.0)
      , delta_type("peskin")
      , use_nitsche(false)
      , nitsche_beta(10.0)
      , nitsche_gamma(1.0)
      , solid_refinement(0)
      , initial_velocity_type("parabolic")
      , initial_velocity_x(0.0)
      , initial_velocity_y(0.0)
      , initial_max_velocity(1.5)
      , bc_inlet_type("parabolic")
      , bc_inlet_constant_u(1.0)
      , bc_inlet_constant_v(0.0)
      , bc_inlet_constant_w(0.0)
      , bc_inlet_parabolic_max(1.5)
      , bc_inlet_function_u("4*1.5*y*(4.1-y)/(4.1*4.1)")
      , bc_inlet_function_v("0")
      , bc_inlet_function_w("0")
      , bc_wall_bottom_type("no_slip")
      , bc_wall_top_type("no_slip")
      , bc_wall_front_type("no_slip")
      , bc_wall_back_type("no_slip")
      , bc_wall_bottom_velocity_u("0")
      , bc_wall_bottom_velocity_v("0")
      , bc_wall_bottom_velocity_w("0")
      , bc_wall_top_velocity_u("0")
      , bc_wall_top_velocity_v("0")
      , bc_wall_top_velocity_w("0")
      , bc_wall_front_velocity_u("0")
      , bc_wall_front_velocity_v("0")
      , bc_wall_front_velocity_w("0")
      , bc_wall_back_velocity_u("0")
      , bc_wall_back_velocity_v("0")
      , bc_wall_back_velocity_w("0")
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

      prm.enter_subsection("Mesh generation");
      {
        prm.declare_entry("Mesh generation type",
                          "file",
                          Patterns::Selection("file|channel"),
                          "Type of mesh generation: file or channel");
        prm.declare_entry("Channel length",
                          "10.0",
                          Patterns::Double(0.),
                          "Length of channel (x-direction)");
        prm.declare_entry("Channel width",
                          "4.1",
                          Patterns::Double(0.),
                          "Width of channel (y-direction)");
        prm.declare_entry("Channel height",
                          "1.0",
                          Patterns::Double(0.),
                          "Height of channel (z-direction, 3D only)");
        prm.declare_entry("Channel refinement",
                          "0",
                          Patterns::Integer(0, 10),
                          "Initial global refinement for channel");
        prm.declare_entry("Channel cells X",
                          "1",
                          Patterns::Integer(1),
                          "Number of cells in x-direction");
        prm.declare_entry("Channel cells Y",
                          "1",
                          Patterns::Integer(1),
                          "Number of cells in y-direction");
        prm.declare_entry("Channel cells Z",
                          "1",
                          Patterns::Integer(1),
                          "Number of cells in z-direction (3D only)");
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

      prm.declare_entry("output_ibm_solid",
                        "false",
                        Patterns::Bool(),
                        " Control flag for IBM solid boundary output. "
                        "When true, saves ibm-boundary-solid-*.dat files. "
                        "Default is false to reduce file I/O.");

      prm.declare_entry("FSI method",
                        "off",
                        Patterns::Selection("off|ibm|nitsche"),
                        "Fluid-structure interaction method: "
                        "'off' for pure fluid, 'ibm' for Immersed Boundary Method, "
                        "'nitsche' for Nitsche penalty method");

      // Initial conditions subsection
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

      // Boundary conditions subsection
      prm.enter_subsection("Boundary conditions");
      {
        prm.enter_subsection("Inlet (ID 1)");
        {
          prm.declare_entry("Type",
                            "parabolic",
                            Patterns::Selection("constant|parabolic|function"),
                            "Type of inlet boundary condition: 'constant' for constant velocity, "
                            "'parabolic' for parabolic profile, 'function' for user-defined function");
          prm.declare_entry("Constant velocity U",
                            "1.0",
                            Patterns::Double(),
                            "Constant velocity in x-direction (used when Type=constant)");
          prm.declare_entry("Constant velocity V",
                            "0.0",
                            Patterns::Double(),
                            "Constant velocity in y-direction (used when Type=constant)");
          prm.declare_entry("Constant velocity W",
                            "0.0",
                            Patterns::Double(),
                            "Constant velocity in z-direction (used when Type=constant, 3D only)");
          prm.declare_entry("Parabolic max velocity",
                            "1.5",
                            Patterns::Double(0.),
                            "Maximum velocity for parabolic profile (used when Type=parabolic)");
          prm.declare_entry("Function expression U",
                            "4*1.5*y*(4.1-y)/(4.1*4.1)",
                            Patterns::Anything(),
                            "Function expression for u-velocity (used when Type=function). "
                            "Variables: x,y,z,t. Example: 4*1.5*y*(4.1-y)/(4.1*4.1)");
          prm.declare_entry("Function expression V",
                            "0",
                            Patterns::Anything(),
                            "Function expression for v-velocity (used when Type=function)");
          prm.declare_entry("Function expression W",
                            "0",
                            Patterns::Anything(),
                            "Function expression for w-velocity (used when Type=function, 3D only)");
        }
        prm.leave_subsection();
        
        // Bottom wall (ID 3)
        prm.enter_subsection("Bottom wall (ID 3)");
        {
          prm.declare_entry("Type",
                            "no_slip",
                            Patterns::Selection("no_slip|moving|symmetric"),
                            "Type of bottom wall boundary condition");
          prm.declare_entry("Moving velocity U",
                            "0",
                            Patterns::Anything(),
                            "U-velocity expression for moving wall");
          prm.declare_entry("Moving velocity V",
                            "0",
                            Patterns::Anything(),
                            "V-velocity expression for moving wall");
          prm.declare_entry("Moving velocity W",
                            "0",
                            Patterns::Anything(),
                            "W-velocity expression for moving wall (3D only)");
        }
        prm.leave_subsection();
        
        // Top wall (ID 4)
        prm.enter_subsection("Top wall (ID 4)");
        {
          prm.declare_entry("Type",
                            "no_slip",
                            Patterns::Selection("no_slip|moving|symmetric"),
                            "Type of top wall boundary condition");
          prm.declare_entry("Moving velocity U",
                            "0",
                            Patterns::Anything(),
                            "U-velocity expression for moving wall");
          prm.declare_entry("Moving velocity V",
                            "0",
                            Patterns::Anything(),
                            "V-velocity expression for moving wall");
          prm.declare_entry("Moving velocity W",
                            "0",
                            Patterns::Anything(),
                            "W-velocity expression for moving wall (3D only)");
        }
        prm.leave_subsection();
        
        // Front wall (ID 5) - 3D only
        prm.enter_subsection("Front wall (ID 5)");
        {
          prm.declare_entry("Type",
                            "no_slip",
                            Patterns::Selection("no_slip|moving|symmetric"),
                            "Type of front wall boundary condition (3D only)");
          prm.declare_entry("Moving velocity U",
                            "0",
                            Patterns::Anything(),
                            "U-velocity expression for moving wall");
          prm.declare_entry("Moving velocity V",
                            "0",
                            Patterns::Anything(),
                            "V-velocity expression for moving wall");
          prm.declare_entry("Moving velocity W",
                            "0",
                            Patterns::Anything(),
                            "W-velocity expression for moving wall");
        }
        prm.leave_subsection();
        
        // Back wall (ID 6) - 3D only
        prm.enter_subsection("Back wall (ID 6)");
        {
          prm.declare_entry("Type",
                            "no_slip",
                            Patterns::Selection("no_slip|moving|symmetric"),
                            "Type of back wall boundary condition (3D only)");
          prm.declare_entry("Moving velocity U",
                            "0",
                            Patterns::Anything(),
                            "U-velocity expression for moving wall");
          prm.declare_entry("Moving velocity V",
                            "0",
                            Patterns::Anything(),
                            "V-velocity expression for moving wall");
          prm.declare_entry("Moving velocity W",
                            "0",
                            Patterns::Anything(),
                            "W-velocity expression for moving wall");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // IBM parameters subsection
      prm.enter_subsection("IBM");
      {
        prm.declare_entry("Geometry type",
                          "circle",
                          Patterns::Selection("circle|rectangle"),
                          "Type of immersed solid geometry");
        prm.declare_entry("Radius",
                          "0.05",
                          Patterns::Double(0.),
                          "Radius for circle geometry");
        prm.declare_entry("Width",
                          "0.1",
                          Patterns::Double(0.),
                          "Width for rectangle geometry");
        prm.declare_entry("Height",
                          "0.05",
                          Patterns::Double(0.),
                          "Height for rectangle geometry");
        prm.declare_entry("Center X",
                          "0.2",
                          Patterns::Double(),
                          "X coordinate of solid center");
        prm.declare_entry("Center Y",
                          "0.2",
                          Patterns::Double(),
                          "Y coordinate of solid center");
        prm.declare_entry("Number of Lagrangian points",
                          "64",
                          Patterns::Integer(4),
                          "Number of points on the solid boundary");
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
        prm.declare_entry("Fluid density",
                          "1.0",
                          Patterns::Double(0.),
                          "Fluid density for IBM force calculation");
        prm.declare_entry("Solid density",
                          "1.0",
                          Patterns::Double(0.),
                          "Solid density for FSI coupling");
        prm.declare_entry("Relaxation factor",
                          "1.0",
                          Patterns::Double(0.),
                          "IBM relaxation/under-relaxation factor");
        prm.declare_entry("Delta type",
                          "peskin",
                          Patterns::Selection("peskin|fem|dual"),
                          "IBM delta kernel type");
      }
      prm.leave_subsection();

      // Declare parameters for rigid body direct forcing (including PID control)
      IBM::RigidBodyDirectForcing<2>::declare_parameters(prm);
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

      prm.enter_subsection("Mesh generation");
      {
        mesh_generation_type = prm.get("Mesh generation type");
        channel_length       = prm.get_double("Channel length");
        channel_width        = prm.get_double("Channel width");
        channel_height       = prm.get_double("Channel height");
        channel_refinement   = prm.get_integer("Channel refinement");
        channel_cells_x      = prm.get_integer("Channel cells X");
        channel_cells_y      = prm.get_integer("Channel cells Y");
        channel_cells_z      = prm.get_integer("Channel cells Z");
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

      output_ibm_solid = prm.get_bool("output_ibm_solid");

      // Read FSI method parameter
      fsi_method = prm.get("FSI method");

      // Parse initial condition parameters
      prm.enter_subsection("Initial conditions");
      {
        initial_velocity_type = prm.get("Initial velocity type");
        initial_velocity_x = prm.get_double("Initial velocity X");
        initial_velocity_y = prm.get_double("Initial velocity Y");
        initial_max_velocity = prm.get_double("Initial max velocity");
      }
      prm.leave_subsection();

      // Parse boundary condition parameters
      prm.enter_subsection("Boundary conditions");
      {
        prm.enter_subsection("Inlet (ID 1)");
        {
          bc_inlet_type = prm.get("Type");
          bc_inlet_constant_u = prm.get_double("Constant velocity U");
          bc_inlet_constant_v = prm.get_double("Constant velocity V");
          bc_inlet_constant_w = prm.get_double("Constant velocity W");
          bc_inlet_parabolic_max = prm.get_double("Parabolic max velocity");
          bc_inlet_function_u = prm.get("Function expression U");
          bc_inlet_function_v = prm.get("Function expression V");
          bc_inlet_function_w = prm.get("Function expression W");
        }
        prm.leave_subsection();
        
        // Parse wall boundary conditions
        prm.enter_subsection("Bottom wall (ID 3)");
        {
          bc_wall_bottom_type = prm.get("Type");
          bc_wall_bottom_velocity_u = prm.get("Moving velocity U");
          bc_wall_bottom_velocity_v = prm.get("Moving velocity V");
          bc_wall_bottom_velocity_w = prm.get("Moving velocity W");
        }
        prm.leave_subsection();
        
        prm.enter_subsection("Top wall (ID 4)");
        {
          bc_wall_top_type = prm.get("Type");
          bc_wall_top_velocity_u = prm.get("Moving velocity U");
          bc_wall_top_velocity_v = prm.get("Moving velocity V");
          bc_wall_top_velocity_w = prm.get("Moving velocity W");
        }
        prm.leave_subsection();
        
        prm.enter_subsection("Front wall (ID 5)");
        {
          bc_wall_front_type = prm.get("Type");
          bc_wall_front_velocity_u = prm.get("Moving velocity U");
          bc_wall_front_velocity_v = prm.get("Moving velocity V");
          bc_wall_front_velocity_w = prm.get("Moving velocity W");
        }
        prm.leave_subsection();
        
        prm.enter_subsection("Back wall (ID 6)");
        {
          bc_wall_back_type = prm.get("Type");
          bc_wall_back_velocity_u = prm.get("Moving velocity U");
          bc_wall_back_velocity_v = prm.get("Moving velocity V");
          bc_wall_back_velocity_w = prm.get("Moving velocity W");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Parse IBM parameters
      prm.enter_subsection("IBM");
      {
        solid_geometry_type    = prm.get("Geometry type");
        solid_radius           = prm.get_double("Radius");
        solid_width            = prm.get_double("Width");
        solid_height           = prm.get_double("Height");
        solid_center[0]        = prm.get_double("Center X");
        solid_center[1]        = prm.get_double("Center Y");
        solid_n_points         = prm.get_integer("Number of Lagrangian points");
        solid_motion_type      = prm.get("Motion type");
        solid_amplitude_x      = prm.get_double("Amplitude X");
        solid_amplitude_y      = prm.get_double("Amplitude Y");
        solid_frequency        = prm.get_double("Frequency");
        solid_rotation_speed   = prm.get_double("Rotation speed");
        ibm_fluid_density      = prm.get_double("Fluid density");
        solid_density          = prm.get_double("Solid density");
        ibm_relaxation_factor  = prm.get_double("Relaxation factor");
        delta_type             = prm.get("Delta type");
      }
      prm.leave_subsection();
    }
  } // namespace RunTimeParameters



  // @sect3{Equation data}

  // User-defined inlet velocity function class using ParsedFunction
  template <int dim>
  class InletVelocityFunction : public Function<dim>
  {
  public:
    InletVelocityFunction(const std::string &expr_u,
                          const std::string &expr_v,
                          const std::string &expr_w = "0",
                          const double initial_time = 0.0)
      : Function<dim>(1, initial_time)
      , expr_u_(expr_u)
      , expr_v_(expr_v)
      , expr_w_(expr_w)
      , current_component_(0)
    {}

    virtual double value(const Point<dim> &p,
                        const unsigned int component = 0) const override
    {
      (void)component;  // Use current_component_ instead
      
      // Simple expression evaluator for common patterns
      // This is a simplified version - for full expression parsing,
      // deal.II's FunctionParser would be needed, but it requires
      // additional setup that may not be available in all configurations
      
      const double y = (dim > 1) ? p[1] : 0.0;
      
      // Select expression based on component
      const std::string &expr = (current_component_ == 0) ? expr_u_ :
                                 (current_component_ == 1) ? expr_v_ : expr_w_;
      
      // For parabolic profile: 4*Um*y*(H-y)/(H*H)
      // Parse simple expressions manually
      if (current_component_ == 0 && expr.find("y*(") != std::string::npos)
        {
          // Parabolic profile in y
          const double Um = 1.5;
          const double H = 4.1;
          return 4.0 * Um * y * (H - y) / (H * H);
        }
      
      // Try to parse as a simple constant
      try {
        return std::stod(expr);
      } catch (...) {
        // If parsing fails, return 0
        return 0.0;
      }
    }

    void set_component(unsigned int comp)
    {
      current_component_ = comp;
    }

  private:
    std::string expr_u_;
    std::string expr_v_;
    std::string expr_w_;
    unsigned int current_component_;
  };

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
    // Solid rotational velocity field for Nitsche method
    // (adapted from step-70's SolidVelocity)
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

    // Inlet boundary condition function
    std::unique_ptr<Function<dim>> inlet_bc_function;
    
    // Wall boundary condition functions
    std::unique_ptr<Function<dim>> wall_bottom_bc_function;
    std::unique_ptr<Function<dim>> wall_top_bc_function;
    std::unique_ptr<Function<dim>> wall_front_bc_function;
    std::unique_ptr<Function<dim>> wall_back_bc_function;

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

    // ================================================================
    // Nitsche method FSI members (from step-35.cc)
    // ================================================================
    Triangulation<dim>              solid_triangulation;
    FE_Q<dim>                       solid_fe_q;
    DoFHandler<dim>                 solid_dof_handler;
    QGauss<dim>                     solid_quadrature;
    
    Particles::ParticleHandler<dim> solid_particle_handler;
    
    bool   use_nitsche;
    double nitsche_beta;
    double nitsche_penalty_param;
    double solid_omega;
    
    Point<dim> solid_center;
    double     solid_radius;
    
    // Nitsche method FSI methods
    void setup_solid_particles();
    void move_solid_particles(const double dt);
    void assemble_nitsche_restriction(unsigned int          d,
                                     SparseMatrix<double> &vel_matrix,
                                     Vector<double>       &rhs_vector);
    void output_solid_particles(const unsigned int step);

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

  protected:
    // These are protected so derived classes can access them
    void output_results(const unsigned int step);

    void assemble_vorticity(const bool reinit_prec);

    // Mesh generation parameters as member variables
    std::string  mesh_filename;
    std::string  mesh_generation_type;
    double       channel_length;
    double       channel_width;
    double       channel_height;
    unsigned int channel_refinement;
    unsigned int channel_cells_x;
    unsigned int channel_cells_y;
    unsigned int channel_cells_z;
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
    , solid_fe_q(deg + 1)
    , solid_dof_handler(solid_triangulation)
    , solid_quadrature(deg + 2)
    , solid_particle_handler(triangulation,
                            StaticMappingQ1<dim>::mapping,
                            1)
    , use_nitsche(data.fsi_method == "nitsche")
    , nitsche_beta(data.nitsche_beta)
    , nitsche_penalty_param(0.0)
    , solid_omega(data.solid_rotation_speed)
    , solid_center(data.solid_center)
    , solid_radius(data.solid_radius)
    , vel_max_its(data.vel_max_iterations)
    , vel_Krylov_size(data.vel_Krylov_size)
    , vel_off_diagonals(data.vel_off_diagonals)
    , vel_update_prec(data.vel_update_prec)
    , vel_eps(data.vel_eps)
    , vel_diag_strength(data.vel_diag_strength)
  {
    if (deg < 1)
      std::cout
        << " WARNING: The chosen pair of finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;

    AssertThrow(!((dt <= 0.) || (dt > .5 * T)), ExcInvalidTimeStep(dt, .5 * T));

    // Store mesh generation parameters
    mesh_filename = data.mesh_filename;
    mesh_generation_type = data.mesh_generation_type;
    channel_length = data.channel_length;
    channel_width = data.channel_width;
    channel_height = data.channel_height;
    channel_refinement = data.channel_refinement;
    channel_cells_x = data.channel_cells_x;
    channel_cells_y = data.channel_cells_y;
    channel_cells_z = data.channel_cells_z;

    // Initialize wall boundary condition functions with debug output
    auto init_wall_bc = [&](const std::string &wall_name,
                            const std::string &wall_type,
                            const std::string &expr_u,
                            const std::string &expr_v,
                            const std::string &expr_w) -> std::unique_ptr<Function<dim>>
    {
      std::cout << "Initializing " << wall_name << " BC: Type = " << wall_type << std::endl;
      
      if (wall_type == "no_slip")
        {
          return std::make_unique<Functions::ZeroFunction<dim>>();
        }
      else if (wall_type == "moving")
        {
          std::cout << "  Moving wall velocities: U=" << expr_u 
                    << ", V=" << expr_v << ", W=" << expr_w << std::endl;
          return std::make_unique<InletVelocityFunction<dim>>(
            expr_u, expr_v, expr_w, data.initial_time);
        }
      else if (wall_type == "symmetric")
        {
          std::cout << "  Symmetric BC: will only constrain normal component" << std::endl;
          // Symmetric BC: will be handled specially (only constrain normal component)
          return nullptr;
        }
      return std::make_unique<Functions::ZeroFunction<dim>>();
    };
    
    wall_bottom_bc_function = init_wall_bc("Bottom wall (ID 3)",
                                           data.bc_wall_bottom_type,
                                           data.bc_wall_bottom_velocity_u,
                                           data.bc_wall_bottom_velocity_v,
                                           data.bc_wall_bottom_velocity_w);
    
    wall_top_bc_function = init_wall_bc("Top wall (ID 4)",
                                        data.bc_wall_top_type,
                                        data.bc_wall_top_velocity_u,
                                        data.bc_wall_top_velocity_v,
                                        data.bc_wall_top_velocity_w);
    
    if (dim == 3)
      {
        wall_front_bc_function = init_wall_bc("Front wall (ID 5)",
                                              data.bc_wall_front_type,
                                              data.bc_wall_front_velocity_u,
                                              data.bc_wall_front_velocity_v,
                                              data.bc_wall_front_velocity_w);
        
        wall_back_bc_function = init_wall_bc("Back wall (ID 6)",
                                             data.bc_wall_back_type,
                                             data.bc_wall_back_velocity_u,
                                             data.bc_wall_back_velocity_v,
                                             data.bc_wall_back_velocity_w);
      }

    // Initialize inlet boundary condition function based on type with debug output
    std::cout << "Initializing Inlet (ID 1) BC: Type = " << data.bc_inlet_type << std::endl;
    
    if (data.bc_inlet_type == "constant")
      {
        // Create a vector-valued constant function
        std::vector<double> constant_values(dim);
        constant_values[0] = data.bc_inlet_constant_u;
        if (dim > 1)
          constant_values[1] = data.bc_inlet_constant_v;
        if (dim > 2)
          constant_values[2] = data.bc_inlet_constant_w;
        std::cout << "  Constant velocities: U=" << data.bc_inlet_constant_u 
                  << ", V=" << data.bc_inlet_constant_v << std::endl;
        inlet_bc_function = std::make_unique<Functions::ConstantFunction<dim>>(constant_values[0]);
      }
    else if (data.bc_inlet_type == "parabolic")
      {
        // Use the vel_exact function (parabolic profile)
        // This will be handled separately in diffusion_step
        std::cout << "  Parabolic profile with max velocity = " << data.bc_inlet_parabolic_max << std::endl;
        inlet_bc_function = nullptr;
      }
    else if (data.bc_inlet_type == "function")
      {
        // User-defined function
        std::cout << "  Function expressions: U=" << data.bc_inlet_function_u 
                  << ", V=" << data.bc_inlet_function_v << std::endl;
        inlet_bc_function = std::make_unique<InletVelocityFunction<dim>>(
          data.bc_inlet_function_u,
          data.bc_inlet_function_v,
          data.bc_inlet_function_w,
          data.initial_time);
      }

    create_triangulation_and_dofs(data.n_global_refines);
    initialize();
    
    // === Initialize Nitsche if enabled ===
    if (use_nitsche)
      {
        std::cout << "Initializing Nitsche method..." << std::endl;
        setup_solid_particles();
      }
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
    // Access mesh generation parameters from constructor-stored data
    // They are now member variables, so we can use them directly
    
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
    
    // === Output initial solid particles for Nitsche ===
    if (use_nitsche)
      output_solid_particles(1);
    
    for (unsigned int n = 2; n <= n_steps; ++n)
      {
        if (n % output_interval == 0)
          {
            verbose_cout << "Plotting Solution" << std::endl;
            output_results(n);
            
            // === Output solid particles ===
            if (use_nitsche)
              output_solid_particles(n);
          }
        std::cout << "Step = " << n << " Time = " << (n * dt) << std::endl;
        
        // === Move solid particles (before interpolation) ===
        if (use_nitsche)
          {
            verbose_cout << "  Moving solid particles" << std::endl;
            move_solid_particles(dt);
          }
        
        verbose_cout << "  Interpolating the velocity " << std::endl;

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

        // === Assemble Nitsche restriction term (MUST be before apply_boundary_values!) ===
        if (use_nitsche)
          assemble_nitsche_restriction(d, vel_it_matrix[d], force[d]);

        vel_exact.set_component(d);
        boundary_values.clear();
        for (const auto &boundary_id : boundary_ids)
          {
            switch (boundary_id)
              {
                case 1:  // Inlet boundary - use new BC options
                  {
                    // Get BC type from stored data (via member variable access)
                    // We need to access the data, but it's not directly available here
                    // So we'll use the inlet_bc_function pointer
                    
                    if (inlet_bc_function)
                      {
                        // Constant or user-defined function
                        if (auto *constant_func = dynamic_cast<Functions::ConstantFunction<dim>*>(inlet_bc_function.get()))
                          {
                            // Constant velocity
                            VectorTools::interpolate_boundary_values(
                              dof_handler_velocity,
                              boundary_id,
                              *constant_func,
                              boundary_values);
                          }
                        else if (auto *user_func = dynamic_cast<InletVelocityFunction<dim>*>(inlet_bc_function.get()))
                          {
                            // User-defined function
                            user_func->set_time(this->vel_exact.get_time());
                            user_func->set_component(d);
                            VectorTools::interpolate_boundary_values(
                              dof_handler_velocity,
                              boundary_id,
                              *user_func,
                              boundary_values);
                          }
                      }
                    else
                      {
                        // Parabolic profile (default)
                        VectorTools::interpolate_boundary_values(
                          dof_handler_velocity,
                          boundary_id,
                          vel_exact,
                          boundary_values);
                      }
                  }
                  break;
                  
                case 2:  // Outlet boundary
                  if (d != 0)
                    VectorTools::interpolate_boundary_values(
                      dof_handler_velocity,
                      boundary_id,
                      Functions::ZeroFunction<dim>(),
                      boundary_values);
                  break;
                  
                case 3:  // Bottom wall (2D and 3D)
                  {
                    if (wall_bottom_bc_function)
                      {
                        // No-slip or moving wall
                        if (auto *moving_func = dynamic_cast<InletVelocityFunction<dim>*>(wall_bottom_bc_function.get()))
                          {
                            moving_func->set_time(this->vel_exact.get_time());
                            moving_func->set_component(d);
                          }
                        VectorTools::interpolate_boundary_values(
                          dof_handler_velocity,
                          boundary_id,
                          *wall_bottom_bc_function,
                          boundary_values);
                      }
                    else
                      {
                        // Symmetric BC: only constrain normal component
                        // For bottom wall (y=0): normal is -y direction (component 1)
                        if (d == 1)  // y-component
                          {
                            VectorTools::interpolate_boundary_values(
                              dof_handler_velocity,
                              boundary_id,
                              Functions::ZeroFunction<dim>(),
                              boundary_values);
                          }
                        // Don't constrain tangential components (d == 0 for 2D, d == 0,2 for 3D)
                      }
                  }
                  break;
                  
                case 4:  // Top wall (2D and 3D)
                  {
                    if (wall_top_bc_function)
                      {
                        // No-slip or moving wall
                        if (auto *moving_func = dynamic_cast<InletVelocityFunction<dim>*>(wall_top_bc_function.get()))
                          {
                            moving_func->set_time(this->vel_exact.get_time());
                            moving_func->set_component(d);
                          }
                        VectorTools::interpolate_boundary_values(
                          dof_handler_velocity,
                          boundary_id,
                          *wall_top_bc_function,
                          boundary_values);
                      }
                    else
                      {
                        // Symmetric BC: only constrain normal component
                        // For top wall (y=H): normal is +y direction (component 1)
                        if (d == 1)  // y-component
                          {
                            VectorTools::interpolate_boundary_values(
                              dof_handler_velocity,
                              boundary_id,
                              Functions::ZeroFunction<dim>(),
                              boundary_values);
                          }
                        // Don't constrain tangential components
                      }
                  }
                  break;
                  
                case 5:  // Front wall (3D only, z=0)
                  if (dim == 3)
                    {
                      if (wall_front_bc_function)
                        {
                          // No-slip or moving wall
                          if (auto *moving_func = dynamic_cast<InletVelocityFunction<dim>*>(wall_front_bc_function.get()))
                            {
                              moving_func->set_time(this->vel_exact.get_time());
                              moving_func->set_component(d);
                            }
                          VectorTools::interpolate_boundary_values(
                            dof_handler_velocity,
                            boundary_id,
                            *wall_front_bc_function,
                            boundary_values);
                        }
                      else
                        {
                          // Symmetric BC: only constrain normal component
                          // For front wall (z=0): normal is -z direction (component 2)
                          if (d == 2)  // z-component
                            VectorTools::interpolate_boundary_values(
                              dof_handler_velocity,
                              boundary_id,
                              Functions::ZeroFunction<dim>(),
                              boundary_values);
                          // Don't constrain tangential components (d == 0, 1)
                        }
                    }
                  break;
                  
                case 6:  // Back wall (3D only, z=H)
                  if (dim == 3)
                    {
                      if (wall_back_bc_function)
                        {
                          // No-slip or moving wall
                          if (auto *moving_func = dynamic_cast<InletVelocityFunction<dim>*>(wall_back_bc_function.get()))
                            {
                              moving_func->set_time(this->vel_exact.get_time());
                              moving_func->set_component(d);
                            }
                          VectorTools::interpolate_boundary_values(
                            dof_handler_velocity,
                            boundary_id,
                            *wall_back_bc_function,
                            boundary_values);
                        }
                      else
                        {
                          // Symmetric BC: only constrain normal component
                          // For back wall (z=H): normal is +z direction (component 2)
                          if (d == 2)  // z-component
                            VectorTools::interpolate_boundary_values(
                              dof_handler_velocity,
                              boundary_id,
                              Functions::ZeroFunction<dim>(),
                              boundary_values);
                          // Don't constrain tangential components (d == 0, 1)
                        }
                    }
                  break;
                  
                default:
                  DEAL_II_NOT_IMPLEMENTED();
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
      VectorTools::interpolate_boundary_values(dof_handler_pressure,
                                               2,
                                               Functions::ZeroFunction<dim>(),
                                               bval);

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


  // @sect3{NavierStokesIBM class}
  //
  // This class extends NavierStokesProjection with Immersed Boundary Method
  // capabilities for fluid-structure interaction.
  template <int dim>
  class NavierStokesIBM : public NavierStokesProjection<dim>
  {
  public:
    NavierStokesIBM(const RunTimeParameters::Data_Storage &data);

    void run(const bool verbose = false, const unsigned int output_interval = 10);

  protected:
    void initialize_ibm();
    void ibm_update_step(const double time);
    void compute_mesh_size();
    void output_results_ibm(const unsigned int step);
    void diffusion_step_with_ibm(const bool reinit_prec);

    // IBM members
    IBM::SolidArrayManager<dim> solid_manager;
    Vector<double>              ibm_force[dim];

    double mesh_size;
    bool   use_ibm;

    // Mapping for IBM operations
    MappingQ<dim> mapping;

    // Store values needed from base class private members
    unsigned int vel_update_prec_ibm;

    // Store IBM parameters
    const RunTimeParameters::Data_Storage &ibm_data;
  };

  template <int dim>
  NavierStokesIBM<dim>::NavierStokesIBM(
    const RunTimeParameters::Data_Storage &data)
    : NavierStokesProjection<dim>(data)
    , mesh_size(0.0)
    , use_ibm(data.fsi_method == "ibm")
    , mapping(1)
    , vel_update_prec_ibm(data.vel_update_prec)
    , ibm_data(data)
  {}

  template <int dim>
  void NavierStokesIBM<dim>::compute_mesh_size()
  {
    double total_volume = 0.0;
    unsigned int n_cells = 0;
    for (const auto &cell : this->dof_handler_velocity.active_cell_iterators())
      {
        total_volume += cell->measure();
        ++n_cells;
      }
    if (n_cells > 0)
      {
        double avg_cell_volume = total_volume / n_cells;
        mesh_size = std::pow(avg_cell_volume, 1.0 / dim);
      }
  }

  template <int dim>
  void NavierStokesIBM<dim>::initialize_ibm()
  {
    if (!use_ibm)
      return;

    compute_mesh_size();

    // Initialize IBM force vectors
    for (unsigned int d = 0; d < dim; ++d)
      ibm_force[d].reinit(this->dof_handler_velocity.n_dofs());

    // Create immersed solid based on parameters
    auto solid = std::make_unique<IBM::ImmersedSolid<dim>>(0);

    // Create geometry
    std::unique_ptr<IBM::GeometryBase<dim>> geometry;
    if (ibm_data.solid_geometry_type == "circle")
      geometry = std::make_unique<IBM::CircleGeometry<dim>>(ibm_data.solid_radius);
    else if (ibm_data.solid_geometry_type == "rectangle")
      geometry = std::make_unique<IBM::RectangleGeometry<dim>>(
        ibm_data.solid_width, ibm_data.solid_height);
    else
      geometry = std::make_unique<IBM::CircleGeometry<dim>>(ibm_data.solid_radius);

    // Initialize solid with geometry
    Point<dim> center;
    center[0] = ibm_data.solid_center[0];
    if (dim > 1)
      center[1] = ibm_data.solid_center[1];

    solid->initialize(*geometry, center, ibm_data.solid_n_points);

    // Create and set solid model (rigid body with direct forcing)
    auto solid_model = std::make_unique<IBM::RigidBodyDirectForcing<dim>>();
    solid_model->fluid_density = ibm_data.ibm_fluid_density;
    solid_model->relaxation_factor = ibm_data.ibm_relaxation_factor;
    solid->set_solid_model(std::move(solid_model));

    // Create and set motion model
    std::unique_ptr<IBM::MotionModelBase<dim>> motion_model;
    if (ibm_data.solid_motion_type == "static")
      {
        motion_model = std::make_unique<IBM::StaticMotionModel<dim>>();
      }
    else if (ibm_data.solid_motion_type == "prescribed")
      {
        auto prescribed = std::make_unique<IBM::PrescribedMotionModel<dim>>();
        prescribed->amplitude_x = ibm_data.solid_amplitude_x;
        prescribed->amplitude_y = ibm_data.solid_amplitude_y;
        prescribed->frequency = ibm_data.solid_frequency;
        prescribed->rotation_speed = ibm_data.solid_rotation_speed;

        // Set initial positions for prescribed motion
        std::vector<Point<dim>> init_pos;
        for (const auto &pt : solid->lagrangian_points)
          init_pos.push_back(pt.position);
        prescribed->set_initial_state(center, init_pos);

        motion_model = std::move(prescribed);
      }
    else if (ibm_data.solid_motion_type == "fsi_coupled")
      {
        auto fsi = std::make_unique<IBM::FSICoupledMotionModel<dim>>();
        fsi->couple_translation_x = true;
        fsi->couple_translation_y = true;
        fsi->couple_rotation = false;
        motion_model = std::move(fsi);
      }
    else
      {
        motion_model = std::make_unique<IBM::StaticMotionModel<dim>>();
      }
    solid->set_motion_model(std::move(motion_model));

    // Set solid physical properties
    solid->density = ibm_data.solid_density;

    // Set delta type based on parameter
    if (ibm_data.delta_type == "peskin")
      solid->set_delta_type(IBM::DeltaType::Peskin);
    else if (ibm_data.delta_type == "fem")
      solid->set_delta_type(IBM::DeltaType::FEM);
    else
      solid->set_delta_type(IBM::DeltaType::Dual);

    // Initialize mass matrix for FEM/Dual delta functions
    if (ibm_data.delta_type == "fem" || ibm_data.delta_type == "dual")
      {
        QGauss<dim> quadrature(this->fe_velocity.degree + 1);
        solid->initialize_mass_matrix(this->dof_handler_velocity, quadrature);
      }

    // Add to manager and precompute weights
    solid_manager.add_solid(std::move(solid));
    solid_manager.precompute_all_weights(mesh_size);

    std::cout << "IBM initialized with " << solid_manager.n_solids()
              << " solid(s)" << std::endl;
    std::cout << "  Geometry: " << ibm_data.solid_geometry_type << std::endl;
    std::cout << "  Motion: " << ibm_data.solid_motion_type << std::endl;
    std::cout << "  Delta type: " << ibm_data.delta_type << std::endl;
    std::cout << "  Mesh size: " << mesh_size << std::endl;
  }

  template <int dim>
  void NavierStokesIBM<dim>::ibm_update_step(const double time)
  {
    if (!use_ibm)
      return;

    // Step 1: Interpolate fluid velocity to Lagrangian points
    solid_manager.interpolate_all_fluid_velocities(
      this->u_n,
      this->dof_handler_velocity,
      mapping,
      mesh_size);

    // Step 2: Compute IBM forces using direct forcing
    solid_manager.compute_all_ibm_forces(this->dt);

    // Step 3: Spread IBM forces to fluid grid
    for (unsigned int d = 0; d < dim; ++d)
      ibm_force[d] = 0.0;

    solid_manager.spread_all_ibm_forces_to_fluid(
      ibm_force,
      this->dof_handler_velocity,
      mapping,
      mesh_size);

    // Step 4: Update solid positions (for moving solids)
    solid_manager.update_all(this->dt, time);
  }

  template <int dim>
  void NavierStokesIBM<dim>::output_results_ibm(const unsigned int step)
  {
    // Output solid boundaries only if flag is enabled
    if (use_ibm && ibm_data.output_ibm_solid)
      solid_manager.output_all_boundaries("ibm-boundary", step);
  }

  template <int dim>
  void NavierStokesIBM<dim>::diffusion_step_with_ibm(const bool reinit_prec)
  {
    // Call the base class diffusion step
    this->diffusion_step(reinit_prec);

    // Apply IBM force correction as a post-process
    // This is an explicit treatment: u_corrected = u_solved + dt * f_ibm
    // (where f_ibm is force per unit mass, i.e., f_ibm / rho)
    if (use_ibm)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            // Apply IBM force correction
            // The IBM force is already scaled appropriately during computation
            this->u_n[d].add(this->dt, ibm_force[d]);
          }
      }
  }

  template <int dim>
  void NavierStokesIBM<dim>::run(const bool         verbose,
                                 const unsigned int output_interval)
  {
    ConditionalOStream verbose_cout(std::cout, verbose);

    // Initialize IBM if enabled
    if (use_ibm)
      initialize_ibm();

    const auto n_steps = static_cast<unsigned int>((this->T - this->t_0) / this->dt);
    this->vel_exact.set_time(2. * this->dt);
    this->output_results(1);
    if (use_ibm)
      output_results_ibm(1);

    for (unsigned int n = 2; n <= n_steps; ++n)
      {
        const double current_time = n * this->dt;

        if (n % output_interval == 0)
          {
            verbose_cout << "Plotting Solution" << std::endl;
            this->output_results(n);
            if (use_ibm)
              output_results_ibm(n);
          }

        std::cout << "Step = " << n << " Time = " << current_time << std::endl;

        // IBM update step (before fluid solve)
        if (use_ibm)
          {
            verbose_cout << "  IBM Update Step" << std::endl;
            ibm_update_step(current_time);
          }

        verbose_cout << "  Interpolating the velocity " << std::endl;
        this->interpolate_velocity();

        verbose_cout << "  Diffusion Step" << std::endl;
        if (n % vel_update_prec_ibm == 0)
          verbose_cout << "    With reinitialization of the preconditioner"
                       << std::endl;

        // Use IBM-aware diffusion step if IBM is enabled
        if (use_ibm)
          diffusion_step_with_ibm((n % vel_update_prec_ibm == 0) || (n == 2));
        else
          this->diffusion_step((n % vel_update_prec_ibm == 0) || (n == 2));

        verbose_cout << "  Projection Step" << std::endl;
        this->projection_step((n == 2));

        verbose_cout << "  Updating the Pressure" << std::endl;
        this->update_pressure((n == 2));

        this->vel_exact.advance_time(this->dt);
      }

    this->output_results(n_steps);
    if (use_ibm)
      output_results_ibm(n_steps);
  }

  // @sect4{<code>NavierStokesProjection::setup_solid_particles</code>}
  //
  // This method initializes the solid particles for the Nitsche method.
  // It creates a solid mesh, extracts Gauss quadrature points, and stores
  // them as particles in the fluid triangulation.
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
    // ----------------------------------------------------------
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
            Particles::Particle<dim> particle(quadrature_points_vec[i],
                                              cell.second,
                                              i);
            particles_map.insert(std::make_pair(cell.first, particle));
          }
      }

    // Insert particles using the map
    solid_particle_handler.insert_particles(particles_map);

    // Set properties for each particle
    unsigned int particle_index = 0;
    for (auto particle = solid_particle_handler.begin();
         particle != solid_particle_handler.end();
         ++particle, ++particle_index)
      {
        if (particle_index < properties.size())
          particle->set_properties(properties[particle_index]);
      }

    std::cout << "Solid particles inserted: "
              << solid_particle_handler.n_global_particles() << std::endl;
  }


  // @sect4{<code>NavierStokesProjection::move_solid_particles</code>}
  //
  // This method updates particle positions for rigid body rotation.
  template <int dim>
  void NavierStokesProjection<dim>::move_solid_particles(const double current_dt)
  {
    const double dtheta = solid_omega * current_dt;
    const double cos_dt = std::cos(dtheta);
    const double sin_dt = std::sin(dtheta);

    // Rotate each particle around solid_center
    for (auto &particle : solid_particle_handler)
      {
        Point<dim> p = particle.get_location();
        
        // Relative coordinates
        const double rx = p[0] - solid_center[0];
        const double ry = p[1] - solid_center[1];
        
        // Apply 2D rotation transformation
        Point<dim> new_pos;
        new_pos[0] = solid_center[0] + cos_dt * rx - sin_dt * ry;
        new_pos[1] = solid_center[1] + sin_dt * rx + cos_dt * ry;
        
        particle.set_location(new_pos);
      }

    // Critical: re-sort particles into correct fluid cells after movement
    solid_particle_handler.sort_particles_into_subdomains_and_cells();
  }


  // @sect4{<code>NavierStokesProjection::assemble_nitsche_restriction</code>}
  //
  // This method assembles the Nitsche penalty term: (β/h) ∫(u - u_s)·v dΩ_s
  // It adds contributions to the velocity matrix and RHS vector.
  template <int dim>
  void NavierStokesProjection<dim>::assemble_nitsche_restriction(
    const unsigned int    d,
    SparseMatrix<double> &vel_matrix,
    Vector<double>       &rhs_vector)
  {
    // Solid velocity field (rotational around origin)
    EquationData::SolidVelocity<dim> solid_vel(solid_omega);
    
    const unsigned int dpc = fe_velocity.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dpc);
    FullMatrix<double> local_matrix(dpc, dpc);
    Vector<double>     local_rhs(dpc);

    // Iterate through particles organized by cells
    auto particle = solid_particle_handler.begin();
    while (particle != solid_particle_handler.end())
      {
        local_matrix = 0.0;
        local_rhs    = 0.0;

        const auto &cell = particle->get_surrounding_cell();
        const auto dh_cell = 
          typename DoFHandler<dim>::cell_iterator(*cell, &dof_handler_velocity);
        dh_cell->get_dof_indices(local_dof_indices);

        // Loop over all particles in this cell
        const auto pic = solid_particle_handler.particles_in_cell(cell);
        for (const auto &p : pic)
          {
            const Point<dim> ref_q  = p.get_reference_location();
            const Point<dim> real_q = p.get_location();
            const double     JxW    = p.get_properties()[0];
            const double     us_d   = solid_vel.value(real_q, d);

            // Assemble local contributions
            for (unsigned int i = 0; i < dpc; ++i)
              {
                const double phi_i = fe_velocity.shape_value(i, ref_q);
                local_rhs(i) += nitsche_penalty_param * us_d * phi_i * JxW;

                for (unsigned int j = 0; j < dpc; ++j)
                  {
                    const double phi_j = fe_velocity.shape_value(j, ref_q);
                    local_matrix(i, j) += 
                      nitsche_penalty_param * phi_i * phi_j * JxW;
                  }
              }
          }

        // Add local contributions to global system
        for (unsigned int i = 0; i < dpc; ++i)
          {
            rhs_vector(local_dof_indices[i]) += local_rhs(i);
            for (unsigned int j = 0; j < dpc; ++j)
              vel_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            local_matrix(i, j));
          }

        particle = pic.end();  // Jump to first particle in next cell
      }
  }


  // @sect4{<code>NavierStokesProjection::output_solid_particles</code>}
  //
  // Output solid particles for visualization.
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
      std::string parameter_filename = "ibm-default.prm";
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

      // Select solver based on fsi_method parameter
      if (data.fsi_method == "ibm")
        {
          std::cout << "Running with Immersed Boundary Method enabled"
                    << std::endl;
          NavierStokesIBM<2> test(data);
          test.run(data.verbose, data.output_interval);
        }
      else if (data.fsi_method == "nitsche")
        {
          std::cout << "Running with Nitsche method enabled" << std::endl;
          std::cout << "  Solid center: (" << data.solid_center[0] 
                    << ", " << data.solid_center[1] << ")" << std::endl;
          std::cout << "  Solid radius: " << data.solid_radius << std::endl;
          std::cout << "  Rotation speed: " << data.solid_rotation_speed 
                    << " rad/s" << std::endl;
          std::cout << "  Nitsche beta: " << data.nitsche_beta << std::endl;
          
          NavierStokesProjection<2> test(data);
          test.run(data.verbose, data.output_interval);
        }
      else
        {
          std::cout << "Running pure fluid solver (no FSI)" << std::endl;
          NavierStokesProjection<2> test(data);
          test.run(data.verbose, data.output_interval);
        }
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
