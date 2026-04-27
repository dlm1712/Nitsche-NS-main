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
 * Modified: Integrated FSI Configuration Manager
 */

// @sect3{Include files}

// We start by including all the necessary deal.II header files and some C++
// related ones.
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
#include <deal.II/lac/full_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/parsed_function.h>
#include <deal.II/base/function_parser.h>

// FSI: Particle handling for Nitsche method (from step-70)
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/generators.h>
#include <deal.II/particles/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <algorithm>

namespace Step35
{
  using namespace dealii;

  // ========================================================================
  // @sect3{Unified Configuration Structure}
  // ========================================================================
  
  /**
   * A flat configuration structure to hold all FSI/IBM parameters.
   * This avoids complex subsection nesting in ParameterHandler.
   */
  struct FSIConfig
  {
    // --- Geometry ---
    std::string geometry_type = "circle"; // circle, rectangle, ellipse, polygon, from_file
    double      radius        = 0.05;
    double      width         = 0.1;
    double      height        = 0.05;
    Point<3>    center;
    unsigned int n_lagrangian_points = 64;
    std::string solid_file     = "";

    // --- Motion ---
    std::string motion_type   = "static"; // static, prescribed, fsi_coupled
    double      amplitude_x   = 0.0;
    double      amplitude_y   = 0.0;
    double      amplitude_z   = 0.0;
    double      frequency     = 1.0;
    double      rotation_speed = 0.0;

    // --- Physics & IBM Model ---
    std::string solid_model_type = "rigid"; // rigid, elastic
    std::string delta_type       = "peskin"; // peskin, fem, dual
    double      fluid_density    = 1.0;
    double      solid_density    = 1.0;
    double      relaxation_factor = 1.0;
    std::string weight_method    = "grid_size"; // grid_size, delta_support, lagrangian_spacing, integrated

    // --- PID Control ---
    bool   use_pid         = false;
    double kp              = 1.0;
    double ki              = 0.0;
    double kd              = 0.0;
    double integral_limit  = 1e10;
    double tau_integral    = 1e10;

    // --- Nitsche (Legacy/Optional) ---
    bool   use_nitsche     = false;
    double nitsche_beta    = 10.0;
    double nitsche_gamma   = 1.0;

    FSIConfig()
    {
      center = Point<3>(0.0, 0.0, 0.0);
    }
  };

  // ========================================================================
  // @sect3{Data Structures for IBM}
  // ========================================================================

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

  enum class MotionType
  {
    Static,     // Fixed in place
    Prescribed, // Prescribed motion function
    FSICoupled  // Fully coupled FSI
  };

  template <int dim>
  struct LagrangianPoint
  {
    Point<dim>       position;
    Point<dim>       reference_position;
    Tensor<1, dim>   velocity;
    Tensor<1, dim>   acceleration;
    Tensor<1, dim>   fluid_velocity;
    Tensor<1, dim>   ibm_force;
    Tensor<1, dim>   internal_force;
    Tensor<1, dim>   external_force;
    
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

  // ========================================================================
  // @sect3{Geometry Classes}
  // ========================================================================

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
                    const double       scale = 1.0) const override
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

    GeometryType get_type() const override
    {
      return GeometryType::Circle;
    }

    double radius;
  };

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
                    const double       scale = 1.0) const override
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

    GeometryType get_type() const override
    {
      return GeometryType::Rectangle;
    }

    double width;
    double height;
  };

  template <int dim>
  class FileGeometry : public GeometryBase<dim>
  {
  public:
    FileGeometry(const std::string &filename)
      : filename(filename)
    {}

    std::vector<LagrangianPoint<dim>>
    generate_points(const unsigned int /*n_points*/,
                    const Point<dim>  &/*center*/,
                    const double       scale = 1.0) const override
    {
      std::vector<LagrangianPoint<dim>> points;
      std::ifstream infile(filename);
      
      if (!infile.is_open())
        {
          AssertThrow(false, ExcMessage("Could not open solid geometry file: " + filename));
        }

      std::string line;
      unsigned int id = 0;
      
      while (std::getline(infile, line))
        {
          if (line.empty() || line[0] == '#')
            continue;

          std::istringstream iss(line);
          LagrangianPoint<dim> pt;
          
          for (unsigned int d = 0; d < dim; ++d)
            {
              if (!(iss >> pt.position[d]))
                {
                  AssertThrow(false, ExcMessage("Invalid format in solid geometry file at point " + std::to_string(id)));
                }
              pt.position[d] *= scale;
            }
            
          pt.reference_position = pt.position;
          pt.arc_length         = 0.0; 
          pt.id                 = id++;
          pt.mass               = 0.0;
          
          points.push_back(pt);
        }
        
      infile.close();
      
      if (points.empty())
        {
           AssertThrow(false, ExcMessage("No points read from solid geometry file: " + filename));
        }

      // Optional: Compute arc_length if needed
      const unsigned int n_pts = points.size();
      std::vector<double> min_dist(n_pts, 1e10);
      
      // Simple nearest neighbor distance calculation for arc_length approximation
      for (unsigned int i = 0; i < n_pts; ++i)
        {
          for (unsigned int j = 0; j < n_pts; ++j)
            {
              if (i != j)
                {
                  double dist = points[i].position.distance(points[j].position);
                  if (dist < min_dist[i])
                    min_dist[i] = dist;
                }
            }
        }

      for (unsigned int i = 0; i < n_pts; ++i) {
        if (min_dist[i] > 1e-14) {
            if (dim == 2) {
                points[i].arc_length = min_dist[i]; 
            } else if (dim == 3) {
                points[i].arc_length = min_dist[i] * min_dist[i]; 
            }
        } else {
            points[i].arc_length = 0.0;
        }
      }

      return points;
    }

    GeometryType get_type() const override
    {
      return GeometryType::FromFile;
    }

  private:
    std::string filename;
  };

  // ========================================================================
  // @sect3{Motion Models}
  // ========================================================================

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
                               const Tensor<1, dim>              &fluid_torque,
                               const double                       dt,
                               const double                       time) = 0;

    virtual MotionType get_motion_type() const = 0;
    
    virtual void set_initial_state(const Point<dim>              &center,
                                   const std::vector<Point<dim>> &positions) = 0;
  };

  template <int dim>
  class StaticMotionModel : public MotionModelBase<dim>
  {
  public:
    void update_motion(std::vector<LagrangianPoint<dim>> & /*points*/,
                       Point<dim> & /*center*/,
                       Tensor<1, dim> & /*vel*/,
                       Tensor<1, dim> & /*angular_vel*/,
                       const Tensor<1, dim> & /*fluid_force*/,
                       const Tensor<1, dim> &/*fluid_torque*/,
                       const double /*dt*/,
                       const double /*time*/) override
    {
      // Static: no position update
    }

    MotionType get_motion_type() const override
    {
      return MotionType::Static;
    }
    
    void set_initial_state(const Point<dim> &, const std::vector<Point<dim>> &) override {}
  };

  template <int dim>
  class PrescribedMotionModel : public MotionModelBase<dim>
  {
  public:
    PrescribedMotionModel(const FSIConfig &config)
      : amplitude_x(config.amplitude_x)
      , amplitude_y(config.amplitude_y)
      , amplitude_z(config.amplitude_z)
      , frequency(config.frequency)
      , rotation_speed(config.rotation_speed)
    {}

    void update_motion(
      std::vector<LagrangianPoint<dim>> &points,
      Point<dim>                        &center,
      Tensor<1, dim>                    &vel,
      Tensor<1, dim>                    & /*angular_vel*/,
      const Tensor<1, dim>              & /*fluid_force*/,
      const Tensor<1, dim>              & /*fluid_torque*/,
      const double /*dt*/,
      const double time) override
    {
      const double pi        = numbers::PI;

      // Compute new center position
      Point<dim> new_center = initial_center;
      new_center[0] += amplitude_x * std::sin(2.0 * pi * frequency * time);
      if (dim > 1)
        new_center[1] += amplitude_y * std::cos(2.0 * pi * frequency * time);
      if (dim > 2)
        new_center[2] += amplitude_z * std::sin(2.0 * pi * frequency * time);

      center = new_center;

      // Update velocity
      vel[0] = amplitude_x * 2.0 * pi * frequency *
               std::cos(2.0 * pi * frequency * time);
      if (dim > 1)
        vel[1] = -amplitude_y * 2.0 * pi * frequency *
                 std::sin(2.0 * pi * frequency * time);
      if (dim > 2)
        vel[2] = amplitude_z * 2.0 * pi * frequency *
                 std::cos(2.0 * pi * frequency * time);

      // Update all Lagrangian points
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          Point<dim> rel;
          for (unsigned int d = 0; d < dim; ++d)
            rel[d] = initial_positions[i][d] - initial_center[d];
            
          // Simple translation for prescribed motion (rotation omitted for simplicity in this snippet)
          for (unsigned int d = 0; d < dim; ++d)
            points[i].position[d] = center[d] + rel[d];
            
          points[i].velocity = vel;
        }
    }

    MotionType get_motion_type() const override
    {
      return MotionType::Prescribed;
    }

    void set_initial_state(const Point<dim>              &center,
                           const std::vector<Point<dim>> &positions) override
    {
      initial_center    = center;
      initial_positions = positions;
    }

  private:
    double amplitude_x;
    double amplitude_y;
    double amplitude_z;
    double frequency;
    double rotation_speed;

    Point<dim>              initial_center;
    std::vector<Point<dim>> initial_positions;
  };

  template <int dim>
  class FSICoupledMotionModel : public MotionModelBase<dim>
  {
  public:
    FSICoupledMotionModel(const FSIConfig &config)
      : mass(config.solid_density) // Simplified mass assignment
      , moment_of_inertia(1.0)
      , couple_translation_x(true)
      , couple_translation_y(true)
      , couple_translation_z(dim == 3)
      , couple_rotation(false)
      , orientation({1.0, 0.0, 0.0, 0.0})
      , external_force(0.0)
      , external_torque_scalar(0.0)
    {}

    void update_motion(
      std::vector<LagrangianPoint<dim>> &points,
      Point<dim>                        &center,
      Tensor<1, dim>                    &vel,
      Tensor<1, dim>                    &angular_vel,
      const Tensor<1, dim>              &fluid_force,
      const Tensor<1, dim>              &fluid_torque,
      const double                       dt,
      const double /*time*/) override
    {
      // 1. Update Translation
      Tensor<1, dim> acceleration;
      for (unsigned int d = 0; d < dim; ++d)
        acceleration[d] = (fluid_force[d] + external_force[d]) / mass;

      if (couple_translation_x)
        vel[0] += acceleration[0] * dt;
      if (dim > 1 && couple_translation_y)
        vel[1] += acceleration[1] * dt;
      if (dim > 2 && couple_translation_z) 
        vel[2] += acceleration[2] * dt;
        
      center += vel * dt;

      // 2. Update Rotation (Simplified 2D for brevity, 3D logic exists in original)
      if (couple_rotation && dim == 2)
        {
          double total_torque = fluid_torque[0] + external_torque_scalar;
          double angular_accel = total_torque / moment_of_inertia;
          angular_vel[0] += angular_accel * dt;
          orientation[0] += angular_vel[0] * dt; // Store angle in orientation[0] for 2D
        }

      // 3. Update Points
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          Point<dim> r_local;
          for (unsigned int d = 0; d < dim; ++d)
            r_local[d] = initial_positions[i][d];

          Point<dim> rotated_r = r_local; // Identity rotation for now
          
          // Apply rotation if needed (2D example)
          if (dim == 2 && couple_rotation)
            {
              double angle = orientation[0];
              double cos_a = std::cos(angle);
              double sin_a = std::sin(angle);
              rotated_r[0] = r_local[0] * cos_a - r_local[1] * sin_a;
              rotated_r[1] = r_local[0] * sin_a + r_local[1] * cos_a;
            }

          for (unsigned int d = 0; d < dim; ++d)
            points[i].position[d] = center[d] + rotated_r[d];

          // Update Velocity: v = v_cm + omega x r
          Tensor<1, dim> point_vel = vel;
          if (dim == 2 && couple_rotation)
            {
              double omega = angular_vel[0];
              point_vel[0] += -omega * rotated_r[1];
              point_vel[1] +=  omega * rotated_r[0];
            }
          points[i].velocity = point_vel;
        }
    }

    MotionType get_motion_type() const override
    {
      return MotionType::FSICoupled;
    }

    void set_initial_state(const Point<dim>              &center,
                           const std::vector<Point<dim>> &positions) override
    {
      initial_center = center;
      initial_positions.resize(positions.size());
      for (unsigned int i = 0; i < positions.size(); ++i)
        {
          for (unsigned int d = 0; d < dim; ++d)
            initial_positions[i][d] = positions[i][d] - center[d];
        }
    }

  private:
    double mass;
    double moment_of_inertia;
    bool couple_translation_x;
    bool couple_translation_y;
    bool couple_translation_z;
    bool couple_rotation;
    std::array<double, 4> orientation; // w, x, y, z
    Tensor<1, dim> external_force;
    double external_torque_scalar;
    
    Point<dim>              initial_center;
    std::vector<Point<dim>> initial_positions;
  };

  // ========================================================================
  // @sect3{Solid Models (Force Calculation)}
  // ========================================================================

  template <int dim>
  class SolidModelBase
  {
  public:
    virtual ~SolidModelBase() = default;

    virtual void compute_ibm_forces(std::vector<LagrangianPoint<dim>> &points,
                                    const double                       dt) = 0;

    virtual void compute_internal_forces(std::vector<LagrangianPoint<dim>> &points,
                                         const double                       time) = 0;

    virtual std::string get_model_type() const = 0;
  };

  template <int dim>
  class RigidBodyDirectForcing : public SolidModelBase<dim>
  {
  public:
    enum class WeightMethod
    {
      GridSize,
      DeltaSupport,
      LagrangianSpacing,
      Integrated
    };

    RigidBodyDirectForcing(const FSIConfig &config)
      : fluid_density(config.fluid_density)
      , relaxation_factor(config.relaxation_factor)
      , use_pid(config.use_pid)
      , Kp(config.kp)
      , Ki(config.ki)
      , Kd(config.kd)
      , integral_limit(config.integral_limit)
      , tau_integral(config.tau_integral)
    {
      if (config.weight_method == "grid_size")
        weight_method = WeightMethod::GridSize;
      else if (config.weight_method == "delta_support")
        weight_method = WeightMethod::DeltaSupport;
      else if (config.weight_method == "lagrangian_spacing")
        weight_method = WeightMethod::LagrangianSpacing;
      else
        weight_method = WeightMethod::Integrated;
    }

    void compute_ibm_forces(std::vector<LagrangianPoint<dim>> &points,
                            const double                       dt) override
    {
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          const Tensor<1, dim> velocity_error = points[i].velocity - points[i].fluid_velocity;

          if (use_pid)
            {
              // PID Control Logic
              const double alpha = std::exp(-dt / tau_integral);
              
              // Update integral
              points[i].velocity_error_integral =
                alpha * points[i].velocity_error_integral + velocity_error * dt;

              // Anti-windup
              for (unsigned int d = 0; d < dim; ++d)
                {
                  if (points[i].velocity_error_integral[d] > integral_limit)
                    points[i].velocity_error_integral[d] = integral_limit;
                  else if (points[i].velocity_error_integral[d] < -integral_limit)
                    points[i].velocity_error_integral[d] = -integral_limit;
                }

              const Tensor<1, dim> I_term = Ki * points[i].velocity_error_integral;
              const Tensor<1, dim> error_rate =
                (velocity_error - points[i].velocity_error_previous) / dt;
              const Tensor<1, dim> D_term = Kd * error_rate;
              const Tensor<1, dim> P_term = Kp * velocity_error;

              points[i].ibm_force =
                point_weights[i] * (P_term + I_term + D_term) / dt;
                
              points[i].velocity_error_previous = velocity_error;
            }
          else
            {
              // Original direct forcing (P-only)
              points[i].ibm_force = point_weights[i] * velocity_error / dt;
            }
        }
    }

    void compute_internal_forces(std::vector<LagrangianPoint<dim>> &points,
                                 const double /*time*/) override
    {
      for (auto &point : points)
        point.internal_force = Tensor<1, dim>();
    }

    std::string get_model_type() const override
    {
      return "rigid_direct_forcing";
    }

    void precompute_weights(const std::vector<LagrangianPoint<dim>> &points,
                            const double                             h)
    {
      point_weights.resize(points.size());
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          double w = 0.0;
          switch (weight_method)
            {
              case WeightMethod::GridSize:
                w = fluid_density * std::pow(h, dim);
                break;
              case WeightMethod::DeltaSupport:
                w = fluid_density * std::pow(2.0 * h, dim);
                break;
              case WeightMethod::LagrangianSpacing:
                {
                  double ds = points[i].arc_length;
                  if (ds < 1e-12) ds = h;
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

  private:
    double       fluid_density;
    double       relaxation_factor;
    WeightMethod weight_method;
    bool         use_pid;
    double       Kp, Ki, Kd;
    double       integral_limit;
    double       tau_integral;
    std::vector<double> point_weights;
  };

  // ========================================================================
  // @sect3{Immersed Solid Class}
  // ========================================================================

  template <int dim>
  class ImmersedSolid
  {
  public:
    ImmersedSolid(const unsigned int solid_id = 0)
      : id(solid_id)
      , mass(1.0)
      , density(1.0)
    {}

    void initialize(const GeometryBase<dim> &geometry,
                    const Point<dim>        &center,
                    const unsigned int       n_points,
                    const double             scale = 1.0)
    {
      lagrangian_points = geometry.generate_points(n_points, center, scale);
      center_of_mass    = center;
      initial_center    = center;

      initial_positions.resize(lagrangian_points.size());
      for (unsigned int i = 0; i < lagrangian_points.size(); ++i)
        initial_positions[i] = lagrangian_points[i].position;
    }

    void set_solid_model(std::unique_ptr<SolidModelBase<dim>> model)
    {
      solid_model = std::move(model);
    }

    void set_motion_model(std::unique_ptr<MotionModelBase<dim>> model)
    {
      motion_model = std::move(model);
    }

    void update(const Tensor<1, dim> &fluid_force,
                const double          fluid_torque_scalar,
                const double          dt,
                const double          time)
    {
      if (motion_model)
        {
          Tensor<1, dim> fluid_torque;
          if (dim == 2)
            fluid_torque[0] = fluid_torque_scalar;
            
          motion_model->update_motion(lagrangian_points,
                                      center_of_mass,
                                      velocity,
                                      angular_velocity,
                                      fluid_force,
                                      fluid_torque,
                                      dt,
                                      time);
        }
    }

    void compute_ibm_forces(const double dt)
    {
      if (solid_model)
        {
          solid_model->compute_internal_forces(lagrangian_points, 0.0);
          solid_model->compute_ibm_forces(lagrangian_points, dt);
        }
    }

    void precompute_weights(const double h)
    {
      if (auto *rigid = dynamic_cast<RigidBodyDirectForcing<dim> *>(solid_model.get()))
        {
          rigid->precompute_weights(lagrangian_points, h);
        }
    }

    Tensor<1, dim> compute_total_force() const
    {
      Tensor<1, dim> total_force;
      for (const auto &point : lagrangian_points)
        total_force += point.ibm_force;
      return total_force;
    }

    Tensor<1, dim> compute_total_torque() const
    {
      Tensor<1, dim> total_torque;
      if (dim == 2)
        {
          double total_torque_z = 0.0;
          for (const auto &point : lagrangian_points)
            {
              Point<dim> r;
              for (unsigned int d = 0; d < dim; ++d)
                r[d] = point.position[d] - center_of_mass[d];
              total_torque_z += r[0] * point.ibm_force[1] - r[1] * point.ibm_force[0];
            }
          total_torque[0] = total_torque_z;
        }
      // 3D torque calculation omitted for brevity
      return total_torque;
    }

    // Accessors for main solver
    const std::vector<LagrangianPoint<dim>> &get_points() const { return lagrangian_points; }
    std::vector<LagrangianPoint<dim>> &get_points() { return lagrangian_points; }
    Point<dim> get_center() const { return center_of_mass; }

  private:
    unsigned int                       id;
    std::vector<LagrangianPoint<dim>>  lagrangian_points;
    std::vector<Point<dim>>            initial_positions;

    Point<dim>     center_of_mass;
    Point<dim>     initial_center;
    Tensor<1, dim> velocity;
    Tensor<1, dim> angular_velocity;

    double mass;
    double density;

    std::unique_ptr<SolidModelBase<dim>>  solid_model;
    std::unique_ptr<MotionModelBase<dim>> motion_model;
  };

  // ========================================================================
  // @sect3{Main Solver Class (Skeleton)}
  // ========================================================================

  template <int dim>
  class NavierStokesIBM
  {
  public:
    NavierStokesIBM()
    {
      // Default initialization if needed
    }

    static void declare_parameters(ParameterHandler &prm)
    {
      // --- Global Parameters ---
      prm.declare_entry("Dimension", "2", Patterns::Integer(2, 3), "Spatial dimension");
      prm.declare_entry("FSI method", "ibm", Patterns::Selection("off|ibm|nitsche"), "FSI Method");

      // --- IBM/FSI Parameters (Flat Structure) ---
      
      // Geometry
      prm.declare_entry("Geometry type", "circle", 
                        Patterns::Selection("circle|rectangle|ellipse|polygon|from_file"),
                        "Type of solid geometry");
      prm.declare_entry("Radius", "0.05", Patterns::Double(0), "Radius for circle/ellipse");
      prm.declare_entry("Width", "0.1", Patterns::Double(0), "Width for rectangle");
      prm.declare_entry("Height", "0.05", Patterns::Double(0), "Height for rectangle");
      prm.declare_entry("Center X", "0.2", Patterns::Double(), "Center X coordinate");
      prm.declare_entry("Center Y", "0.2", Patterns::Double(), "Center Y coordinate");
      prm.declare_entry("Center Z", "0.2", Patterns::Double(), "Center Z coordinate");
      prm.declare_entry("Number of Lagrangian points", "64", Patterns::Integer(1), "Number of Lagrangian markers");
      prm.declare_entry("Solid Point file", "", Patterns::Anything(), "File containing point coordinates if geometry=from_file");

      // Motion
      prm.declare_entry("Motion type", "static", 
                        Patterns::Selection("static|prescribed|fsi_coupled"),
                        "Type of motion");
      prm.declare_entry("Amplitude X", "0.0", Patterns::Double(), "Prescribed motion amplitude X");
      prm.declare_entry("Amplitude Y", "0.0", Patterns::Double(), "Prescribed motion amplitude Y");
      prm.declare_entry("Amplitude Z", "0.0", Patterns::Double(), "Prescribed motion amplitude Z");
      prm.declare_entry("Frequency", "1.0", Patterns::Double(0), "Prescribed motion frequency");
      prm.declare_entry("Rotation speed", "0.0", Patterns::Double(), "Prescribed rotation speed");

      // Physics/IBM
      prm.declare_entry("Fluid density", "1.0", Patterns::Double(0), "Fluid density");
      prm.declare_entry("Solid density", "1.0", Patterns::Double(0), "Solid density");
      prm.declare_entry("Relaxation factor", "1.0", Patterns::Double(0), "Relaxation factor for IBM forcing");
      prm.declare_entry("Delta type", "peskin", 
                        Patterns::Selection("peskin|fem|dual"),
                        "Type of Delta function");
      prm.declare_entry("Weight method", "grid_size", 
                        Patterns::Selection("grid_size|delta_support|lagrangian_spacing|integrated"),
                        "Method to compute IBM weight");

      // PID
      prm.declare_entry("Use PID control", "false", Patterns::Bool(), "Enable PID controller");
      prm.declare_entry("Kp", "1.0", Patterns::Double(0), "Proportional gain");
      prm.declare_entry("Ki", "0.0", Patterns::Double(0), "Integral gain");
      prm.declare_entry("Kd", "0.0", Patterns::Double(0), "Derivative gain");
      prm.declare_entry("Integral limit", "1e10", Patterns::Double(0), "Integral anti-windup limit");
      prm.declare_entry("Integral time constant", "1e10", Patterns::Double(0), "Integral decay time constant");
      
      // Nitsche
      prm.declare_entry("Use Nitsche", "false", Patterns::Bool(), "Use Nitsche method instead of IBM");
      prm.declare_entry("Nitsche Beta", "10.0", Patterns::Double(0), "Nitsche penalty parameter");
      prm.declare_entry("Nitsche Gamma", "1.0", Patterns::Double(0), "Nitsche stability parameter");
    }

    void parse_parameters(ParameterHandler &prm)
    {
      // 1. Read all parameters into the flat config struct
      config.geometry_type = prm.get("Geometry type");
      config.radius = prm.get_double("Radius");
      config.width = prm.get_double("Width");
      config.height = prm.get_double("Height");
      config.center[0] = prm.get_double("Center X");
      config.center[1] = prm.get_double("Center Y");
      config.center[2] = prm.get_double("Center Z");
      config.n_lagrangian_points = prm.get_integer("Number of Lagrangian points");
      config.solid_file = prm.get("Solid Point file");

      config.motion_type = prm.get("Motion type");
      config.amplitude_x = prm.get_double("Amplitude X");
      config.amplitude_y = prm.get_double("Amplitude Y");
      config.amplitude_z = prm.get_double("Amplitude Z");
      config.frequency = prm.get_double("Frequency");
      config.rotation_speed = prm.get_double("Rotation speed");

      config.fluid_density = prm.get_double("Fluid density");
      config.solid_density = prm.get_double("Solid density");
      config.relaxation_factor = prm.get_double("Relaxation factor");
      config.delta_type = prm.get("Delta type");
      config.weight_method = prm.get("Weight method");

      config.use_pid = prm.get_bool("Use PID control");
      config.kp = prm.get_double("Kp");
      config.ki = prm.get_double("Ki");
      config.kd = prm.get_double("Kd");
      config.integral_limit = prm.get_double("Integral limit");
      config.tau_integral = prm.get_double("Integral time constant");
      
      config.use_nitsche = prm.get_bool("Use Nitsche");
      config.nitsche_beta = prm.get_double("Nitsche Beta");
      config.nitsche_gamma = prm.get_double("Nitsche Gamma");

      // 2. Initialize Objects based on Config
      setup_ibm_objects();
    }

    void run()
    {
      // Main simulation loop would go here
      std::cout << "Running IBM Simulation with Config:" << std::endl;
      std::cout << "  Geometry: " << config.geometry_type << std::endl;
      std::cout << "  Motion: " << config.motion_type << std::endl;
      std::cout << "  PID Enabled: " << config.use_pid << std::endl;
    }

  private:
    FSIConfig config;
    std::unique_ptr<ImmersedSolid<dim>> immersed_solid;

    void setup_ibm_objects()
    {
      immersed_solid = std::make_unique<ImmersedSolid<dim>>(0);

      // 1. Setup Geometry
      std::unique_ptr<GeometryBase<dim>> geometry;
      if (config.geometry_type == "circle")
        geometry = std::make_unique<CircleGeometry<dim>>(config.radius);
      else if (config.geometry_type == "rectangle")
        geometry = std::make_unique<RectangleGeometry<dim>>(config.width, config.height);
      else if (config.geometry_type == "from_file")
        geometry = std::make_unique<FileGeometry<dim>>(config.solid_file);
      else
        AssertThrow(false, ExcMessage("Unknown geometry type"));

      immersed_solid->initialize(*geometry, config.center, config.n_lagrangian_points);

      // 2. Setup Motion Model
      std::unique_ptr<MotionModelBase<dim>> motion_model;
      if (config.motion_type == "static")
        motion_model = std::make_unique<StaticMotionModel<dim>>();
      else if (config.motion_type == "prescribed")
        motion_model = std::make_unique<PrescribedMotionModel<dim>>(config);
      else if (config.motion_type == "fsi_coupled")
        motion_model = std::make_unique<FSICoupledMotionModel<dim>>(config);
      
      immersed_solid->set_motion_model(std::move(motion_model));

      // 3. Setup Solid Model (Force Calculation)
      std::unique_ptr<SolidModelBase<dim>> solid_model;
      if (config.solid_model_type == "rigid")
        solid_model = std::make_unique<RigidBodyDirectForcing<dim>>(config);
      
      immersed_solid->set_solid_model(std::move(solid_model));
      
      // Precompute weights if needed (requires mesh size h, usually passed later)
      // immersed_solid->precompute_weights(h);
    }
  };

} // namespace Step35

int main()
{
  using namespace dealii;
  using namespace Step35;

  try
    {
      LogStream::declare_all_exceptions_as_exceptions();
      
      ParameterHandler prm;
      NavierStokesIBM<2>::declare_parameters(prm);
      
      // Parse the parameter file
      prm.parse_input("parameter-file.prm");
      
      NavierStokesIBM<2> solver;
      solver.parse_parameters(prm);
      solver.run();
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

  return 0;
}