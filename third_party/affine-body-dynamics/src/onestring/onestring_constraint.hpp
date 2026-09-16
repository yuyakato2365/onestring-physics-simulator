#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ipc::rigid {

struct OneStringMetrics {
    double string_length = 0.0;
    double command_length = 0.0;
    double constraint_violation = 0.0;
    bool active = false;
};

/// Unilateral total-path-length constraint for the OneString backend.
///
/// The constraint is
///     L(q) <= L_command(t)
/// and is enforced with a one-sided quadratic penalty.  When the string is
/// slack (L <= L_command), the energy, gradient, and Hessian are exactly zero.
class OneStringConstraint {
public:
    bool load(
        const std::string& filename,
        std::size_t num_bodies,
        std::string& error_message);

    bool enabled() const { return m_enabled; }

    double compute_energy(
        const Eigen::VectorXd& x,
        int dim,
        double time,
        double timestep_energy_scale,
        Eigen::VectorXd& grad,
        Eigen::SparseMatrix<double>& hess,
        bool compute_grad,
        bool compute_hess) const;

    OneStringMetrics metrics(
        const Eigen::VectorXd& x,
        int dim,
        double time) const;

private:
    enum class PointType { WORLD_ANCHOR, BODY_GUIDE };

    struct PathPoint {
        PointType type = PointType::WORLD_ANCHOR;
        std::string id;
        int body_id = -1;
        Eigen::Vector3d value = Eigen::Vector3d::Zero();
    };

    struct EvaluatedPoint {
        Eigen::Vector3d world = Eigen::Vector3d::Zero();
        int body_id = -1;
        Eigen::Matrix<double, 3, 12> jacobian =
            Eigen::Matrix<double, 3, 12>::Zero();
    };

    struct GeometryEvaluation {
        double length = 0.0;
        Eigen::VectorXd gradient;
        Eigen::SparseMatrix<double> hessian;
    };

    EvaluatedPoint evaluate_point(
        const PathPoint& point,
        const Eigen::VectorXd& x,
        int dim,
        double time) const;

    GeometryEvaluation evaluate_geometry(
        const Eigen::VectorXd& x,
        int dim,
        double time,
        bool compute_hessian) const;

    double command_length(double time) const;
    Eigen::Vector3d anchor_shake_offset(
        const PathPoint& point,
        double time) const;

    bool m_enabled = false;
    std::size_t m_num_bodies = 0;
    std::vector<PathPoint> m_path_points;
    std::vector<std::pair<double, double>> m_pull_schedule;

    double m_stiffness = 1.0e6;
    double m_smoothing_epsilon = 1.0e-9;
    bool m_use_exact_hessian = false;

    // A fixed horizontal desk represented by a one-sided normal penalty at
    // material points on the bottom face of each panel. This avoids adding a
    // static collision body, which is not robust in the upstream ABD fork.
    bool m_desk_enabled = false;
    double m_desk_height = 0.0;
    double m_desk_stiffness = 1.0e6;
    double m_desk_smoothing_epsilon = 1.0e-3;
    std::vector<PathPoint> m_desk_points;

    double m_shake_amplitude = 0.0;
    double m_shake_frequency_hz = 0.0;
    double m_shake_start_time = 0.0;
    double m_shake_end_time = 0.0;
    Eigen::Vector3d m_shake_direction = Eigen::Vector3d::UnitX();
    std::string m_shake_target_anchor = "support";
};

} // namespace ipc::rigid
