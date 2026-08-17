#include "onestring_constraint.hpp"

#include <physics/pose.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ipc::rigid {
namespace {

Eigen::Vector3d read_vec3(const nlohmann::json& value, const char* label)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string(label) + " must be a 3-element array");
    }
    return Eigen::Vector3d(
        value.at(0).get<double>(),
        value.at(1).get<double>(),
        value.at(2).get<double>());
}

void add_dense_block_triplets(
    std::vector<Eigen::Triplet<double>>& triplets,
    int row0,
    int col0,
    const Eigen::Matrix<double, 12, 12>& block,
    double threshold = 1.0e-15)
{
    for (int r = 0; r < block.rows(); ++r) {
        for (int c = 0; c < block.cols(); ++c) {
            const double value = block(r, c);
            if (std::abs(value) > threshold) {
                triplets.emplace_back(row0 + r, col0 + c, value);
            }
        }
    }
}

} // namespace

bool OneStringConstraint::load(
    const std::string& filename,
    std::size_t num_bodies,
    std::string& error_message)
{
    try {
        std::ifstream input(filename);
        if (!input) {
            throw std::runtime_error("unable to open OneString manifest: " + filename);
        }

        nlohmann::json root;
        input >> root;
        if (!root.is_object()) {
            throw std::runtime_error("OneString manifest root must be a JSON object");
        }
        if (!root.contains("string") || !root.at("string").is_object()) {
            throw std::runtime_error("OneString manifest does not contain a string object");
        }

        const nlohmann::json& string = root.at("string");
        m_num_bodies = num_bodies;
        m_path_points.clear();
        m_pull_schedule.clear();
        m_desk_points.clear();
        m_desk_enabled = false;

        // Version 2 explicitly lists world anchors and body guides.
        if (string.contains("path_points")) {
            const auto& points = string.at("path_points");
            if (!points.is_array()) {
                throw std::runtime_error("string.path_points must be an array");
            }
            for (const auto& point_json : points) {
                const std::string type = point_json.at("type").get<std::string>();
                PathPoint point;
                if (type == "world_anchor") {
                    point.type = PointType::WORLD_ANCHOR;
                    point.id = point_json.value("id", std::string("anchor"));
                    point.value = read_vec3(point_json.at("position"), "world anchor position");
                } else if (type == "body_guide") {
                    point.type = PointType::BODY_GUIDE;
                    point.body_id = point_json.at("body_id").get<int>();
                    point.value = read_vec3(
                        point_json.at("material_point"),
                        "body guide material_point");
                } else {
                    throw std::runtime_error("unknown string path point type: " + type);
                }
                m_path_points.push_back(point);
            }
        } else if (string.contains("guide_points")) {
            // Backward compatibility with onestring-abd-bridge-v1.  A fixed
            // support anchor is synthesized at the initial location of the
            // first guide.  This prevents the entire assembly from simply
            // free-falling while preserving the original guide path length.
            const auto& guides = string.at("guide_points");
            if (!guides.is_array() || guides.empty()) {
                throw std::runtime_error("string.guide_points must be a non-empty array");
            }

            PathPoint support;
            support.type = PointType::WORLD_ANCHOR;
            support.id = "support";
            support.value = read_vec3(
                guides.front().at("initial_world_point"),
                "first guide initial_world_point");
            m_path_points.push_back(support);

            for (const auto& guide_json : guides) {
                PathPoint guide;
                guide.type = PointType::BODY_GUIDE;
                guide.body_id = guide_json.at("body_id").get<int>();
                guide.value = read_vec3(
                    guide_json.at("material_point"),
                    "guide material_point");
                m_path_points.push_back(guide);
            }
        } else {
            throw std::runtime_error(
                "string must contain path_points (v2) or guide_points (v1)");
        }

        if (m_path_points.size() < 2) {
            throw std::runtime_error("OneString path requires at least two points");
        }
        for (const PathPoint& point : m_path_points) {
            if (point.type == PointType::BODY_GUIDE
                && (point.body_id < 0
                    || static_cast<std::size_t>(point.body_id) >= m_num_bodies)) {
                throw std::runtime_error(
                    "OneString body_id is outside the scene body range");
            }
        }

        const auto& schedule = string.at("pull_schedule");
        if (!schedule.is_array() || schedule.empty()) {
            throw std::runtime_error("string.pull_schedule must be a non-empty array");
        }
        for (const auto& entry : schedule) {
            const double time = entry.at("time").get<double>();
            const double length = entry.at("command_length").get<double>();
            if (!std::isfinite(time) || !std::isfinite(length) || length < 0.0) {
                throw std::runtime_error("pull schedule contains an invalid time or length");
            }
            m_pull_schedule.emplace_back(time, length);
        }
        std::sort(m_pull_schedule.begin(), m_pull_schedule.end());

        m_stiffness = string.value("stiffness", 1.0e6);
        m_smoothing_epsilon = string.value("smoothing_epsilon", 1.0e-9);
        m_use_exact_hessian = string.value("use_exact_hessian", false);
        if (!std::isfinite(m_stiffness) || m_stiffness <= 0.0) {
            throw std::runtime_error("string.stiffness must be positive");
        }
        if (!std::isfinite(m_smoothing_epsilon)
            || m_smoothing_epsilon <= 0.0) {
            throw std::runtime_error("string.smoothing_epsilon must be positive");
        }

        if (root.contains("shake_trajectory")) {
            const auto& shake = root.at("shake_trajectory");
            m_shake_amplitude = shake.value("amplitude", 0.0);
            m_shake_frequency_hz = shake.value("frequency_hz", 0.0);
            m_shake_start_time = shake.value("start_time", 0.0);
            m_shake_end_time = shake.value("end_time", 0.0);
            m_shake_target_anchor =
                shake.value("target_anchor", std::string("support"));
            if (shake.contains("direction")) {
                m_shake_direction = read_vec3(
                    shake.at("direction"), "shake direction");
            }
            const double direction_norm = m_shake_direction.norm();
            if (direction_norm > 1.0e-15) {
                m_shake_direction /= direction_norm;
            } else {
                m_shake_direction = Eigen::Vector3d::UnitX();
            }
        }

        if (root.contains("desk")) {
            const auto& desk = root.at("desk");
            if (!desk.is_object()) {
                throw std::runtime_error("desk must be an object");
            }
            m_desk_enabled = desk.value("enabled", false);
            m_desk_height = desk.value("top_z", 0.0);
            m_desk_stiffness = desk.value("stiffness", 1.0e6);
            m_desk_smoothing_epsilon =
                desk.value("smoothing_epsilon", 1.0e-3);
            if (!std::isfinite(m_desk_height)) {
                throw std::runtime_error("desk.top_z must be finite");
            }
            if (!std::isfinite(m_desk_stiffness) || m_desk_stiffness <= 0.0) {
                throw std::runtime_error("desk.stiffness must be positive");
            }
            if (!std::isfinite(m_desk_smoothing_epsilon)
                || m_desk_smoothing_epsilon <= 0.0) {
                throw std::runtime_error(
                    "desk.smoothing_epsilon must be positive");
            }
            if (m_desk_enabled) {
                if (!desk.contains("support_points")
                    || !desk.at("support_points").is_array()
                    || desk.at("support_points").empty()) {
                    throw std::runtime_error(
                        "enabled desk requires a non-empty support_points array");
                }
                for (const auto& point_json : desk.at("support_points")) {
                    PathPoint point;
                    point.type = PointType::BODY_GUIDE;
                    point.body_id = point_json.at("body_id").get<int>();
                    point.value = read_vec3(
                        point_json.at("material_point"),
                        "desk support material_point");
                    if (point.body_id < 0
                        || static_cast<std::size_t>(point.body_id) >= m_num_bodies) {
                        throw std::runtime_error(
                            "desk support body_id is outside the scene body range");
                    }
                    m_desk_points.push_back(point);
                }
            }
        }

        m_enabled = true;
        error_message.clear();
        return true;
    } catch (const std::exception& error) {
        m_enabled = false;
        error_message = error.what();
        return false;
    }
}

OneStringConstraint::EvaluatedPoint OneStringConstraint::evaluate_point(
    const PathPoint& point,
    const Eigen::VectorXd& x,
    int dim,
    double time) const
{
    if (dim != 3) {
        throw std::runtime_error("OneString ABD constraint currently supports only 3D scenes");
    }

    EvaluatedPoint evaluated;
    if (point.type == PointType::WORLD_ANCHOR) {
        evaluated.world = point.value + anchor_shake_offset(point, time);
        return evaluated;
    }

    constexpr int ndof = 12;
    const int offset = point.body_id * ndof;
    if (offset < 0 || offset + ndof > x.size()) {
        throw std::runtime_error("OneString guide references an unavailable body DoF block");
    }

    VectorMax12d pose_dof = x.segment(offset, ndof);
    const PoseD pose(pose_dof);
    evaluated.world = pose.position + pose.transform * point.value;
    evaluated.body_id = point.body_id;
    evaluated.jacobian.setZero();
    evaluated.jacobian.block<3, 3>(0, 0).setIdentity();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            evaluated.jacobian(row, 3 + row * 3 + col) = point.value(col);
        }
    }
    return evaluated;
}

OneStringConstraint::GeometryEvaluation OneStringConstraint::evaluate_geometry(
    const Eigen::VectorXd& x,
    int dim,
    double time,
    bool compute_hessian) const
{
    GeometryEvaluation result;
    result.gradient = Eigen::VectorXd::Zero(x.size());

    std::vector<Eigen::Triplet<double>> hessian_triplets;
    const double epsilon2 = m_smoothing_epsilon * m_smoothing_epsilon;

    std::vector<EvaluatedPoint> points;
    points.reserve(m_path_points.size());
    for (const PathPoint& point : m_path_points) {
        points.push_back(evaluate_point(point, x, dim, time));
    }

    for (std::size_t segment = 0; segment + 1 < points.size(); ++segment) {
        const EvaluatedPoint& a = points[segment];
        const EvaluatedPoint& b = points[segment + 1];
        const Eigen::Vector3d delta = b.world - a.world;
        const double length = std::sqrt(delta.squaredNorm() + epsilon2);
        result.length += length;

        const Eigen::Vector3d unit = delta / length;
        const Eigen::Matrix3d length_hessian =
            Eigen::Matrix3d::Identity() / length
            - (delta * delta.transpose()) / (length * length * length);

        if (a.body_id >= 0) {
            result.gradient.segment<12>(a.body_id * 12) -=
                a.jacobian.transpose() * unit;
        }
        if (b.body_id >= 0) {
            result.gradient.segment<12>(b.body_id * 12) +=
                b.jacobian.transpose() * unit;
        }

        if (compute_hessian) {
            if (a.body_id >= 0) {
                add_dense_block_triplets(
                    hessian_triplets,
                    a.body_id * 12,
                    a.body_id * 12,
                    a.jacobian.transpose() * length_hessian * a.jacobian);
            }
            if (b.body_id >= 0) {
                add_dense_block_triplets(
                    hessian_triplets,
                    b.body_id * 12,
                    b.body_id * 12,
                    b.jacobian.transpose() * length_hessian * b.jacobian);
            }
            if (a.body_id >= 0 && b.body_id >= 0) {
                const Eigen::Matrix<double, 12, 12> cross =
                    -a.jacobian.transpose() * length_hessian * b.jacobian;
                add_dense_block_triplets(
                    hessian_triplets,
                    a.body_id * 12,
                    b.body_id * 12,
                    cross);
                add_dense_block_triplets(
                    hessian_triplets,
                    b.body_id * 12,
                    a.body_id * 12,
                    cross.transpose());
            }
        }
    }

    if (compute_hessian) {
        result.hessian.resize(x.size(), x.size());
        result.hessian.setFromTriplets(
            hessian_triplets.begin(), hessian_triplets.end());
    }
    return result;
}

double OneStringConstraint::compute_energy(
    const Eigen::VectorXd& x,
    int dim,
    double time,
    double timestep_energy_scale,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess) const
{
    if (compute_grad) {
        grad = Eigen::VectorXd::Zero(x.size());
    }
    if (compute_hess) {
        hess.resize(x.size(), x.size());
        hess.setZero();
    }
    if (!m_enabled) {
        return 0.0;
    }

    double total_energy = 0.0;
    const GeometryEvaluation geometry =
        evaluate_geometry(x, dim, time, compute_hess && m_use_exact_hessian);
    const double violation = geometry.length - command_length(time);
    if (violation > 0.0) {
        const double factor = timestep_energy_scale * m_stiffness;
        total_energy += 0.5 * factor * violation * violation;
        if (compute_grad) {
            grad += factor * violation * geometry.gradient;
        }

        if (compute_hess) {
            std::vector<int> nonzero;
            nonzero.reserve(geometry.gradient.size());
            for (int i = 0; i < geometry.gradient.size(); ++i) {
                if (std::abs(geometry.gradient(i)) > 1.0e-14) {
                    nonzero.push_back(i);
                }
            }

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(nonzero.size() * nonzero.size());
            for (int row : nonzero) {
                for (int col : nonzero) {
                    triplets.emplace_back(
                        row,
                        col,
                        factor * geometry.gradient(row) * geometry.gradient(col));
                }
            }
            Eigen::SparseMatrix<double> string_hess(x.size(), x.size());
            string_hess.setFromTriplets(triplets.begin(), triplets.end());
            hess += string_hess;
            if (m_use_exact_hessian) {
                hess += factor * violation * geometry.hessian;
            }
        }
    }

    if (m_desk_enabled) {
        const double factor = timestep_energy_scale * m_desk_stiffness;
        std::vector<Eigen::Triplet<double>> desk_triplets;
        desk_triplets.reserve(m_desk_points.size() * 16);
        for (const PathPoint& point : m_desk_points) {
            const EvaluatedPoint evaluated = evaluate_point(point, x, dim, time);
            const double signed_penetration =
                m_desk_height - evaluated.world.z();
            const double smooth_norm = std::sqrt(
                signed_penetration * signed_penetration
                + m_desk_smoothing_epsilon * m_desk_smoothing_epsilon);
            const double penetration =
                0.5 * (signed_penetration + smooth_norm);
            const double penetration_derivative =
                0.5 * (1.0 + signed_penetration / smooth_norm);
            const double penetration_second_derivative =
                0.5 * m_desk_smoothing_epsilon * m_desk_smoothing_epsilon
                / (smooth_norm * smooth_norm * smooth_norm);
            total_energy += 0.5 * factor * penetration * penetration;
            const Eigen::Matrix<double, 1, 12> vertical_jacobian =
                evaluated.jacobian.row(2);
            if (compute_grad) {
                grad.segment<12>(evaluated.body_id * 12) -=
                    factor * penetration * penetration_derivative
                    * vertical_jacobian.transpose();
            }
            if (compute_hess) {
                const double curvature = factor
                    * (penetration_derivative * penetration_derivative
                        + penetration * penetration_second_derivative);
                add_dense_block_triplets(
                    desk_triplets,
                    evaluated.body_id * 12,
                    evaluated.body_id * 12,
                    curvature
                        * vertical_jacobian.transpose() * vertical_jacobian);
            }
        }
        if (compute_hess && !desk_triplets.empty()) {
            Eigen::SparseMatrix<double> desk_hess(x.size(), x.size());
            desk_hess.setFromTriplets(
                desk_triplets.begin(), desk_triplets.end());
            hess += desk_hess;
        }
    }

    return total_energy;
}

OneStringMetrics OneStringConstraint::metrics(
    const Eigen::VectorXd& x,
    int dim,
    double time) const
{
    OneStringMetrics result;
    if (!m_enabled) {
        return result;
    }
    const GeometryEvaluation geometry = evaluate_geometry(x, dim, time, false);
    result.string_length = geometry.length;
    result.command_length = command_length(time);
    result.constraint_violation =
        std::max(0.0, result.string_length - result.command_length);
    result.active = result.constraint_violation > 0.0;
    return result;
}

double OneStringConstraint::command_length(double time) const
{
    if (m_pull_schedule.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    if (time <= m_pull_schedule.front().first) {
        return m_pull_schedule.front().second;
    }
    if (time >= m_pull_schedule.back().first) {
        return m_pull_schedule.back().second;
    }

    const auto upper = std::upper_bound(
        m_pull_schedule.begin(),
        m_pull_schedule.end(),
        std::make_pair(time, std::numeric_limits<double>::infinity()));
    const auto lower = upper - 1;
    const double dt = upper->first - lower->first;
    if (dt <= 0.0) {
        return upper->second;
    }
    const double alpha = (time - lower->first) / dt;
    return (1.0 - alpha) * lower->second + alpha * upper->second;
}

Eigen::Vector3d OneStringConstraint::anchor_shake_offset(
    const PathPoint& point,
    double time) const
{
    if (point.type != PointType::WORLD_ANCHOR
        || point.id != m_shake_target_anchor
        || m_shake_amplitude == 0.0
        || m_shake_frequency_hz == 0.0
        || time < m_shake_start_time
        || time > m_shake_end_time) {
        return Eigen::Vector3d::Zero();
    }

    constexpr double two_pi = 6.283185307179586476925286766559;
    const double phase =
        two_pi * m_shake_frequency_hz * (time - m_shake_start_time);
    return m_shake_amplitude * std::sin(phase) * m_shake_direction;
}

} // namespace ipc::rigid
