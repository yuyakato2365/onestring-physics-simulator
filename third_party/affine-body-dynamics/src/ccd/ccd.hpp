#pragma once

#include <Eigen/Core>

#include <nlohmann/json.hpp>

#include "ipc/candidates/vertex_vertex.hpp"
#include "ipc/candidates/edge_edge.hpp"
#include "ipc/candidates/edge_face.hpp"
#include "ipc/candidates/edge_vertex.hpp"
#include "ipc/candidates/face_vertex.hpp"

#include <ccd/impact.hpp>
#include <physics/rigid_body_assembler.hpp>

namespace ipc::rigid {

/// @brief Possible methods for detecting all edge vertex collisions.
enum DetectionMethod {
    BRUTE_FORCE, ///< @brief Use brute-force to detect all collisions
    HASH_GRID, ///< @brief Use a spatial data structure to detect all collisions
    BVH,       ///< @brief Use a BVH to detect all collisions
    IAABB,     ///< @brief Use the intersecting space to collect the potential candidates
    // @javidf: is IAABB a detectionMethod.
};
// @javidf: making this dumb serialize_enum case insensitive.
NLOHMANN_JSON_SERIALIZE_ENUM(
    DetectionMethod,
    { { HASH_GRID, "hash_grid" },
      { BRUTE_FORCE, "brute_force" },
      { BVH, "bvh" },
      { IAABB, "IAABB"} });

/// @brief Possible trajectories of vertices in a rigid body.
enum TrajectoryType {
    /// @brief Linearization of the rotation component of rigid body
    /// trajectories.
    LINEAR,
    /// @brief Using additive contineous collision detection (ACCD).
    ACCD,
    // /// @brief Piceiwise linearization of the the rotation component of rigid
    // /// body trajectories. CCD is computed conservativly using minimum
    // /// separation CCD to bound the difference between RIGID.
    // PIECEWISE_LINEAR,
    /// @brief Fully nonlinear rigid trajectories.
    RIGID
    // /// @brief Same trajectory as RIGID, but the time of impact is computed
    // /// using Redon et al. [2002].
    // REDON
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    TrajectoryType,
    { { LINEAR, "linear" },
      { RIGID, "rigid" },
    //   { REDON, "redon" },
      { ACCD, "ACCD" } });

namespace CollisionType {
    static const int EDGE_VERTEX = 1;
    static const int EDGE_EDGE = 2;
    static const int FACE_VERTEX = 4;
} // namespace CollisionType

///////////////////////////////////////////////////////////////////////////////
// CCD
///////////////////////////////////////////////////////////////////////////////

/// @brief Find all collisions in one time step.
void detect_collisions(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const int collision_types,
    Impacts& impacts,
    DetectionMethod method,
    TrajectoryType trajectory,
    const double inflation_radius = 0.0);

///////////////////////////////////////////////////////////////////////////////
// Broad-Phase
///////////////////////////////////////////////////////////////////////////////

/// @brief Use broad-phase method to create a set of candidate collisions.
void detect_collision_candidates(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const int collision_types,
    Candidates& candidates,
    DetectionMethod method,
    TrajectoryType trajectory,
    const double inflation_radius = 0.0);

// // Discontineous collision detection. A single pose input is needed
// /// @brief Use broad-phase method to create a set of candidate collisions.
// void detect_collision_candidates(
//     const RigidBodyAssembler& bodies,
//     const PosesD& poses,
//     const int collision_types,
//     Candidates& candidates,
//     DetectionMethod method,
//     const double inflation_radius = 0.0);

///////////////////////////////////////////////////////////////////////////////
// Narrow-Phase
///////////////////////////////////////////////////////////////////////////////

void detect_collisions_from_candidates(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const Candidates& candidates,
    Impacts& impacts,
    TrajectoryType trajectory,
    const double inflation_radius = 0.0);

/// @brief Determine if a single edge-vertext pair intersects.
bool edge_vertex_ccd(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const EdgeVertexCandidate& ev_candidate,
    double& toi,
    TrajectoryType trajectory,
    double earliest_toi = 1,
    double minimum_separation_distance = 0);

bool edge_edge_ccd(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const EdgeEdgeCandidate& ee_candidate,
    double& toi,
    TrajectoryType trajectory,
    double earliest_toi = 1,
    double minimum_separation_distance = 0);

bool face_vertex_ccd(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const FaceVertexCandidate& fv_candidate,
    double& toi,
    TrajectoryType trajectory,
    double earliest_toi = 1,
    double minimum_separation_distance = 0);

double edge_vertex_closest_point(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const EdgeVertexCandidate& candidate,
    double toi,
    TrajectoryType trajectory);

void edge_edge_closest_point(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const EdgeEdgeCandidate& candidate,
    double toi,
    double& alpha,
    double& beta,
    TrajectoryType trajectory);

void face_vertex_closest_point(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const FaceVertexCandidate& candidate,
    double toi,
    double& u,
    double& v,
    TrajectoryType trajectory);

} // namespace ipc::rigid
