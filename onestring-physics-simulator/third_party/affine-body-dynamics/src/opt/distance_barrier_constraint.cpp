#include <igl/Timer.h>

#include "distance_barrier_constraint.hpp"

#include <mutex>
#include <tbb/parallel_for_each.h>

#include <igl/slice_mask.h>
#include <ipc/ipc.hpp>

#include <ccd/rigid/broad_phase.hpp>
#include <ccd/linear/broad_phase.hpp>
// #include <ccd/rigid/rigid_body_hash_grid.hpp>
#include <ccd/save_queries.hpp>
// #include <geometry/distance.hpp>
#include <io/serialize_json.hpp>
#include <logger.hpp>
#include <profiler.hpp>

namespace ipc::rigid {

DistanceBarrierConstraint::DistanceBarrierConstraint(const std::string& name)
    : CollisionConstraint(name)
    , initial_barrier_activation_distance(1e-3)
    , minimum_separation_distance(0.0)
    , m_barrier_activation_distance(0.0)
{
}

void DistanceBarrierConstraint::settings(const nlohmann::json& json)
{
    CollisionConstraint::settings(json);
    initial_barrier_activation_distance =
        json["initial_barrier_activation_distance"];
    minimum_separation_distance = json["minimum_separation_distance"];
}

nlohmann::json DistanceBarrierConstraint::settings() const
{
    nlohmann::json json = CollisionConstraint::settings();
    json["initial_barrier_activation_distance"] =
        initial_barrier_activation_distance;
    json["minimum_separation_distance"] = minimum_separation_distance;
    return json;
}

void DistanceBarrierConstraint::initialize()
{
    m_barrier_activation_distance = initial_barrier_activation_distance;
    CollisionConstraint::initialize();
}
// @javidf: it seems that has_active_collisions() and compute_earliest_toi() has
// the same functionality. PLEASE CHECK IT AGAIN.
bool DistanceBarrierConstraint::has_active_collisions(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1) const
{
    PROFILE_POINT("DistanceBarrierConstraint::has_active_collisions");
    PROFILE_START();

    // NAMED_PROFILE_POINT(
    //    "DistanceBarrierConstraint::has_active_collisions_narrow_phase",
    //    NARROW_PHASE);

    // This function will profile itself
    Candidates candidates;
    // std::cout << "has collision t0 p1 : \n" << poses_t0.at(1).position << "
    // \n     t1 p1 : \n" << poses_t1.at(1).position << std::endl;
    detect_collision_candidates(
        bodies, poses_t0, poses_t1, dim_to_collision_type(bodies.dim()),
        candidates, detection_method, trajectory_type,
        /*inflation_radius=*/0.5 * minimum_separation_distance);

    // PROFILE_START(NARROW_PHASE)
    bool has_collisions = has_active_collisions_narrow_phase(
        bodies, poses_t0, poses_t1, candidates);
    // PROFILE_END(NARROW_PHASE)
    PROFILE_END();

    return has_collisions;
}

bool DistanceBarrierConstraint::has_active_collisions_narrow_phase(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const Candidates& candidates) const
{
    PROFILE_POINT("DistanceBarrierConstraint::has_active_collisions");
    PROFILE_START();

    TrajectoryType overloaded_trajectory =
        TrajectoryType::ACCD; // ABD default follows:
                              // trajectory_type ==
                              // TrajectoryType::PIECEWISE_LINEAR ?
                              // TrajectoryType::RIGID : trajectory_type;

    for (const auto& ev_candidate : candidates.ev_candidates) {
        double toi;
        bool are_colliding = edge_vertex_ccd(
            bodies, poses_t0, poses_t1, ev_candidate, toi,
            overloaded_trajectory, minimum_separation_distance);
        if (are_colliding) {
            // save_ccd_candidate(bodies, poses_t0, poses_t1, ev_candidate);
            PROFILE_END();
            return true;
        }
    }
    for (const auto& fv_candidate : candidates.fv_candidates) {
        double toi;
        bool are_colliding = face_vertex_ccd(
            bodies, poses_t0, poses_t1, fv_candidate, toi,
            overloaded_trajectory, minimum_separation_distance);
        if (are_colliding) {
            // save_ccd_candidate(bodies, poses_t0, poses_t1, fv_candidate);
            PROFILE_END();
            return true;
        }
    }
    for (const auto& ee_candidate : candidates.ee_candidates) {
        double toi;
        bool are_colliding = edge_edge_ccd(
            bodies, poses_t0, poses_t1, ee_candidate, toi,
            overloaded_trajectory, minimum_separation_distance);
        if (are_colliding) {
            // save_ccd_candidate(bodies, poses_t0, poses_t1, ee_candidate);
            PROFILE_END();
            return true;
        }
    }
    PROFILE_END();
    return false;
}

double DistanceBarrierConstraint::compute_earliest_toi(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1)
{
    PROFILE_POINT("DistanceBarrierConstraint::compute_earliest_toi");
    PROFILE_START();

    Candidates candidates;

    detect_collision_candidates(
        bodies, poses_t0, poses_t1, dim_to_collision_type(bodies.dim()),
        candidates, detection_method, trajectory_type,
        /*inflation_radius=*/0.5 * minimum_separation_distance);

    double earliest_toi = compute_earliest_toi_narrow_phase(
        bodies, poses_t0, poses_t1, candidates);

    PROFILE_END();

    return earliest_toi;
}

double DistanceBarrierConstraint::compute_earliest_toi_narrow_phase(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const Candidates& candidates) const
{
    PROFILE_POINT(
        "DistanceBarrierConstraint::compute_earliest_toi_narrow_phase");
    PROFILE_START();

    int collision_count = 0;
    double earliest_toi = 1;
    std::mutex earliest_toi_mutex;

    const size_t num_ev = candidates.ev_candidates.size();
    const size_t num_ee = candidates.ee_candidates.size();
    const size_t num_fv = candidates.fv_candidates.size();

    // Do a single block range over all three candidate vectors
    tbb::parallel_for(
        tbb::blocked_range<int>(0, candidates.size()),
        [&](tbb::blocked_range<int> r) {
            for (int i = r.begin(); i < r.end(); i++) {
                double toi = std::numeric_limits<double>::infinity();
                bool are_colliding;

                if (i < num_ev) {
                    // PROFILE_START(EV_NARROW_PHASE);
                    are_colliding = edge_vertex_ccd(
                        bodies, poses_t0, poses_t1, candidates.ev_candidates[i],
                        toi, trajectory_type, earliest_toi,
                        minimum_separation_distance);
                    // PROFILE_END(EV_NARROW_PHASE);
                } else if (i - num_ev < num_ee) {
                    // PROFILE_START(EE_NARROW_PHASE);
                    are_colliding = edge_edge_ccd(
                        bodies, poses_t0, poses_t1,
                        candidates.ee_candidates[i - num_ev], toi,
                        trajectory_type, earliest_toi,
                        minimum_separation_distance);
                    // PROFILE_END(EE_NARROW_PHASE);
                } else {
                    assert(i - num_ev - num_ee < num_fv);
                    // PROFILE_START(FV_NARROW_PHASE);
                    are_colliding = face_vertex_ccd(
                        bodies, poses_t0, poses_t1,
                        candidates.fv_candidates[i - num_ev - num_ee], toi,
                        trajectory_type, earliest_toi,
                        minimum_separation_distance);
                    // PROFILE_END(FV_NARROW_PHASE);
                }

                if (are_colliding && toi == 0) {
                    if (i < num_ev) {
                        spdlog::error("Edge-vertex CCD resulted in toi=0!");
                        save_ccd_candidate(
                            bodies, poses_t0, poses_t1,
                            candidates.ev_candidates[i]);
                    } else if (i - num_ev < num_ee) {
                        spdlog::error("Edge-edge CCD resulted in toi=0!");
                        save_ccd_candidate(
                            bodies, poses_t0, poses_t1,
                            candidates.ee_candidates[i - num_ev]);
                    } else {
                        assert(i - num_ev - num_ee < num_fv);
                        spdlog::error("Face-vertex CCD resulted in toi=0!");
                        save_ccd_candidate(
                            bodies, poses_t0, poses_t1,
                            candidates.fv_candidates[i - num_ev - num_ee]);
                    }
                }

                if (are_colliding) {
                    std::scoped_lock lock(earliest_toi_mutex);
                    collision_count++;
                    if (toi < earliest_toi) {
                        earliest_toi = toi;
                    }
                }
            }
        });

    double percent_correct = candidates.size() == 0
        ? 100
        : (double(collision_count) / candidates.size() * 100);

    spdlog::debug(
        "num_candidates={:d} num_collisions={:d} percentage={:g}%",
        candidates.size(), collision_count, percent_correct);

    PROFILE_END();

    return collision_count ? earliest_toi
                           : std::numeric_limits<double>::infinity();
}

void DistanceBarrierConstraint::construct_constraint_set(
    const CollisionMesh& collision_mesh,
    const RigidBodyAssembler& bodies,
    const PosesD& poses,
    CollisionConstraints& constraint_set) const
{
    static PosesD cached_poses;
    static CollisionConstraints cached_constraint_set;

    if (bodies.num_bodies() <= 1) {
        return;
    }

    if (poses == cached_poses) {
        constraint_set = cached_constraint_set;
        return;
    }

    PROFILE_POINT("DistanceBarrierConstraint::construct_constraint_set");
    PROFILE_START();

    double dhat = this->m_barrier_activation_distance;
    double dmin = this->minimum_separation_distance;
    const double inflation_radius = (dhat + dmin) / 2.0;

    Candidates candidates;
    switch (trajectory_type) {
    case TrajectoryType::LINEAR:
    case TrajectoryType::ACCD:
        detect_collision_candidates_linear(
            bodies, poses, dim_to_collision_type(bodies.dim()), candidates,
            detection_method, inflation_radius);
        // detect_collision_candidates_rigid(
        //     bodies, poses, dim_to_collision_type(bodies.dim()), candidates,
        //     detection_method, inflation_radius);
        break;
    // case TrajectoryType::PIECEWISE_LINEAR:
    case TrajectoryType::RIGID:
    // case TrajectoryType::REDON:
        // default; @fjavid this is not required for ABD but has been kept as an
        // option
        detect_collision_candidates_rigid(
            bodies, poses, dim_to_collision_type(bodies.dim()), candidates,
            detection_method, inflation_radius);
        break;
    }
    // detect_collision_candidates_linear(
    //         bodies, poses, dim_to_collision_type(bodies.dim()), candidates,
    //         detection_method, inflation_radius); // Using linear instead of
    //         rigid collision candidates.
    // default follows:
    // detect_collision_candidates_rigid(
    //     bodies, poses, dim_to_collision_type(bodies.dim()), candidates,
    //     detection_method, inflation_radius);
    Eigen::MatrixXd V = bodies.world_vertices(poses);
    // @javidf: PLEASE REVISE THIS. Move everything to the toolkit or keep all
    // here. This is a bit confusing. THe first part is done in the main code,
    // src/ccd,while hte following section uses ipc toolkit. But this should
    // work for now.
    {
        PROFILE_POINT("CollisionConstraints::build");
        PROFILE_START();
        constraint_set.build(candidates, collision_mesh, V, dhat, dmin);
        PROFILE_END();
    }
    // ipc::construct_constraint_set(
    //    candidates, /*V_rest=*/V, V, bodies.m_edges, bodies.m_faces,
    //    /*dhat=*/dhat, constraint_set, bodies.m_faces_to_edges,
    //    /*dmin=*/dmin);

    cached_poses = poses;
    cached_constraint_set = constraint_set;

    PROFILE_END();
}

double DistanceBarrierConstraint::compute_minimum_distance(
    const CollisionMesh& collision_mesh,
    const RigidBodyAssembler& bodies,
    const PosesD& poses) const
{
    PROFILE_POINT("DistanceBarrierConstraint::compute_minimum_distance");
    PROFILE_START();

    Eigen::MatrixXd V = bodies.world_vertices(poses);
    CollisionConstraints constraint_set;
    construct_constraint_set(collision_mesh, bodies, poses, constraint_set);
    double minimum_distance =
        sqrt(constraint_set.compute_minimum_distance(collision_mesh, V));

    PROFILE_END();

    return minimum_distance;
}

} // namespace ipc::rigid
