#include "rigid_body_problem.hpp"

#include <iostream>

#include <tbb/parallel_for_each.h>

#include <finitediff.hpp>
#include <igl/PI.h>
#include <igl/predicates/segment_segment_intersect.h>

#include <ipc/utils/intersection.hpp>

#include <ccd/rigid/broad_phase.hpp>
// #include <ccd/rigid/rigid_body_hash_grid.hpp>
#include <io/read_rb_scene.hpp>
#include <io/serialize_json.hpp>
#include <logger.hpp>
#include <profiler.hpp>
#include <utils/eigen_ext.hpp>

#include <ipc/ipc.hpp>

namespace ipc::rigid {

RigidBodyProblem::RigidBodyProblem()
    : coefficient_restitution(0)
    , coefficient_friction(0)
    , collision_eps(2)
    , m_timestep(0.01)
    , do_intersection_check(false)
{
    gravity.setZero(3);
}

bool RigidBodyProblem::settings(const nlohmann::json& params)
{
    collision_eps = params["collision_eps"];
    coefficient_restitution = params["coefficient_restitution"];
    coefficient_friction = params["coefficient_friction"];
    if (coefficient_friction < 0 || coefficient_friction > 1) {
        spdlog::warn(
            "Coefficient of friction (μ={:g}) is outside the standard "
            "[0, 1]",
            coefficient_friction);
    } else if (coefficient_friction == 0) {
        spdlog::info(
            "Disabling friction because coefficient of friction is zero");
    }

    std::vector<RigidBody> rbs;
    bool success = read_rb_scene(params, rbs);
    if (!success) {
        spdlog::error("Unable to read rigid body scene!");
        return false;
    }

    init(rbs);

    from_json(params["gravity"], gravity);
    assert(gravity.size() >= dim());
    gravity.conservativeResize(dim());

    do_intersection_check = params["do_intersection_check"];
    return true;
}

nlohmann::json RigidBodyProblem::settings() const
{
    nlohmann::json json;

    json["collision_eps"] = collision_eps;
    json["coefficient_restitution"] = coefficient_restitution;
    json["coefficient_friction"] = coefficient_friction;
    json["gravity"] = to_json(gravity);
    json["do_intersection_check"] = do_intersection_check;
    return json;
}

void RigidBodyProblem::init(const std::vector<RigidBody>& rbs)
{
    m_assembler.init(rbs);

    m_collision_mesh = CollisionMesh(
        m_assembler.world_vertices(), m_assembler.m_edges, m_assembler.m_faces);

    update_constraints();

    for (size_t i = 0; i < num_bodies(); ++i) {
        auto& rb = m_assembler[i];
        spdlog::info(
            "rb={:d} group_id={:d} mass={:g}", i, rb.group_id,
            rb.mass, fmt_eigen(rb.moment_of_inertia));
    }
    spdlog::info("average_mass={:g}", m_assembler.average_mass);

    // Compute world diagonal
    Eigen::MatrixXd V = vertices();
    init_bbox_diagonal =
        (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();
    spdlog::info("init_bbox_diagonal={:g}", init_bbox_diagonal);

    // Ensure the dimension of gravity matches the dimension of the problem.
    gravity = gravity.head(dim());

    if (detect_intersections(m_assembler.rb_poses_t1())) {
        spdlog::error("The initial state contains intersections!");
    } else {
        spdlog::info("no intersections found in initial state");
    }
}

nlohmann::json RigidBodyProblem::state() const
{
    nlohmann::json json;
    std::vector<nlohmann::json> rbs;
    Eigen::VectorXd p =
        Eigen::VectorXd::Zero(PoseD::dim_to_pos_ndof(dim())); // Linear momentum
    Eigen::VectorXd L = Eigen::VectorXd::Zero(PoseD::dim_to_trans_ndof(
        dim()));    // Generalized momemtum associated with linear transform
    double T = 0.0; // Kinetic energy
    double G = 0.0; // Potential energy

    // Temp holder for generalized momentum for each rb
    Eigen::VectorXd gm_i = Eigen::VectorXd::Zero(PoseD::dim_to_ndof(
        dim())); // Generalized momemtum associated with linear transform
    for (auto& rb : m_assembler.m_rbs) {
        nlohmann::json jrb;
        jrb["position"] = to_json(Eigen::VectorXd(rb.pose.position));
        jrb["transform"] = to_json(Eigen::MatrixXd(rb.pose.transform));
        jrb["linear_velocity"] = to_json(Eigen::VectorXd(rb.velocity.position));
        jrb["transform_velocity"] =
            to_json(Eigen::MatrixXd(rb.velocity.transform));
        rbs.push_back(jrb);

        // Generalized momentum
        gm_i = rb.mass_matrix * rb.velocity.dof();

        // Translational and tranformation momentum
        p += gm_i.head(PoseD::dim_to_pos_ndof(dim()));
        L += gm_i.tail(PoseD::dim_to_trans_ndof(dim()));

        T += 0.5 * rb.velocity.dof().transpose() * rb.mass_matrix
            * rb.velocity.dof();

        // TODO: How does constraining the body's position affect potential
        // energy?
        G -= rb.mass * gravity.dot(rb.pose.position);
    }

    json["rigid_bodies"] = rbs;
    json["linear_momentum"] = to_json(p);
    json["angular_momentum"] = to_json(L);
    json["kinetic_energy"] = T;
    json["potential_energy"] = G;
    return json;
}

void RigidBodyProblem::state(const nlohmann::json& args)
{
    nlohmann::json json;
    auto& rbs = args["rigid_bodies"];
    assert(rbs.size() == num_bodies());
    size_t i = 0;
    for (auto& jrb : args["rigid_bodies"]) {
        from_json(jrb["position"], m_assembler[i].pose.position);
        from_json(jrb["transform"], m_assembler[i].pose.transform);
        from_json(jrb["linear_velocity"], m_assembler[i].velocity.position);
        from_json(jrb["transform_velocity"], m_assembler[i].velocity.transform);
        i++;
    }
}

void RigidBodyProblem::update_dof()
{
    poses_t0 = m_assembler.rb_poses_t0();
    x0 = this->poses_to_dofs(poses_t0);
    num_vars_ = x0.size();
}

void RigidBodyProblem::update_constraints()
{
    PROFILE_POINT("RigidBodyProblem::update_constraints");
    PROFILE_START();

    update_dof();
    constraint().initialize();

    PROFILE_END();
}

void RigidBodyProblem::init_solve()
{
    return solver().init_solve(starting_point());
}

OptimizationResults RigidBodyProblem::solve_constraints()
{
    return solver().solve(starting_point());
}

OptimizationResults RigidBodyProblem::step_solve()
{
    return solver().step_solve();
}

bool RigidBodyProblem::take_step(const Eigen::VectorXd& dof)
{
    ////////////////////////////////////////////////////////////////////////
    // WARNING: This only assumes an implicit euler velocity update. For
    // more updates look at the overridden version in
    // distance_barrier_rb_problem.
    ////////////////////////////////////////////////////////////////////////

    // update final pose
    // -------------------------------------
    m_assembler.set_rb_poses(this->dofs_to_poses(dof));
    PosesD poses_q1 = m_assembler.rb_poses_t1();

    // Update the velocities
    // This need to be done AFTER updating poses
    for (RigidBody& rb : m_assembler.m_rbs) {
        if (rb.type != RigidBodyType::DYNAMIC) {
            continue;
        }

        // Assume linear velocity through the time-step.
        rb.velocity.position =
            (rb.pose.position - rb.pose_prev.position) / timestep();

        rb.velocity.transform =
            (rb.pose.transform - rb.pose_prev.transform) / timestep();
    }

    if (do_intersection_check) {
        // Check for intersections instead of collision along the entire
        // step. We only guarentee a piecewise collision-free trajectory.
        // return detect_collisions(poses_t0, poses_q1,
        // CollisionCheck::EXACT);
        return detect_intersections(poses_q1);
    }
    return false;
}

bool RigidBodyProblem::detect_collisions(
    const PosesD& poses_q0,
    const PosesD& poses_q1,
    const CollisionCheck check_type) const
{
    Impacts impacts;

    double scale =
        check_type == CollisionCheck::EXACT ? 1.0 : (1.0 + collision_eps);
    PosesD scaled_pose_q1 = interpolate(poses_q0, poses_q1, scale);

    constraint().construct_collision_set(
        m_assembler, poses_q0, scaled_pose_q1, impacts);

    return impacts.size();
}

// Check if the geometry is intersecting
bool RigidBodyProblem::detect_intersections(const PosesD& poses) const
{
    if (num_bodies() <= 1) {
        return false;
    }

    PROFILE_POINT("RigidBodyProblem::detect_intersections");
    PROFILE_START();

    const Eigen::MatrixXd vertices = m_assembler.world_vertices(poses);
    const Eigen::MatrixXi& edges = this->edges();
    const Eigen::MatrixXi& faces = this->faces();

    bool is_intersecting = false;
    if (dim() == 2) { // Need to check segment-segment intersections in 2D
        assert(vertices.cols() == 2);

        double inflation_radius = 1e-8; // Conservative broad phase
        std::vector<std::pair<int, int>> close_bodies =
            m_assembler.close_bodies(poses, poses, inflation_radius);
        if (close_bodies.size() == 0) {
            PROFILE_END();
            return false;
        }

        // RigidBodyHashGrid hashgrid;
        // hashgrid.resize(m_assembler, poses, close_bodies, inflation_radius);
        // hashgrid.addBodies(m_assembler, poses, close_bodies,
        // inflation_radius);

        const Eigen::VectorXi& vertex_group_ids = group_ids();
        auto can_vertices_collide = [&vertex_group_ids](size_t vi, size_t vj) {
            return vertex_group_ids[vi] != vertex_group_ids[vj];
        };

        // std::vector<EdgeEdgeCandidate> ee_candidates;
        // hashgrid.getEdgeEdgePairs(edges, ee_candidates,
        // can_vertices_collide);

        // for (const EdgeEdgeCandidate& ee_candidate : ee_candidates) {
        //    if (igl::predicates::segment_segment_intersect(
        //            vertices.row(edges(ee_candidate.edge0_id,
        //            0)).head<2>(),
        //            vertices.row(edges(ee_candidate.edge0_id,
        //            1)).head<2>(),
        //            vertices.row(edges(ee_candidate.edge1_id,
        //            0)).head<2>(),
        //            vertices.row(edges(ee_candidate.edge1_id,
        //            1)).head<2>())
        //        ) {
        //        is_intersecting = true;
        //        break;
        //    }
        //}

        CollisionMesh collision_mesh(vertices, edges, faces);
        collision_mesh.can_collide = can_vertices_collide;

        is_intersecting = ipc::has_intersections(
            collision_mesh, vertices, BroadPhaseMethod::HASH_GRID,
            inflation_radius);

    } else { // Need to check segment-triangle intersections in 3D
        assert(dim() == 3);

        std::vector<EdgeFaceCandidate> ef_candidates;
        detect_intersection_candidates_rigid_bvh(
            m_assembler, poses, ef_candidates);

        for (const EdgeFaceCandidate& ef_candidate : ef_candidates) {
            if (is_edge_intersecting_triangle(
                    vertices.row(edges(ef_candidate.edge_id, 0)),
                    vertices.row(edges(ef_candidate.edge_id, 1)),
                    vertices.row(faces(ef_candidate.face_id, 0)),
                    vertices.row(faces(ef_candidate.face_id, 1)),
                    vertices.row(faces(ef_candidate.face_id, 2)))) {
                is_intersecting = true;
                break;
            }
        }
    }

    PROFILE_END();

    return is_intersecting;
}

} // namespace ipc::rigid
