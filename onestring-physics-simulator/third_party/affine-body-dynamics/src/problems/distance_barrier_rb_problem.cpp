#include "distance_barrier_rb_problem.hpp"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/ipc.hpp>

#ifdef ABD_WITH_DERIVATIVE_CHECK
#include <finitediff.hpp>
#endif

#include <utils/eigen_ext.hpp>
#include "physics/pose.hpp"
#include <constants.hpp>
// #include <geometry/distance.hpp>
#include <solvers/ipc_solver.hpp>
#include <utils/not_implemented_error.hpp>

#include <logger.hpp>
#include <profiler.hpp>

namespace ipc::rigid {

DistanceBarrierRBProblem::DistanceBarrierRBProblem()
    : m_barrier_stiffness(1)
    , m_orthogonal_stiffness(1.0E11)
    , min_distance(-1)
    , m_had_collisions(false)
    , static_friction_speed_bound(1e-3)
    , friction_iterations(1)
    , body_energy_integration_method(DEFAULT_BODY_ENERGY_INTEGRATION_METHOD)
{
}

bool DistanceBarrierRBProblem::settings(const nlohmann::json& params)
{
    m_constraint.settings(params["distance_barrier_constraint"]);

    // Initialize IPC solver
    std::string solver_name = params["solver"].get<std::string>();
    if (solver_name != "ipc_solver"){
        spdlog::warn("Only ipc_solver is supported!");
    }
    m_opt_solver = std::make_shared<IPCSolver>();
    m_opt_solver->settings(params[solver_name]);
    m_opt_solver->set_problem(*this);
    if (m_opt_solver->has_inner_solver()) {
        m_opt_solver->inner_solver().settings(
            params[m_opt_solver->inner_solver().name()]);
    }

    // Friction
    static_friction_speed_bound =
        params["friction_constraints"]["static_friction_speed_bound"];
    friction_iterations = params["friction_constraints"]["iterations"];

    body_energy_integration_method =
        params["rigid_body_problem"]["time_stepper"]
            .get<BodyEnergyIntegrationMethod>();

    m_orthogonal_stiffness =
        params["rigid_body_problem"]["orthogonality_stiffness"].get<double>();

    bool success = RigidBodyProblem::settings(params["rigid_body_problem"]);

    // Linear Constraints
    m_linear_cons =
        LinearConstraintHandler(); // Deleting old handler in case of reload
    success = success
        && m_linear_cons.settings(
            params["rigid_body_problem"]["linear_constraints"], m_assembler);
    m_linear_cons.init();

    if (!success) {
        return false;
    }

    if (friction_iterations == 0) {
        spdlog::info("Disabling friction because friction iterations is zero");
        coefficient_friction = 0; // This disables all friction computation
    }
    // else if (friction_iterations < 0) {
    //     friction_iterations = Constants::MAXIMUM_FRICTION_ITERATIONS;
    // }

    min_distance = compute_min_distance(starting_point());
    if (min_distance < 0) {
        spdlog::info(
            "init_min_distance>d+dmin={:.8e}",
            barrier_activation_distance()
                + m_constraint.minimum_separation_distance);
    } else {
        spdlog::info("init_min_distance={:.8e}", min_distance);
    }

    return true;
}

nlohmann::json DistanceBarrierRBProblem::settings() const
{
    nlohmann::json json = RigidBodyProblem::settings();
    json["friction_iterations"] = friction_iterations;
    json["static_friction_speed_bound"] = static_friction_speed_bound;
    json["time_stepper"] = body_energy_integration_method;
    return json;
}

nlohmann::json DistanceBarrierRBProblem::state() const
{
    nlohmann::json json = RigidBodyProblem::state();
    if (min_distance < 0) {
        json["min_distance"] = nullptr;
    } else {
        json["min_distance"] = min_distance;
    }
    return json;
}

Eigen::VectorXi DistanceBarrierRBProblem::free_dof() const
{

    Eigen::VectorXi free_dof(this->num_vars());
    for (int i = 0; i < this->num_vars(); ++i) {
        free_dof[i] = i;
    }
    return free_dof;
}

////////////////////////////////////////////////////////////
// Rigid Body Problem

void DistanceBarrierRBProblem::simulation_step(
    bool& had_collisions, bool& _has_intersections, bool solve_collisions)
{
    PROFILE_POINT("DistanceBarrierRBProblem::simulation_step");
    PROFILE_START();

    // Advance the poses, but leave the current pose unchanged for now.
    for (size_t i = 0; i < num_bodies(); i++) {
        m_assembler[i].pose_prev = m_assembler[i].pose;
        m_assembler[i].velocity_prev = m_assembler[i].velocity;
    }
    // Update the stored poses and inital value for the solver
    // @javidf: we should comment this line because this is executed in
    // update_constraints() in RigidBodyProblem::update_constraints(), line 159
    // below. update_dof();

    // Reset m_had_collision which will be filled in by has_collisions().
    m_had_collisions = false;
    m_num_contacts = 0;

    // Disable barriers if solve_collision == false
    this->m_use_barriers = solve_collisions;

    // Solve constraints updates the constraints and takes the step
    update_constraints();
    // std::cout << "step restarts: \n";
    opt_result = solve_constraints();
    _has_intersections = take_step(opt_result.x);
    step_kinematic_bodies();
    had_collisions = m_had_collisions;

    PROFILE_END();
}

void DistanceBarrierRBProblem::update_constraints()
{
    PROFILE_POINT("DistanceBarrierRBProblem::update_constraints");
    PROFILE_START();

    RigidBodyProblem::update_constraints();

    CollisionConstraints collision_constraints;
    m_constraint.construct_constraint_set(
        m_collision_mesh, m_assembler, poses_t0, collision_constraints);

    Eigen::SparseMatrix<double> hess;
    compute_barrier_term(
        x0, collision_constraints, grad_barrier_t0, hess,
        /*compute_grad=*/true, /*compute_hess=*/false);

    update_friction_constraints(collision_constraints, poses_t0);

    init_augmented_lagrangian();

    PROFILE_END();
}

void DistanceBarrierRBProblem::update_friction_constraints(
    const CollisionConstraints& collision_constraints, const PosesD& poses)
{
    if (coefficient_friction <= 0) {
        return;
    }

    PROFILE_POINT("DistanceBarrierRBProblem::update_friction_constraints");
    PROFILE_START();

    // The fricition constraints are constant through out the entire
    // lagging iteration.
    friction_constraints.clear();
    Eigen::MatrixXd V0 = m_assembler.world_vertices(poses);
    // Eigen::MatrixXd V0 =
    //     m_collision_mesh.displace_vertices(m_assembler.world_vertices(poses));
    friction_constraints.build(
        m_collision_mesh, V0, collision_constraints,
        barrier_activation_distance(), barrier_stiffness(),
        coefficient_friction);

    PROFILE_END();
}

// @javidf: this should replace init_augmented_lagrangian(). They both exist
// here now.
void DistanceBarrierRBProblem::init_augmented_lagrangian()
{
    PROFILE_POINT("DistanceBarrierRBProblem::init_augmented_lagrangian");
    PROFILE_START();

    int ndof = PoseD::dim_to_ndof(dim());
    // int pos_ndof = PoseD::dim_to_pos_ndof(dim());
    // int rot_ndof = PoseD::dim_to_rot_ndof(dim());

    augmented_lagrangian_penalty = 1e3;
    size_t num_kinematic_bodies = m_assembler.count_kinematic_bodies();
    augmented_lagrangian_multiplier.setZero(ndof * num_kinematic_bodies);

    for (int i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::KINEMATIC
            && m_assembler[i].kinematic_max_time < 0) {
            m_assembler[i].convert_to_static();
        }
    }

    x_pred = x0;
    for (int i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].kinematic_poses.size()) {
            const PoseD& pose = m_assembler[i].kinematic_poses.front();
            // Kinematic position
            x_pred.segment(ndof * i, ndof) = pose.dof();
            // // Kinematic rotation
            // x_pred.segment(ndof * i + pos_ndof, rot_ndof) = pose.rotation;
        } else {
            // Kinematic position
            x_pred.segment(ndof * i, ndof) +=
                timestep() * m_assembler[i].velocity.dof();
            // // Kinematic rotation
            // x_pred.segment(ndof * i + pos_ndof, rot_ndof) +=
            //     timestep() * m_assembler[i].velocity.rotation;
        }
    }

    is_dof_satisfied.setZero(x0.size());
    for (int i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::STATIC) {
            is_dof_satisfied.segment(ndof * i, ndof).setOnes();
        }
    }
    PROFILE_END();
}

void DistanceBarrierRBProblem::step_kinematic_bodies()
{
    PROFILE_POINT("DistanceBarrierRBProblem::step_kinematic_bodies");
    PROFILE_START();

    for (int i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::KINEMATIC) {
            if (m_assembler[i].kinematic_max_time < 0) {
                m_assembler[i].convert_to_static();
            } else {
                m_assembler[i].kinematic_max_time -= timestep();
                if (m_assembler[i].kinematic_poses.size()) {
                    m_assembler[i].kinematic_poses.pop_front();
                }
            }
        }
    }
    PROFILE_END();
}
// @javidf: compute_J(), compute_Jinv(), and computeJsqrt() are not needed for
// affine body dynamic. Can be deleted.
inline DiagonalMatrix3d compute_J(const VectorMax3d& I)
{
    return DiagonalMatrix3d(
        0.5 * (-I.x() + I.y() + I.z()), //
        0.5 * (I.x() - I.y() + I.z()),  //
        0.5 * (I.x() + I.y() - I.z()));
}

inline DiagonalMatrix3d compute_Jinv(const VectorMax3d& I)
{
    return DiagonalMatrix3d(
        2 / (-I.x() + I.y() + I.z()), //
        2 / (I.x() - I.y() + I.z()),  //
        2 / (I.x() + I.y() - I.z()));
}

inline DiagonalMatrix3d compute_Jsqrt(const VectorMax3d& I)
{
    assert(0.5 * (-I.x() + I.y() + I.z()) >= 0);
    assert(0.5 * (I.x() - I.y() + I.z()) >= 0);
    assert(0.5 * (I.x() + I.y() - I.z()) >= 0);
    return DiagonalMatrix3d(
        sqrt(std::max(0.5 * (-I.x() + I.y() + I.z()), 0.0)),
        sqrt(std::max(0.5 * (I.x() - I.y() + I.z()), 0.0)),
        sqrt(std::max(0.5 * (I.x() + I.y() - I.z()), 0.0)));
}

// @javidf: this should replace compute_linear_augment_lagrangian_progress() and
// compute_angular_augment_lagrangian_progress(). They all exist now.
double DistanceBarrierRBProblem::compute_augment_lagrangian_progress(
    const Eigen::VectorXd& x) const
{
    int ndof = PoseD::dim_to_ndof(dim());
    // int pos_ndof = PoseD::dim_to_pos_ndof(dim());

    double a = 0, b = 0;

    for (size_t i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::KINEMATIC) {
            a += (x_pred.segment(ndof * i, ndof) - x.segment(ndof * i, ndof))
                     .squaredNorm();
            b += (x_pred.segment(ndof * i, ndof) - x0.segment(ndof * i, ndof))
                     .squaredNorm();
        }
    }

    if (a == 0 && b == 0) {
        return 1;
    }
    return 1 - sqrt(a / (b == 0 ? 1 : b));
}

// @javidf: This method is updated for ABD.
void DistanceBarrierRBProblem::update_augmented_lagrangian(
    const Eigen::VectorXd& x)
{
    // @jaivdf: Check the dimension to make sure they are correctly updated.
    // ndof = 12 if dim=3
    int ndof = PoseD::dim_to_ndof(dim());
    // int pos_ndof = PoseD::dim_to_pos_ndof(dim());
    // int rot_ndof = PoseD::dim_to_rot_ndof(dim());

    // double eta_q = compute_linear_augment_lagrangian_progress(x);
    // double eta_Q = compute_angular_augment_lagrangian_progress(x);
    double eta = compute_augment_lagrangian_progress(x);

    if (eta >= 0.999) {
        // Fix the kinematic DoF that have converged
        for (size_t i = 0; i < num_bodies(); i++) {
            if (m_assembler[i].type == RigidBodyType::KINEMATIC) {
                is_dof_satisfied.segment(ndof * i, ndof).setOnes();
            }
        }
    } else if (eta < 0.99 && augmented_lagrangian_penalty < 1e8) {
        // Increase the κ_q
        augmented_lagrangian_penalty *= 2;
    } else {
        // Increase the λ
        for (size_t i = 0, ki = 0; i < num_bodies(); i++) {
            if (m_assembler[i].type == RigidBodyType::KINEMATIC) {
                augmented_lagrangian_multiplier.segment(ki * ndof, ndof) -=
                    linear_augmented_lagrangian_penalty
                    * sqrt(m_assembler[i].mass)
                    * (x.segment(ndof * i, ndof)
                       - x_pred.segment(ndof * i, ndof));
                ki++;
            }
        }
    }

    if (eta < 0.999) {
        spdlog::info(
            "updated augmented Lagrangian "
            "κ_q={:g} κ_Q={:g} ||λ||∞={:g} ||Λ||∞={:g} η_q={:g} η_Q={:g}",
            augmented_lagrangian_penalty,
            augmented_lagrangian_multiplier.lpNorm<Eigen::Infinity>(), eta);
    }
}

bool DistanceBarrierRBProblem::are_equality_constraints_satisfied(
    const Eigen::VectorXd& x) const
{
    if (m_assembler.count_kinematic_bodies()) {
        return compute_augment_lagrangian_progress(x) >= 0.999;
        // return compute_linear_augment_lagrangian_progress(x) >= 0.999
        //     && compute_angular_augment_lagrangian_progress(x) >= 0.999;
    }
    return true;
}

OptimizationResults DistanceBarrierRBProblem::solve_constraints()
{
    PROFILE_POINT("DistanceBarrierRBProblem::solve_constraints");
    PROFILE_START();

    OptimizationResults opt_result;
    opt_result.x = starting_point();
    double momentum_balance, eps_d = 1e-2 * world_bbox_diagonal();
    int i = 0;
    int total_newton_iterations = 0;
    // int count = 0;
    do {
        // std::cout << "do count : " << count << std::endl;
        opt_result = solver().solve(opt_result.x);
        total_newton_iterations += opt_result.num_iterations;
        if (!opt_result.success) {
            break;
        }

        PosesD poses = this->dofs_to_poses(opt_result.x);

        CollisionConstraints collision_constraints;
        m_constraint.construct_constraint_set(
            m_collision_mesh, m_assembler, poses, collision_constraints);
        update_friction_constraints(collision_constraints, poses);

        Eigen::VectorXd grad_Ex, grad_Bx, grad_Dx;
        compute_energy_term(opt_result.x, grad_Ex);
        compute_barrier_term(opt_result.x, collision_constraints, grad_Bx);
        compute_friction_term(opt_result.x, grad_Dx);

        momentum_balance = compute_momentum_balance(grad_Ex, grad_Bx, grad_Dx);

        i++;

        spdlog::info(
            "friction_solve lagging_iteration={:d} momentum_balance={:g} "
            "eps_d={:g}",
            i, momentum_balance, eps_d);
    } while ((friction_iterations < 0 || i < friction_iterations)
             && momentum_balance > eps_d);

    if (opt_result.success) {
        spdlog::info(
            "Finished friction solve after {:d} lagging iteration(s) and a "
            "momentum balance error of {:g}",
            i, momentum_balance);
    } else {
        spdlog::error(
            "Ending friction solve early because newton solve {:d} failed!", i);
    }

    opt_result.num_iterations = total_newton_iterations;

    PROFILE_END();

    return opt_result;
}

bool DistanceBarrierRBProblem::take_step(const Eigen::VectorXd& x)
{
    PROFILE_POINT("DistanceBarrierRBProblem::take_step");
    PROFILE_START();

    min_distance = compute_min_distance(x);
    if (min_distance < 0) {
        spdlog::info("final_step min_distance=N/A");
    } else {
        spdlog::info("final_step min_distance={:.8e}", min_distance);
    }

    const double h = timestep();

    // update final pose
    // -------------------------------------
    m_assembler.set_rb_poses(this->dofs_to_poses(x));
    PosesD poses_q1 = m_assembler.rb_poses_t1();

    // Update the velocities and accelerations
    // This need to be done AFTER updating poses
    for (RigidBody& rb : m_assembler.m_rbs) {
        if (rb.type != RigidBodyType::DYNAMIC) {
            continue;
        }

        switch (body_energy_integration_method) {
        case IMPLICIT_EULER:
            rb.velocity = (rb.pose - rb.pose_prev) / h;
            break;

        case IMPLICIT_NEWMARK:
        case STABILIZED_NEWMARK: {
            double c = 1 / (h * h * newmark_beta);
            PoseD acc_new = c
                * (rb.pose - rb.pose_prev - h * rb.velocity_prev
                   - h * h * (0.5 - newmark_beta) * rb.acceleration);
            rb.velocity = rb.velocity_prev
                + h
                    * ((1 - newmark_gamma) * rb.acceleration
                       + newmark_gamma * acc_new);
            rb.acceleration = acc_new;
            break;
        }
        }
    }

    if (do_intersection_check) {
        // Check for intersections instead of collision along the entire
        // step. We only guarentee a piecewise collision-free trajectory.
        // return detect_collisions(poses_t0, poses_q1,
        // CollisionCheck::EXACT);
        PROFILE_END();
        return detect_intersections(poses_q1);
    }
    PROFILE_END();
    return false;
}

////////////////////////////////////////////////////////////
// Barrier Problem

// Compute the objective function:
// f(x) = E(x) + κ ∑_{k ∈ C} b(d(x_k)) + ∑_{k ∈ C} D(x_k)
double DistanceBarrierRBProblem::compute_objective(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    PROFILE_POINT("DistanceBarrierRBProblem::compute_objective");
    PROFILE_START();

    // Compute rigid body energy term
    double Ex = compute_energy_term(x, grad, hess, compute_grad, compute_hess);
    Ex /= average_mass();
    if (compute_grad) {
        grad /= average_mass();
    }
    if (compute_hess) {
        hess /= average_mass();
    }

    Eigen::VectorXd grad_AL;
    Eigen::SparseMatrix<double> hess_AL;
    double ALx = compute_augmented_lagrangian(
        x, grad_AL, hess_AL, compute_grad, compute_hess);
    Ex += ALx / average_mass();
    if (compute_grad) {
        grad += grad_AL / average_mass();
    }
    if (compute_hess) {
        hess += hess_AL / average_mass();
    }

    // The following is used to disable constraints if desired
    // (useful for testing).
    if (!m_use_barriers) {
        PROFILE_END();
        return Ex;
    }

    // Compute a common constraint set to use for contacts and friction
    // Start by updating the constraint set
    CollisionConstraints constraints;
    m_constraint.construct_constraint_set(
        m_collision_mesh, m_assembler, this->dofs_to_poses(x), constraints);

    spdlog::debug(
        "problem={} num_vertex_vertex_constraint={:d} "
        "num_edge_vertex_constraints={:d} num_edge_edge_constraints={:d} "
        "num_face_vertex_constraints={:d}",
        name(), constraints.vv_constraints.size(),
        constraints.ev_constraints.size(), constraints.ee_constraints.size(),
        constraints.fv_constraints.size());

    Eigen::VectorXd grad_Bx;
    Eigen::SparseMatrix<double> hess_Bx;
    double Bx = compute_barrier_term(
        x, constraints, grad_Bx, hess_Bx, compute_grad, compute_hess);

    // D(x) is the friction potential (Equation 15 in the IPC paper)
    Eigen::VectorXd grad_Dx;
    Eigen::SparseMatrix<double> hess_Dx;
    double Dx =
        compute_friction_term(x, grad_Dx, hess_Dx, compute_grad, compute_hess);

    // // D(x) = h^2 * ΣD_i(x)
    // double h2 = timestep() * timestep();
    // Dx *= h2;
    // grad_Dx *= h2;
    // hess_Dx *= h2;

    // Sum all the potentials
    double kappa_over_avg_mass = barrier_stiffness() / average_mass();

    // Timestepping coeff
    double coeff = 1.0;
    if (body_energy_integration_method == STABILIZED_NEWMARK
        || body_energy_integration_method == IMPLICIT_NEWMARK) {
        coeff = newmark_beta;
    }

    if (compute_grad) {
        grad += coeff * kappa_over_avg_mass * grad_Bx
            + coeff * grad_Dx / average_mass();
    }
    if (compute_hess) {
        hess += coeff * kappa_over_avg_mass * hess_Bx
            + coeff * hess_Dx / average_mass();
    }

    PROFILE_END();

    return Ex + coeff * kappa_over_avg_mass * Bx + coeff * Dx / average_mass();
}

// Compute E(x) in f(x) = E(x) + κ ∑_{k ∈ C} b(d(x_k))
double DistanceBarrierRBProblem::compute_energy_term(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    PROFILE_POINT("DistanceBarrierRBProblem::compute_energy_term");
    PROFILE_START();

    // typedef AutodiffType<Eigen::Dynamic, /*maxN=*/6> Diff;

    int ndof = PoseD::dim_to_ndof(dim());
    int pos_ndof = PoseD::dim_to_pos_ndof(dim());
    int trans_ndof = PoseD::dim_to_trans_ndof(dim());
    // std::cout << "before energy initiation \n";
    Eigen::VectorXd energies = Eigen::VectorXd::Zero(num_bodies());
    if (compute_grad) {
        grad.setZero(x.size());
    }
    tbb::concurrent_vector<Eigen::Triplet<double>> hess_triplets;
    if (compute_hess) {
        // Hessian is a block diagonal with (ndof x ndof) blocks
        hess_triplets.reserve(num_bodies() * ndof * ndof);
    }

    const std::vector<PoseD> poses = this->dofs_to_poses(x);
    assert(poses.size() == num_bodies());

    // tbb::parallel_for(size_t(0), poses.size(), [&](size_t i) {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), poses.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            // Activate autodiff with the correct number of variables
            // Diff::activate(ndof);
            for (long i = range.begin(); i != range.end(); ++i) {
                const PoseD& pose = poses[i];
                const RigidBody& body = m_assembler[i];

                // Do not compute the body energy for static and kinematic
                // bodies
                if (body.type != RigidBodyType::DYNAMIC) {
                    continue;
                }
                energies[i] = compute_body_energy<double>(body, pose);
                VectorMax12d grad_i;
                if (compute_grad) {
                    grad_i = compute_body_energy_diff<double>(body, pose);
                    grad.segment(i * ndof, ndof) = grad_i;
                }
                MatrixMax12d hess_i;
                if (compute_hess) {
                    hess_i = compute_body_energy_hess<double>(body, pose);
                    for (int r = 0; r < hess_i.rows(); r++) {
                        for (int c = 0; c < hess_i.cols(); c++) {
                            hess_triplets.emplace_back(
                                i * ndof + r, i * ndof + c, hess_i(r, c));
                        }
                    }
                }
            }
        });

    if (compute_hess) {
        // NAMED_PROFILE_POINT(
        //    "DistanceBarrierRBProblem::compute_energy_term:"
        //    "assemble_hessian",
        //    ASSEMBLE_ENERGY_HESS);
        // PROFILE_START(ASSEMBLE_ENERGY_HESS);

        // ∇²E: Rⁿ ↦ Rⁿˣⁿ
        hess.resize(x.size(), x.size());
        hess.setFromTriplets(hess_triplets.begin(), hess_triplets.end());

        // PROFILE_END(ASSEMBLE_ENERGY_HESS);
    }

#ifdef ABD_WITH_DERIVATIVE_CHECK
    if (!is_checking_derivative) {
        is_checking_derivative = true;
        // A large mass (e.g., from a large ground plane) can affect the
        // accuracy.
        double mass_Linf =
            m_assembler.m_rb_mass_matrix.diagonal().lpNorm<Eigen::Infinity>();
        double tol = std::max(1e-8 * mass_Linf, 1e-4);
        if (compute_grad) {
            Eigen::VectorXd grad_approx = eval_grad_energy_approx(*this, x);
            if (!fd::compare_gradient(grad, grad_approx, tol)) {
                spdlog::error("finite gradient check failed for E(x)");
            }
        }
        if (compute_hess) {
            Eigen::MatrixXd hess_approx = eval_hess_energy_approx(*this, x);
            if (!fd::compare_jacobian(hess, hess_approx, tol)) {
                spdlog::error("finite hessian check failed for E(x)");
            }
        }
        is_checking_derivative = false;
    }
#endif
    PROFILE_END();
    return energies.sum();
}

// @javidf: theses are the added methods. Most of them are supposed to replace
// old rigidIPC methods but the old ones are kept for now. Double check them.
// PLEASE MAKE SURE CALCULATIONS ARE CORRECT. Compute the energy term for a
// single rigid body
template <typename T>
T DistanceBarrierRBProblem::compute_body_energy(
    const RigidBody& body, const Pose<T>& pose)
{
    int ndof = PoseD::dim_to_ndof(dim());
    double h = timestep();

    T energy(0.0);

    energy += compute_inertial_potential(body, pose);

    switch (body_energy_integration_method) {
    case IMPLICIT_EULER:
        energy += h * h * compute_orthogonal_penalty(body, pose);
        break;
    case IMPLICIT_NEWMARK:
    case STABILIZED_NEWMARK:
        energy += newmark_beta * h * h * compute_orthogonal_penalty(body, pose);
        break;
    }

    return energy;
}

// Compute the energy term for a single rigid body
template <typename T>
VectorMax12<T> DistanceBarrierRBProblem::compute_body_energy_diff(
    const RigidBody& body, const Pose<T>& pose)
{
    int ndof = PoseD::dim_to_ndof(dim());
    double h = timestep();

    VectorMax12<T> energy_diff;
    energy_diff.resize(ndof, 1);
    energy_diff.setZero();

    energy_diff += compute_inertial_potential_diff<double>(body, pose);

    VectorMax12d VO_diff = compute_orthogonal_penalty_diff<double>(body, pose);
    switch (body_energy_integration_method) {
    case IMPLICIT_EULER:
        energy_diff += h * h * VO_diff;
        break;
    case IMPLICIT_NEWMARK:
    case STABILIZED_NEWMARK:
        energy_diff += newmark_beta * h * h * VO_diff;
        break;
    }

    return energy_diff;
}

// Compute the energy term for a single rigid body
template <typename T>
MatrixMax12<T> DistanceBarrierRBProblem::compute_body_energy_hess(
    const RigidBody& body, const Pose<T>& pose)
{
    int ndof = PoseD::dim_to_ndof(dim());
    double h = timestep();

    MatrixMax12<T> body_energy_hess;
    body_energy_hess.resize(ndof, ndof);
    body_energy_hess.setZero();

    body_energy_hess += compute_inertial_potential_hess(body, pose);

    switch (body_energy_integration_method) {
    case IMPLICIT_EULER:
        body_energy_hess += h * h * compute_orthogonal_penalty_hess(body, pose);
        break;
    case IMPLICIT_NEWMARK:
    case STABILIZED_NEWMARK:
        body_energy_hess +=
            newmark_beta * h * h * compute_orthogonal_penalty_hess(body, pose);
        break;
    }

    return body_energy_hess;
}

// Compute the inertial term in the body energy for ABD
template <typename T>
T DistanceBarrierRBProblem::compute_inertial_potential(
    const RigidBody& body, const Pose<T>& pose)
{
    int ndof = PoseD::dim_to_ndof(dim());
    // NOTE: t0 suffix indicates the current value not the inital value
    double h = timestep();
    T energy(0.0);
    // if (!body.is_dof_fixed.head(pose.pos_ndof()).all()) {
    // make sure there is a dof() for all pose velocity force etc.
    VectorMax12<T> q = pose.dof();
    const VectorMax12d& q_t0 = body.pose.dof();
    const VectorMax12d& qdot_t0 = body.velocity.dof();
    VectorMax12d ext_gravity; // = Vector12d::Zero();
    ext_gravity.resize(ndof, 1);
    ext_gravity.setZero();
    ext_gravity.head(dim()) = gravity.array();

    //@javidf: force here can be calculated as J.T * f as mentioned in Eq. 5 of
    // ABD paper.
    VectorMax12<T> force = body.force.dof();
    // Adding the follower force (force dependant on the position / orientation
    // of the body)
    VectorMax12<T> follower_force;
    follower_force = body.follower_force_jacobian * q_t0;
    force += follower_force;

    VectorMax12d qddot;
    switch (body_energy_integration_method) {
    case IMPLICIT_EULER:
        qddot = ext_gravity + body.mass_matrix.inverse() * force;
        break;
    case IMPLICIT_NEWMARK:
    case STABILIZED_NEWMARK:
        qddot = (0.5 - newmark_beta) * body.acceleration.dof()
            + newmark_beta * (ext_gravity + body.mass_matrix.inverse() * force);
        break;
    }
    const VectorMax12<double>& q_bar = q_t0 + h * qdot_t0 + h * h * qddot;

    // 0.5 ||q_b - q^bar_b||_M
    return 0.5 * (q - q_bar).dot(body.mass_matrix * (q - q_bar));
}

// Compute the gradient of the inertial term in the body energy for ABD
template <typename T>
VectorMax12<T> DistanceBarrierRBProblem::compute_inertial_potential_diff(
    const RigidBody& body, const Pose<T>& pose)
{
    int ndof = PoseD::dim_to_ndof(dim());
    double h = timestep();

    VectorMax12d q = pose.dof();
    const VectorMax12d& q_t0 = body.pose.dof();
    const VectorMax12d& qdot_t0 = body.velocity.dof();
    VectorMax12d ext_gravity;
    ext_gravity.resize(ndof, 1);
    ext_gravity.setZero();
    ext_gravity.head(body.dim()) = gravity;

    //@javidf: force here can be calculated as J.T * f as mentioned in Eq. 5 of
    // ABD paper.
    VectorMax12<T> force = body.force.dof();
    // Adding the follower force (force dependant on the position / orientation
    // of the body)
    VectorMax12<T> follower_force;
    follower_force = body.follower_force_jacobian * q_t0;
    force += follower_force;

    // std::cout << "Force:\n" << force << std::endl;
    // std::cout << std::endl;

    VectorMax12d qddot;
    switch (body_energy_integration_method) {
    case IMPLICIT_EULER:
        qddot = ext_gravity + body.mass_matrix.inverse() * force;
        break;
    case IMPLICIT_NEWMARK:
    case STABILIZED_NEWMARK:
        qddot = (0.5 - newmark_beta) * body.acceleration.dof()
            + newmark_beta * (ext_gravity + body.mass_matrix.inverse() * force);
        break;
    }

    VectorMax12<double> q_bar = q_t0 + h * qdot_t0 + h * h * qddot;
    // std::cout << "q_bar:\n" << q_bar << std::endl;
    // std::cout << std::endl;

    // 0.5 ||q_b - q^bar_b||_M
    return body.mass_matrix * (q - q_bar);
}

// Compute the hessian of the inertial term in the body energy for ABD
template <typename T>
MatrixMax12<T> DistanceBarrierRBProblem::compute_inertial_potential_hess(
    const RigidBody& body, const Pose<T>& pose)
{
    return body.mass_matrix;
}

// Compute the orthogonality penalty for ABD
template <typename T>
T DistanceBarrierRBProblem::compute_orthogonal_penalty(
    const RigidBody& body, const Pose<T>& pose)
{
    // NOTE: t0 suffix indicates the current value not the inital value
    // double h = timestep();
    T ortho_penalty(0.0);
    // V_ortogonality = k * vol * (||AA^T - I||_F)^2
    MatrixMax3d QQT = pose.transform * pose.transform.transpose();
    QQT.diagonal().array() -= 1.0;
    ortho_penalty = body.volume * orthogonal_stiffness() * QQT.squaredNorm();
    return ortho_penalty;
}

// Compute the orthogonality penalty for ABD
template <typename T>
VectorMax12<T> DistanceBarrierRBProblem::compute_orthogonal_penalty_diff(
    const RigidBody& body, const Pose<T>& pose) const
{
    int ndof = PoseD::dim_to_ndof(dim());
    // NOTE: t0 suffix indicates the current value not the inital value
    double h = timestep();

    VectorMax12<T> ortho_penalty_diff;
    ortho_penalty_diff.resize(ndof, 1);
    ortho_penalty_diff.setZero();

    MatrixMax3<double> Q = pose.transform;
    VectorMax3d a1 = Q.row(0).transpose();
    VectorMax3d a2 = Q.row(1).transpose();

    double a1a1 = a1.dot(a1);
    double a2a2 = a2.dot(a2);
    double a1a2 = a1.dot(a2);
    if (dim() == 3) {
        VectorMax3d a3 = Q.row(2).transpose();
        double a3a3 = a3.dot(a3);
        double a2a3 = a2.dot(a3);
        double a1a3 = a1.dot(a3);
        ortho_penalty_diff.segment(dim(), dim()) +=
            (a1a1 - 1.0) * a1 + a1a2 * a2 + a1a3 * a3;
        ortho_penalty_diff.segment(2 * dim(), dim()) +=
            (a2a2 - 1.0) * a2 + a1a2 * a1 + a2a3 * a3;
        ortho_penalty_diff.segment(3 * dim(), dim()) +=
            (a3a3 - 1.0) * a3 + a2a3 * a2 + a1a3 * a1;
    } else {
        ortho_penalty_diff.segment(dim(), dim()) +=
            (a1a1 - 1.0) * a1 + a1a2 * a2;
        ortho_penalty_diff.segment(2 * dim(), dim()) +=
            (a2a2 - 1.0) * a2 + a1a2 * a1;
    }
    ortho_penalty_diff *= 4 * body.volume * orthogonal_stiffness();
    return ortho_penalty_diff;
}

// Compute the orthogonality penalty for ABD
template <typename T>
MatrixMax12<T> DistanceBarrierRBProblem::compute_orthogonal_penalty_hess(
    const RigidBody& body, const Pose<T>& pose) const
{
    int ndof = PoseD::dim_to_ndof(dim());
    // NOTE: t0 suffix indicates the current value not the inital value
    double h = timestep();

    MatrixMax12<T> ortho_penalty_hess;
    ortho_penalty_hess.resize(ndof, ndof);
    ortho_penalty_hess.setZero();
    MatrixMax3d eye;
    eye.resize(dim(), dim());
    eye.setZero();
    eye.diagonal().array() = 1.0;
    MatrixMax3d Q = pose.transform;
    MatrixMax3d QQTI = Q * Q.transpose();
    QQTI.diagonal().array() -= 1.0;

    size_t n_blocks = dim() * (dim() + 1) / 2;
    std::vector<std::pair<size_t, size_t>> pairs(n_blocks);
    if (dim() == 3)
        pairs = { { 0, 0 }, { 1, 1 }, { 2, 2 }, { 0, 1 }, { 0, 2 }, { 1, 2 } };
    else
        pairs = { { 0, 0 }, { 1, 1 }, { 0, 1 } };

    std::vector<MatrixMax3d> outers;
    // outers.resize(n_blocks);
    for (size_t nb = 0; nb < n_blocks; nb++) {
        size_t n_i = pairs.at(nb).first;
        size_t n_j = pairs.at(nb).second;
        outers.push_back(Q.row(n_i).transpose() * Q.row(n_j));
    }
    // diagonal
    for (size_t n_i = 0; n_i < dim(); n_i++) {
        size_t n_p = (size_t)n_i * dim() + dim();
        size_t n_q = n_p;
        size_t n_j = (n_i + 1) % dim();
        ortho_penalty_hess.block(n_p, n_q, dim(), dim()) +=
            2.0 * outers.at(n_i) + QQTI(n_i, n_i) * eye + outers.at(n_j);
        if (dim() == 3) {
            size_t n_k = (n_i + 2) % dim();
            ortho_penalty_hess.block(n_p, n_q, dim(), dim()) += outers.at(n_k);
        }
    }
    // off diagonal
    for (size_t n_b = dim(); n_b < n_blocks; n_b++) {
        size_t n_p = (size_t)pairs.at(n_b).first * dim() + dim();
        size_t n_q = (size_t)pairs.at(n_b).second * dim() + dim();
        ortho_penalty_hess.block(n_p, n_q, dim(), dim()) +=
            QQTI(pairs.at(n_b).first, pairs.at(n_b).second) * eye
            + outers.at(n_b).transpose();
        ortho_penalty_hess.block(n_q, n_p, dim(), dim()) +=
            ortho_penalty_hess.block(n_p, n_q, dim(), dim()).transpose();
        // if (dim()==3)
        // {
        //     size_t n_k = (n_i + 2) % dim();
        //     ortho_penalty_hess.block(n_p, n_q, dim(), dim())
        //     += 2.0*outers.at(n_k);
        // }
    }

    // for (size_t ni=0; ni<Q.rows(); ni++)
    // {
    //     for (size_t nj=0; nj<Q.rows(); nj++)
    //     {
    //         size_t n_k = (size_t) ni*dim()+dim();
    //         size_t n_l = (size_t) nj*dim()+dim();
    //         if (ni == nj)
    //         {
    //             size_t n_jj = (size_t) (ni+1)%dim();
    //             ortho_penalty_hess.block(n_k, n_l, dim(), dim()) +=
    //                 4.0*Q.row(ni).transpose()*Q.row(ni)
    //                 + 2.0*(Q.row(ni).squaredNorm() - 1.0)*eye
    //                 + 2.0*(Q.row(n_jj).transpose()*Q.row(n_jj));
    //             if (dim() == 3)
    //             {
    //                 size_t n_kk = (size_t) (ni+2)%dim();
    //                 ortho_penalty_hess.block(n_k, n_l, dim(), dim()) +=
    //                     2.0*(Q.row(n_kk).transpose()*Q.row(n_kk));
    //             }
    //         }
    //         else{
    //             size_t n_jj = (size_t) (ni+1)%dim();
    //             size_t n_kk = (size_t) (ni+2)%dim();
    //             if (dim() == 3)
    //                 ortho_penalty_hess.block(n_k, n_l, dim(), dim()) +=
    //                     2.0*(Q.row(n_jj).transpose()*Q.row(n_jj))
    //                     + 2.0*(Q.row(n_kk).transpose()*Q.row(n_kk));
    //             else
    //                 ortho_penalty_hess.block(n_k, n_l, dim(), dim()) +=
    //                     2.0*(Q.row(n_jj).transpose()*Q.row(n_jj));
    //         }
    //     }
    // }
    ortho_penalty_hess *= 4.0 * body.volume * orthogonal_stiffness();
    return ortho_penalty_hess;
}

double DistanceBarrierRBProblem::compute_augmented_lagrangian(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    PROFILE_POINT("DistanceBarrierRBProblem::compute_augmented_lagrangian");
    PROFILE_START();

    int ndof = PoseD::dim_to_ndof(dim());
    int pos_ndof = PoseD::dim_to_pos_ndof(dim());
    int rot_ndof = PoseD::dim_to_trans_ndof(dim());
    size_t num_kinematic_bodies = m_assembler.count_kinematic_bodies();

    double potential = 0;
    if (compute_grad) {
        grad.setZero(x.size());
    }
    std::vector<Eigen::Triplet<double>> hess_triplets;
    if (compute_hess) {
        hess.resize(x.size(), x.size());
        hess_triplets.reserve(num_kinematic_bodies * ndof);
    }

    bool all_kinematic_dof_satisfied = true;
    for (size_t i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::KINEMATIC
            && !is_dof_satisfied.segment(ndof * i, ndof).all()) {
            all_kinematic_dof_satisfied = false;
            break;
        }
    }

    if (all_kinematic_dof_satisfied) {
        PROFILE_END();
        return potential;
    }

    const double& kappa = augmented_lagrangian_penalty;
    for (size_t i = 0, ki = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type != RigidBodyType::KINEMATIC) {
            continue;
        }

        double m = m_assembler[i].mass;
        const auto& lambda =
            augmented_lagrangian_multiplier.segment(ki * pos_ndof, pos_ndof);

        const auto& q = x.segment(i * ndof, pos_ndof);
        const auto& q_pred = x_pred.segment(i * ndof, pos_ndof);

        potential += kappa / 2 * m * (q - q_pred).squaredNorm()
            - sqrt(m) * lambda.dot(q - q_pred);
        if (compute_grad) {
            grad.segment(i * ndof, pos_ndof) =
                kappa * m * (q - q_pred) - sqrt(m) * lambda;
        }
        if (compute_hess) {
            for (int j = 0; j < pos_ndof; j++) {
                hess_triplets.emplace_back(
                    ndof * i + j, ndof * i + j, kappa * m);
            }
        }

        ki++;
    }

    if (compute_hess) {
        // NAMED_PROFILE_POINT(
        //    "DistanceBarrierRBProblem::compute_augmented_lagrangian:"
        //    "assemble_hessian",
        //    ASSEMBLE_AL_HESS);
        // PROFILE_START(ASSEMBLE_AL_HESS);

        hess.setFromTriplets(hess_triplets.begin(), hess_triplets.end());

        // PROFILE_END(ASSEMBLE_AL_HESS);
    }

#ifdef ABD_WITH_DERIVATIVE_CHECK
    if (!is_checking_derivative) {
        is_checking_derivative = true;
        if (compute_grad) {
            check_augmented_lagrangian_gradient(x, grad);
        }
        if (compute_hess) {
            check_augmented_lagrangian_hessian(x, hess);
        }
        is_checking_derivative = false;
    }
#endif

    PROFILE_END();

    return potential;
}

// void DistanceBarrierRBProblem::to_linear_constraint_space(
//     Eigen::Ref<Eigen::VectorXd> grad, Eigen::SparseMatrix<double>& hess)
// {
//     this->m_linear_cons.grad_to_constraint_space(grad);
//     this->m_linear_cons.hess_to_constraint_space(hess);
// }

// Eigen::VectorXd
// DistanceBarrierRBProblem::from_linear_constraint_space(const Eigen::VectorXd& z)
// {
//     Eigen::VectorXd x = z;
//     this->m_linear_cons.constraint_dofs_to_affine(x);
//     return x;
// }

// Eigen::VectorXd
// DistanceBarrierRBProblem::to_linear_constraint_space(const Eigen::VectorXd& x)
// {
//     Eigen::VectorXd z = x;
//     this->m_linear_cons.constraint_dofs_to_affine(z);
//     return z;
// }

// Eigen::VectorXi DistanceBarrierRBProblem::free_dofs_in_linear_constraint_space()
// {
//     return this->m_linear_cons.free_dof();
// }

Eigen::VectorXd
DistanceBarrierRBProblem::enforce_linear_constraint(Eigen::VectorXd x)
{
    return m_linear_cons.enforce_constraints(x);
}

Eigen::VectorXd
DistanceBarrierRBProblem::linear_constraint_error(Eigen::VectorXd x)
{
    return m_linear_cons.calc_error(x);
}

void DistanceBarrierRBProblem::compute_inertial_and_ortho_potentials(
    const RigidBody& body,
    const Pose<double>& pose,
    double& inertial_potential,
    double& ortho_potential,
    VectorMax12d& inertial_grad,
    VectorMax12d& ortho_grad,
    MatrixMax12d& inertial_hess,
    MatrixMax12d& ortho_hess)
{
    inertial_potential = compute_inertial_potential(body, pose);
    inertial_grad = compute_inertial_potential_diff(body, pose);
    inertial_hess = compute_inertial_potential_hess(body, pose);

    ortho_potential = compute_orthogonal_penalty(body, pose);
    ortho_grad = compute_orthogonal_penalty_diff(body, pose);
    ortho_hess = compute_orthogonal_penalty_hess(body, pose);
}

// Compute B(x) = ∑_{k ∈ C} b(d(x_k)) in f(x) = E(x) + κ ∑_{k ∈ C} b(d(x_k))
double DistanceBarrierRBProblem::compute_barrier_term(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    int& num_constraints,
    bool compute_grad,
    bool compute_hess)
{
    // Start by updating the constraint set
    PosesD poses = this->dofs_to_poses(x);
    CollisionConstraints constraints;
    m_constraint.construct_constraint_set(
        m_collision_mesh, m_assembler, poses, constraints);
    num_constraints = constraints.size();

    m_num_contacts = std::max(m_num_contacts, num_constraints);

    spdlog::debug(
        "problem={} num_vertex_vertex_constraint={:d} "
        "num_edge_vertex_constraints={:d} num_edge_edge_constraints={:d} "
        "num_face_vertex_constraints={:d}",
        name(), constraints.vv_constraints.size(),
        constraints.ev_constraints.size(), constraints.ee_constraints.size(),
        constraints.fv_constraints.size());

    double Bx = compute_barrier_term(
        x, constraints, grad, hess, compute_grad, compute_hess);

    return Bx;
}

// Convert from a local hessian to the triplets in the global hessian
template <typename DerivedLocalGradient>
void local_gradient_to_global(
    const Eigen::MatrixBase<DerivedLocalGradient>& local_gradient,
    const std::array<long, 2>& body_ids,
    int ndof,
    Eigen::VectorXd& grad)
{
    assert(local_gradient.size() == 2 * ndof);
    for (int b_i = 0; b_i < body_ids.size(); b_i++) {
        grad.segment(ndof * body_ids[b_i], ndof) +=
            local_gradient.segment(ndof * b_i, ndof);
    }
}

template <typename DerivedLocalHessian>
void local_hessian_to_global_triplets(
    const Eigen::MatrixBase<DerivedLocalHessian>& local_hessian,
    const std::array<long, 2>& body_ids,
    int ndof,
    std::vector<Eigen::Triplet<double>>& triplets)
{
    assert(local_hessian.rows() == 2 * ndof);
    assert(local_hessian.cols() == 2 * ndof);
    for (int b_i = 0; b_i < body_ids.size(); b_i++) {
        for (int b_j = 0; b_j < body_ids.size(); b_j++) {
            for (int dof_i = 0; dof_i < ndof; dof_i++) {
                for (int dof_j = 0; dof_j < ndof; dof_j++) {
                    double v =
                        local_hessian(ndof * b_i + dof_i, ndof * b_j + dof_j);
                    int r = ndof * body_ids[b_i] + dof_i;
                    int c = ndof * body_ids[b_j] + dof_j;
                    triplets.emplace_back(r, c, v);
                }
            }
        }
    }
}

// Apply the chain rule of f(V(x)) given ∇ᵥf(V) and ∇ₓV(x)
void apply_chain_rule(
    const VectorMax12d& grad_f,
    const Eigen::MatrixXd& jac_V,
    const MatrixMax12d& hess_f,
    const Eigen::MatrixXd& hess_V,
    const std::array<long, 4>& vertex_ids,
    const std::vector<uint8_t>& local_body_ids,
    const std::array<long, 2>& body_ids,
    const int dim,
    Eigen::VectorXd& grad,
    std::vector<Eigen::Triplet<double>>& hess_triplets,
    bool compute_grad,
    bool compute_hess)
{
    if (!compute_grad && !compute_hess) {
        return;
    }

    // PROFILE_POINT("apply_chain_rule");
    // PROFILE_START();
    std::vector<long> vert_ids;
    for (int i = 0; i < vertex_ids.size(); i++) {
        if (vertex_ids.at(i) >= 0) {
            vert_ids.resize(i + 1);
            vert_ids.at(i) = vertex_ids.at(i);
        }
    }
    // std::cout << "vert ids : " << vert_ids.size() << std::endl;
    // vertex_ids = std::remove(vertex_ids.begin(), vertex_ids.end(), -1);
    const int rb_ndof = PoseD::dim_to_ndof(dim);

    if (compute_grad) {
        // jac_Vi ∈ R^{4n × 2m}
        VectorMax24d local_grad = VectorMax24d::Zero(2 * rb_ndof);
        // VectorX<double> local_grad = VectorX<double>::Zero(24);
        for (int i = 0; i < vert_ids.size(); i++) {
            if (vert_ids[i] != -1) {
                local_grad.segment(rb_ndof * local_body_ids[i], rb_ndof) +=
                    jac_V.middleRows(vert_ids[i] * dim, dim).transpose()
                    * grad_f.segment(i * dim, dim);
            }
        }

        local_gradient_to_global(local_grad, body_ids, rb_ndof, grad);
    }

    if (compute_hess) {
        // jac_Vi ∈ R^{4n × 2m}
        MatrixMax24d jac_Vi =
            MatrixMax24d::Zero(vert_ids.size() * dim, 2 * rb_ndof);
        // MatrixX<double> jac_Vi = MatrixX<double>::Zero(vertex_ids.size() *
        // dim, 24);
        for (int i = 0; i < vert_ids.size(); i++) {
            if (vert_ids[i] != -1) {
                jac_Vi.block(
                    i * dim, local_body_ids[i] * rb_ndof, dim, rb_ndof) =
                    jac_V.middleRows(vert_ids[i] * dim, dim);
            }
        }

        // hess ∈ R^{2m × 2m}
        MatrixMax24d hess = jac_Vi.transpose() * hess_f * jac_Vi;
        // MatrixMax<double, 24, 24> hess = jac_Vi.transpose() * hess_f *
        // jac_Vi;

        // Skip the second term ∑(∂f/∂y_k ⋅ Hessian(V)_k) if Hessian(V) is all
        // zeros
        if (!hess_V.isZero()) {
            for (int i = 0; i < vert_ids.size(); i++) {
                if (vert_ids[i] != -1) {
                    for (int j = 0; j < dim; j++) {
                        // Off diagaonal blocks are all zero because the
                        // derivative of a vertex of body A with body B is zero.
                        hess.block(
                            local_body_ids[i] * rb_ndof,
                            local_body_ids[i] * rb_ndof, rb_ndof, rb_ndof) +=
                            hess_V.middleRows(
                                rb_ndof * (vert_ids[i] * dim + j), rb_ndof)
                            * grad_f[i * dim + j];
                    }
                }
            }
        }

        hess = project_to_psd(hess);

        local_hessian_to_global_triplets(
            hess, body_ids, rb_ndof, hess_triplets);
    }

    // PROFILE_END();
}

struct PotentialStorage {
    PotentialStorage() { }
    PotentialStorage(size_t nvars) { gradient.setZero(nvars); }
    double potential = 0;
    Eigen::VectorXd gradient;
    std::vector<Eigen::Triplet<double>> hessian_triplets;
};
typedef tbb::enumerable_thread_specific<PotentialStorage>
    ThreadSpecificPotentials;

double merge_derivative_storage(
    const ThreadSpecificPotentials& potentials,
    size_t nvars,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    // PROFILE_POINT("merge_derivative_storage");
    // PROFILE_START();

    if (compute_grad) {
        grad.setZero(nvars);
    }
    if (compute_hess) {
        hess.resize(nvars, nvars);
    }

    double potential = 0;
    for (const auto& p : potentials) {
        potential += p.potential;

        if (compute_grad) {
            grad += p.gradient;
        }

        if (compute_hess) {
            Eigen::SparseMatrix<double> p_hess(nvars, nvars);
            p_hess.setFromTriplets(
                p.hessian_triplets.begin(), p.hessian_triplets.end());
            hess += p_hess;
        }
    }

    // PROFILE_END();

    return potential;
}

double DistanceBarrierRBProblem::compute_barrier_term(
    const Eigen::VectorXd& x,
    const CollisionConstraints& constraints,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    if (constraints.size() == 0) {
        grad.setZero(x.size());
        hess.resize(x.size(), x.size());
        return 0;
    }

    PROFILE_POINT("DistanceBarrierRBProblem::compute_barrier_term");
    PROFILE_START();

    int rb_ndof = PoseD::dim_to_ndof(dim());

    // Compute V(x)
    Eigen::MatrixXd jac_V, hess_V;
    Eigen::MatrixXd V = m_assembler.world_vertices_diff(x, jac_V, compute_grad);

    double dhat = barrier_activation_distance();
    // double dmin = m_constraint.minimum_separation_distance;

    ThreadSpecificPotentials thread_storage(x.size());
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), constraints.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            // Get references to the local derivative storage
            auto& local_storage = thread_storage.local();
            auto& potential = local_storage.potential;
            auto& local_grad = local_storage.gradient;
            auto& hess_triplets = local_storage.hessian_triplets;

            for (size_t ci = range.begin(); ci != range.end(); ++ci) {
                const auto& constraint = constraints[ci];

                potential +=
                    constraint.compute_potential(V, edges(), faces(), dhat);

                VectorMax12d grad_B;
                if (compute_grad || compute_hess) {
                    grad_B = constraint.compute_potential_gradient(
                        V, edges(), faces(), dhat);
                }

                MatrixMax12d hess_B;
                if (compute_hess) {
                    hess_B = constraint.compute_potential_hessian(
                        V, edges(), faces(), dhat,
                        /*project_hessian_to_psd=*/false);
                }

                apply_chain_rule(
                    grad_B, jac_V, hess_B, hess_V,
                    constraint.vertex_ids(edges(), faces()),
                    vertex_local_body_ids(constraints, ci),
                    body_ids(m_assembler, constraints, ci), dim(), local_grad,
                    hess_triplets, compute_grad, compute_hess);
            }
        });

    double potential = merge_derivative_storage(
        thread_storage, x.size(), grad, hess, compute_grad, compute_hess);

#ifdef ABD_WITH_DERIVATIVE_CHECK
    if (!is_checking_derivative) {
        is_checking_derivative = true;
        if (compute_grad) {
            check_barrier_gradient(x, constraints, grad);
        }
        if (compute_hess) {
            check_barrier_hessian(x, constraints, hess);
        }
        is_checking_derivative = false;
    }
#endif

    PROFILE_END();

    return potential;
}

// NAMED_PROFILE_POINT(
//     "DistanceBarrierRBProblem::compute_friction_potential:value",
//     COMPUTE_FRICTION_VAL);
// NAMED_PROFILE_POINT(
//     "DistanceBarrierRBProblem::compute_friction_potential:gradient",
//     COMPUTE_FRICTION_GRAD);
// NAMED_PROFILE_POINT(
//     "DistanceBarrierRBProblem::compute_friction_potential:hessian",
//     COMPUTE_FRICTION_HESS);
template <typename RigidBodyConstraint, typename FrictionConstraint>
double DistanceBarrierRBProblem::compute_friction_potential(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXd& jac_V,
    const Eigen::MatrixXd& hess_V,
    const FrictionConstraint& constraint,
    Eigen::VectorXd& grad,
    std::vector<Eigen::Triplet<double>>& hess_triplets,
    bool compute_grad,
    bool compute_hess)
{
    // for each constraint:
    //     let m ∈ {6}, n ∈ {2, 3, 4}
    //     compute    V(x) ∈ R^{3n},
    //             ∇ₓ V(x) ∈ R^{3n × 2m},
    //             ∇ₓ²V(x) ∈ R^{3n × 2m × 2m}
    //     compute    D(V) ∈ R,
    //             ∇ᵥ D(V) ∈ R^{3n},
    //             ∇ᵥ²D(V) ∈ R^{3n × 3n}
    //     ∇ₓD(V(x)) = ∇ₓV(x)ᵀ∇ᵥD(V) ∈ R^{12x12}
    //     ∇ₓ²D(V(x)) = ∇ₓV(x)ᵀ∇ᵥ²D(V)∇ₓV(x) + ∑ᵢ ∇ₓᵢ²V * ∇ᵥD[i]
    //     local_to_global(∇ₓD(V(x)))
    //     local_to_global(project_to_psd(∇ₓ²D(V(x))))

    int rb_ndof = PoseD::dim_to_ndof(dim());

    double epsv_times_h = static_friction_speed_bound * timestep();

    // PROFILE_START(COMPUTE_FRICTION_VAL);
    double Dx = constraint.compute_potential(U, edges(), faces(), epsv_times_h);
    // PROFILE_END(COMPUTE_FRICTION_VAL);

    VectorMax12d grad_D;
    if (compute_grad || compute_hess) {
        // PROFILE_START(COMPUTE_FRICTION_GRAD);
        grad_D = constraint.compute_potential_gradient(
            U, edges(), faces(), epsv_times_h);
        // PROFILE_END(COMPUTE_FRICTION_GRAD);
    }

    MatrixMax12d hess_D;
    if (compute_hess) {
        // PROFILE_START(COMPUTE_FRICTION_HESS);
        hess_D = constraint.compute_potential_hessian(
            U, edges(), faces(), epsv_times_h,
            /*project_hessian_to_psd=*/false);
        // PROFILE_END(COMPUTE_FRICTION_HESS);
    }

    RigidBodyConstraint rbc(m_assembler, constraint);
    apply_chain_rule(
        grad_D, jac_V, hess_D, hess_V, constraint.vertex_ids(edges(), faces()),
        rbc.vertex_local_body_ids(), rbc.body_ids(), dim(), //
        grad, hess_triplets, compute_grad, compute_hess);

    return Dx;
}

double DistanceBarrierRBProblem::compute_friction_term(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    if (coefficient_friction <= 0 || friction_constraints.size() == 0) {
        grad.setZero(x.size());
        hess.resize(x.size(), x.size());
        return 0;
    }

    PROFILE_POINT("DistanceBarrierRBProblem::compute_friction_term");
    PROFILE_START();

    int rb_ndof = PoseD::dim_to_ndof(dim());

    // Compute V(x)
    Eigen::MatrixXd jac_V, hess_V;
    Eigen::MatrixXd V1 =
        m_assembler.world_vertices_diff(x, jac_V, compute_grad);

    // absolute linear dislacement of each point
    Eigen::MatrixXd U = V1 - m_assembler.world_vertices(poses_t0);

    ThreadSpecificPotentials thread_storage(x.size());
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), friction_constraints.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            // Get references to the local derivative storage
            auto& local_storage = thread_storage.local();
            auto& potential = local_storage.potential;
            auto& local_grad = local_storage.gradient;
            auto& hess_triplets = local_storage.hessian_triplets;

            for (size_t ci = range.begin(); ci != range.end(); ++ci) {
                size_t local_ci = ci;

                if (local_ci < friction_constraints.vv_constraints.size()) {
                    potential += compute_friction_potential<
                        RigidBodyVertexVertexConstraint>(
                        U, jac_V, hess_V,
                        friction_constraints.vv_constraints[local_ci],
                        local_grad, hess_triplets, compute_grad, compute_hess);
                    continue;
                }

                local_ci -= friction_constraints.vv_constraints.size();
                if (local_ci < friction_constraints.ev_constraints.size()) {
                    potential += compute_friction_potential<
                        RigidBodyEdgeVertexConstraint>(
                        U, jac_V, hess_V,
                        friction_constraints.ev_constraints[local_ci],
                        local_grad, hess_triplets, compute_grad, compute_hess);
                    continue;
                }

                local_ci -= friction_constraints.ev_constraints.size();
                if (local_ci < friction_constraints.ee_constraints.size()) {
                    potential +=
                        compute_friction_potential<RigidBodyEdgeEdgeConstraint>(
                            U, jac_V, hess_V,
                            friction_constraints.ee_constraints[local_ci],
                            local_grad, hess_triplets, compute_grad,
                            compute_hess);
                    continue;
                }

                local_ci -= friction_constraints.ee_constraints.size();
                assert(local_ci < friction_constraints.fv_constraints.size());
                potential +=
                    compute_friction_potential<RigidBodyFaceVertexConstraint>(
                        U, jac_V, hess_V,
                        friction_constraints.fv_constraints[local_ci],
                        local_grad, hess_triplets, compute_grad, compute_hess);
            }
        });

    double potential = merge_derivative_storage(
        thread_storage, x.size(), grad, hess, compute_grad, compute_hess);

#ifdef ABD_WITH_DERIVATIVE_CHECK
    if (!is_checking_derivative) {
        is_checking_derivative = true;
        if (compute_grad) {
            check_friction_gradient(x, grad);
        }
        if (compute_hess) {
            check_friction_hessian(x, hess);
        }
        is_checking_derivative = false;
    }
#endif

    PROFILE_END();
    return potential;
}

double DistanceBarrierRBProblem::compute_momentum_balance(
    const Eigen::VectorXd& grad_Ex,
    const Eigen::VectorXd& grad_Bx,
    const Eigen::VectorXd& grad_Dx)
{
    Eigen::VectorXd tmp;
    if (body_energy_integration_method == STABILIZED_NEWMARK
        || body_energy_integration_method == IMPLICIT_NEWMARK) {
        tmp = grad_Ex + newmark_beta * barrier_stiffness() * grad_Bx
            + newmark_beta * grad_Dx;
    } else {
        tmp = grad_Ex + barrier_stiffness() * grad_Bx + grad_Dx;
    }

    // If there are linear constraints,
    // We'll project the momentum to the unconstrained subspace Z, then reproject it back to X
    if (has_linear_constraints()) {
        const Eigen::MatrixXd& Q_f = unconstrained_basis();
        tmp = Q_f * (Q_f.transpose() * tmp);
    }

    return tmp.norm();
}

///////////////////////////////////////////////////////////////////////////

double DistanceBarrierRBProblem::compute_min_distance() const
{
    double min_distance = m_constraint.compute_minimum_distance(
        m_collision_mesh, m_assembler, m_assembler.rb_poses());
    return std::isfinite(min_distance) ? min_distance : -1;
}

double
DistanceBarrierRBProblem::compute_min_distance(const Eigen::VectorXd& x) const
{
    PosesD poses = this->dofs_to_poses(x);
    double min_distance = m_constraint.compute_minimum_distance(
        m_collision_mesh, m_assembler, poses);
    return std::isfinite(min_distance) ? min_distance : -1;
}

bool DistanceBarrierRBProblem::has_collisions(
    const Eigen::VectorXd& x_i, const Eigen::VectorXd& x_j)
{
    PROFILE_POINT("DistanceBarrierRBProblem::has_collisions");
    PROFILE_START();

    PosesD poses_i = this->dofs_to_poses(x_i);
    PosesD poses_j = this->dofs_to_poses(x_j);
    bool collisions =
        m_constraint.has_active_collisions(m_assembler, poses_i, poses_j);
    m_had_collisions |= collisions;
    // m_use_barriers := solve_collisions

    PROFILE_END();
    return m_use_barriers ? collisions : false;
}

double DistanceBarrierRBProblem::compute_earliest_toi(
    const Eigen::VectorXd& x_i, const Eigen::VectorXd& x_j)
{
    PROFILE_POINT("DistanceBarrierRBProblem::compute_earliest_toi");
    PROFILE_START();

    // m_use_barriers := solve_collisions
    // If we are not solve collisions then just compute if there was a
    // collision.
    if (!m_use_barriers) {
        this->has_collisions(x_i, x_j); // will set m_had_collisions
        PROFILE_END();
        return std::numeric_limits<double>::infinity();
    }

    PosesD poses_i = this->dofs_to_poses(x_i);
    PosesD poses_j = this->dofs_to_poses(x_j);
    double earliest_toi =
        m_constraint.compute_earliest_toi(m_assembler, poses_i, poses_j);
    m_had_collisions |= earliest_toi <= 1;
    // std::cout << "had collision: " << m_had_collisions << " at : " <<
    // earliest_toi << std::endl;
    PROFILE_END();
    return earliest_toi;
}

#ifdef ABD_WITH_DERIVATIVE_CHECK
// The following functions are used exclusivly to check that the
// gradient and hessian match a finite difference version.

void DistanceBarrierRBProblem::check_barrier_gradient(
    const Eigen::VectorXd& x,
    const CollisionConstraints& constraints,
    const Eigen::VectorXd& grad)
{
    ///////////////////////////////////////////////////////////////////////
    // Check that everything went well
    for (int i = 0; i < grad.size(); i++) {
        if (!std::isfinite(grad(i))) {
            spdlog::error("barrier gradient is not finite");
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // Finite difference check
    auto b = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_b;
        Eigen::SparseMatrix<double> hess_b;
        return compute_barrier_term(
            x, constraints, grad_b, hess_b,
            /*compute_grad=*/false, /*compute_hess=*/false);
    };
    Eigen::VectorXd grad_approx;
    fd::finite_gradient(x, b, grad_approx);
    if (!fd::compare_gradient(grad, grad_approx, 1e-3)) {
        spdlog::error("finite gradient check failed for barrier");
    }
}

void DistanceBarrierRBProblem::check_barrier_hessian(
    const Eigen::VectorXd& x,
    const CollisionConstraints& constraints,
    const Eigen::SparseMatrix<double>& hess)
{
    ///////////////////////////////////////////////////////////////////////
    // Check that everything went well
    typedef Eigen::SparseMatrix<double>::InnerIterator Iterator;
    for (int k = 0; k < hess.outerSize(); ++k) {
        for (Iterator it(hess, k); it; ++it) {
            if (!std::isfinite(it.value())) {
                spdlog::error("barrier hessian is not finite");
                return;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // Finite difference check
    // WARNING: The following check does not work well because the
    // different projections to PSD can affect results.
    Eigen::MatrixXd dense_hess(hess);
    auto b = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_b;
        Eigen::SparseMatrix<double> hess_b;
        compute_barrier_term(
            x, constraints, grad_b, hess_b,
            /*compute_grad=*/true, /*compute_hess=*/false);
        return grad_b;
    };
    Eigen::MatrixXd hess_approx;
    fd::finite_jacobian(x, b, hess_approx);
    // hess_approx = project_to_psd(hess_approx);
    if (!fd::compare_jacobian(hess, hess_approx, Constants::FINITE_DIFF_TEST)) {
        spdlog::error("finite hessian check failed for barrier");
    }
}

void DistanceBarrierRBProblem::check_friction_gradient(
    const Eigen::VectorXd& x, const Eigen::VectorXd& grad)
{
    ///////////////////////////////////////////////////////////////////////
    // Check that everything went well
    for (int i = 0; i < grad.size(); i++) {
        if (!std::isfinite(grad(i))) {
            spdlog::error("friction gradient is not finite");
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // Finite difference check
    auto f = [&](const Eigen::VectorXd& x) { return compute_friction_term(x); };
    Eigen::VectorXd grad_approx;
    fd::finite_gradient(x, f, grad_approx);
    if (!fd::compare_gradient(grad, grad_approx)) {
        spdlog::error("finite gradient check failed for friction");
    }

    ///////////////////////////////////////////////////////////////////////
    // Auto. diff. check
    typedef AutodiffType<Eigen::Dynamic> Diff;
    Diff::activate(x.size());
    Diff::D1MatrixXd V_diff =
        m_assembler.world_vertices(this->dofs_to_poses(Diff::d1vars(0, x)));

    Eigen::MatrixXd V0 = m_assembler.world_vertices(poses_t0);
    Diff::DDouble1 f_diff = ipc::compute_friction_potential(
        V0, V_diff, edges(), faces(), friction_constraints,
        static_friction_speed_bound * timestep());

    if (std::isfinite(f_diff.getGradient().sum())
        && !fd::compare_gradient(grad, f_diff.getGradient())) {
        spdlog::error("autodiff gradient check failed for friction");
    }
}

void DistanceBarrierRBProblem::check_friction_hessian(
    const Eigen::VectorXd& x, const Eigen::SparseMatrix<double>& hess)
{
    ///////////////////////////////////////////////////////////////////////
    // Check that everything went well
    typedef Eigen::SparseMatrix<double>::InnerIterator Iterator;
    for (int k = 0; k < hess.outerSize(); ++k) {
        for (Iterator it(hess, k); it; ++it) {
            if (!std::isfinite(it.value())) {
                spdlog::error("barrier hessian is not finite");
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // Finite difference check
    // Finite differences breaks when the displacements are zero.
    Eigen::MatrixXd V0 = m_assembler.world_vertices(poses_t0);
    Eigen::MatrixXd V1 = m_assembler.world_vertices(this->dofs_to_poses(x));
    if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0) {
        return;
    }

    Eigen::MatrixXd dense_hess(hess);

    auto f = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_f;
        compute_friction_term(x, grad_f);
        return grad_f;
    };
    Eigen::MatrixXd hess_approx;
    fd::finite_jacobian(x, f, hess_approx);
    hess_approx = project_to_psd(hess_approx);
    if (!fd::compare_hessian(hess, hess_approx, 1e-2)) {
        spdlog::error(
            "finite hessian check failed for friction "
            "(hess_L_inf_norm={:g} diff_L_inf_norm={:g})",
            dense_hess.lpNorm<Eigen::Infinity>(),
            (hess_approx - dense_hess).lpNorm<Eigen::Infinity>());
    }

    ///////////////////////////////////////////////////////////////////////
    // Auto. diff. check
    typedef AutodiffType<Eigen::Dynamic> Diff;
    Diff::activate(x.size());
    Diff::D2MatrixXd V_diff =
        m_assembler.world_vertices(this->dofs_to_poses(Diff::d2vars(0, x)));

    Diff::DDouble2 f_diff = ipc::compute_friction_potential(
        V0, V_diff, edges(), faces(), friction_constraints,
        static_friction_speed_bound * timestep());

    Eigen::MatrixXd hess_autodiff = project_to_psd(f_diff.getHessian());

    if (std::isfinite(hess_autodiff.sum())) {
        if (!fd::compare_hessian(dense_hess, hess_autodiff, 1e-3)) {
            spdlog::error(
                "autodiff hessian check failed for friction "
                "(hess_L_inf_norm={:g} diff_L_inf_norm={:g})",
                dense_hess.lpNorm<Eigen::Infinity>(),
                (hess_autodiff - dense_hess).lpNorm<Eigen::Infinity>());
        }
    } else {
        spdlog::warn("autodiff hessian failed for friction");
    }
}

void DistanceBarrierRBProblem::check_augmented_lagrangian_gradient(
    const Eigen::VectorXd& x, const Eigen::VectorXd& grad)
{
    ///////////////////////////////////////////////////////////////////////
    // Check that everything went well
    for (int i = 0; i < grad.size(); i++) {
        if (!std::isfinite(grad(i))) {
            spdlog::error("augmented lagrangian gradient is not finite");
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // Finite difference check
    auto AL = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_AL;
        Eigen::SparseMatrix<double> hess_AL;
        return compute_augmented_lagrangian(
            x, grad_AL, hess_AL,
            /*compute_grad=*/false, /*compute_hess=*/false);
    };
    Eigen::VectorXd grad_approx;
    fd::finite_gradient(x, AL, grad_approx);
    if (!fd::compare_gradient(grad, grad_approx)) {
        spdlog::error("finite gradient check failed for augmented lagrangian");
    }
}

void DistanceBarrierRBProblem::check_augmented_lagrangian_hessian(
    const Eigen::VectorXd& x, const Eigen::SparseMatrix<double>& hess)
{
    ///////////////////////////////////////////////////////////////////////
    // Check that everything went well
    typedef Eigen::SparseMatrix<double>::InnerIterator Iterator;
    for (int k = 0; k < hess.outerSize(); ++k) {
        for (Iterator it(hess, k); it; ++it) {
            if (!std::isfinite(it.value())) {
                spdlog::error("augmented lagrangian hessian is not finite");
                return;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // Finite difference check
    Eigen::MatrixXd dense_hess(hess);
    auto AL = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_AL;
        Eigen::SparseMatrix<double> hess_AL;
        compute_augmented_lagrangian(
            x, grad_AL, hess_AL,
            /*compute_grad=*/true, /*compute_hess=*/false);
        return grad_AL;
    };
    Eigen::MatrixXd hess_approx;
    fd::finite_jacobian(x, AL, hess_approx);
    if (!fd::compare_jacobian(hess, hess_approx)) {
        spdlog::error("finite hessian check failed for augmented lagrangian");
    }
}
#endif

} // namespace ipc::rigid
