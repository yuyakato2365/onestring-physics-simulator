// Functions for optimizing functions.
#include "newton_solver.hpp"

#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/writeOBJ.h>

#include <constants.hpp>
#include <logger.hpp>
#include <profiler.hpp>

#include <Eigen/SparseCore>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/task_arena.h>

// #define USE_GRADIENT_DESCENT

namespace ipc::rigid {

namespace {
double parallel_dot(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    return tbb::parallel_reduce(
        tbb::blocked_range<Eigen::Index>(0, a.size()), 0.0,
        [&](const tbb::blocked_range<Eigen::Index>& range, double sum) {
            for (Eigen::Index i = range.begin(); i != range.end(); ++i) {
                sum += a[i] * b[i];
            }
            return sum;
        },
        std::plus<double>());
}

bool parallel_bicgstab_solve(
    const Eigen::SparseMatrix<double>& matrix,
    const Eigen::VectorXd& rhs,
    Eigen::VectorXd& solution,
    int max_iterations,
    double tolerance)
{
    using RowSparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    const RowSparseMatrix A = matrix;
    const Eigen::Index n = A.rows();
    if (n == 0 || A.cols() != n || rhs.size() != n) {
        return false;
    }

    Eigen::VectorXd inverse_diagonal(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double diagonal = A.coeff(i, i);
        inverse_diagonal[i] =
            std::isfinite(diagonal) && std::abs(diagonal) > 1e-18
            ? 1.0 / diagonal
            : 1.0;
    }

    auto multiply = [&](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
        y.resize(n);
        tbb::parallel_for(
            tbb::blocked_range<Eigen::Index>(0, n),
            [&](const tbb::blocked_range<Eigen::Index>& range) {
                for (Eigen::Index row = range.begin(); row != range.end(); ++row) {
                    double value = 0.0;
                    for (RowSparseMatrix::InnerIterator entry(A, row); entry; ++entry) {
                        value += entry.value() * x[entry.col()];
                    }
                    y[row] = value;
                }
            });
    };

    solution.setZero(n);
    Eigen::VectorXd residual = rhs;
    Eigen::VectorXd shadow_residual = residual;
    Eigen::VectorXd direction = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd direction_preconditioned(n);
    Eigen::VectorXd product = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd intermediate(n);
    Eigen::VectorXd intermediate_preconditioned(n);
    Eigen::VectorXd second_product(n);
    const double rhs_norm = std::max(std::sqrt(parallel_dot(rhs, rhs)), 1e-30);
    double rho_previous = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        const double rho = parallel_dot(shadow_residual, residual);
        if (!std::isfinite(rho) || std::abs(rho) <= 1e-30) {
            return false;
        }
        const double beta = (rho / rho_previous) * (alpha / omega);
        direction = residual + beta * (direction - omega * product);
        direction_preconditioned = inverse_diagonal.cwiseProduct(direction);
        multiply(direction_preconditioned, product);
        const double denominator = parallel_dot(shadow_residual, product);
        if (!std::isfinite(denominator) || std::abs(denominator) <= 1e-30) {
            return false;
        }
        alpha = rho / denominator;
        intermediate = residual - alpha * product;
        const double intermediate_relative =
            std::sqrt(parallel_dot(intermediate, intermediate)) / rhs_norm;
        if (std::isfinite(intermediate_relative)
            && intermediate_relative <= tolerance) {
            solution += alpha * direction_preconditioned;
            return true;
        }
        intermediate_preconditioned =
            inverse_diagonal.cwiseProduct(intermediate);
        multiply(intermediate_preconditioned, second_product);
        const double second_norm = parallel_dot(second_product, second_product);
        if (!std::isfinite(second_norm) || second_norm <= 1e-30) {
            return false;
        }
        omega = parallel_dot(second_product, intermediate) / second_norm;
        if (!std::isfinite(omega) || std::abs(omega) <= 1e-30) {
            return false;
        }
        solution += alpha * direction_preconditioned
            + omega * intermediate_preconditioned;
        residual = intermediate - omega * second_product;
        const double relative_residual =
            std::sqrt(parallel_dot(residual, residual)) / rhs_norm;
        if (std::isfinite(relative_residual) && relative_residual <= tolerance) {
            return true;
        }
        rho_previous = rho;
    }
    return false;
}
} // namespace

NewtonSolver::NewtonSolver()
    : max_iterations(1000)
    , use_parallel_pcg(true)
    , parallel_pcg_max_iterations(300)
    , parallel_pcg_tolerance(1e-5)
    , iteration_number(0)
    , convergence_criteria(ConvergenceCriteria::ENERGY)
    , m_line_search_lower_bound(Constants::DEFAULT_LINE_SEARCH_LOWER_BOUND)
    , energy_conv_tol(Constants::DEFAULT_NEWTON_ENERGY_CONVERGENCE_TOL)
    , velocity_conv_tol(Constants::DEFAULT_NEWTON_VELOCITY_CONVERGENCE_TOL)
    , is_velocity_conv_tol_abs(false)
    , is_energy_converged(false)
{
    linear_solver = polysolve::LinearSolver::create("", "");
}

void NewtonSolver::settings(const nlohmann::json& json)
{
    max_iterations = json["max_iterations"];
    convergence_criteria = json["convergence_criteria"];
    energy_conv_tol = json["energy_conv_tol"];
    velocity_conv_tol = json["velocity_conv_tol"];
    is_velocity_conv_tol_abs = json["is_velocity_conv_tol_abs"];
    m_line_search_lower_bound = json["line_search_lower_bound"];
    use_parallel_pcg = json.value("use_parallel_pcg", true);
    parallel_pcg_max_iterations = json.value("parallel_pcg_max_iterations", 300);
    parallel_pcg_tolerance = json.value("parallel_pcg_tolerance", 1e-5);

    linear_solver_settings = json["linear_solver"];
    try {
        linear_solver =
            polysolve::LinearSolver::create(linear_solver_settings["name"], "");
    } catch (const std::runtime_error& err) {
        spdlog::error("{}! Using Eigen::SimplicialLDLT instead.", err.what());
        linear_solver_settings["name"] = "Eigen::SimplicialLDLT";
        linear_solver =
            polysolve::LinearSolver::create(linear_solver_settings["name"], "");
    }
    linear_solver->setParameters(linear_solver_settings);

    reset_stats();
}

nlohmann::json NewtonSolver::settings() const
{
    nlohmann::json settings;
    settings["max_iterations"] = max_iterations;
    settings["convergence_criteria"] = convergence_criteria;
    settings["linear_solver"] = linear_solver_settings;
    settings["energy_conv_tol"] = energy_conv_tol;
    settings["velocity_conv_tol"] = velocity_conv_tol;
    settings["is_velocity_conv_tol_abs"] = is_velocity_conv_tol_abs;
    settings["use_parallel_pcg"] = use_parallel_pcg;
    settings["parallel_pcg_max_iterations"] = parallel_pcg_max_iterations;
    settings["parallel_pcg_tolerance"] = parallel_pcg_tolerance;
    return settings;
}

void NewtonSolver::init_solve(const Eigen::VectorXd& x0)
{
    assert(problem_ptr != nullptr);
    is_energy_converged = false;
}

nlohmann::json NewtonSolver::stats() const
{
    return { { "total_newton_steps", newton_iterations },
             { "total_ls_steps", ls_iterations },
             { "num_newton_ls_fails", num_newton_ls_fails },
             { "num_grad_ls_fails", num_grad_ls_fails },
             { "count_fx", num_fx },
             { "count_grad", num_grad_fx },
             { "count_hess", num_hessian_fx },
             { "count_ccd", num_collision_check },
             { "parallel_bicgstab_solves", parallel_pcg_solves },
             { "parallel_bicgstab_fallbacks", parallel_pcg_fallbacks },
             { "total_regularizations", regularization_iterations } };
}

std::string NewtonSolver::stats_string() const
{
    return fmt::format(
        "total_newton_steps={:d} total_ls_steps={:d} "
        "num_newton_ls_fails={:d} num_grad_ls_fails={:d} count_fx={:d} "
        "count_grad={:d} count_hess={:d} count_ccd={:d} "
        "parallel_bicgstab_solves={:d} parallel_bicgstab_fallbacks={:d} "
        "total_regularizations={:d}",
        newton_iterations, ls_iterations, num_newton_ls_fails,
        num_grad_ls_fails, num_fx, num_grad_fx, num_hessian_fx,
        num_collision_check, parallel_pcg_solves, parallel_pcg_fallbacks,
        regularization_iterations);
}

void NewtonSolver::reset_stats()
{
    num_fx = 0;
    num_grad_fx = 0;
    num_hessian_fx = 0;
    num_collision_check = 0;
    ls_iterations = 0;
    newton_iterations = 0;
    num_newton_ls_fails = 0;
    num_grad_ls_fails = 0;
    parallel_pcg_solves = 0;
    parallel_pcg_fallbacks = 0;
    regularization_iterations = 0;
}

bool NewtonSolver::converged()
{
    switch (convergence_criteria) {
    case ConvergenceCriteria::VELOCITY: {
        Eigen::MatrixXd V_prev = problem_ptr->world_vertices(x);
        Eigen::MatrixXd V = problem_ptr->world_vertices(x + direction);
        // igl::writeOBJ(
        //     fmt::format("x{:04d}.obj", iteration_number), V,
        //     dynamic_cast<SimulationProblem*>(problem_ptr)
        //         ->faces());
        double step_max_speed =
            (V - V_prev).lpNorm<Eigen::Infinity>() / problem_ptr->timestep();

        // TODO: Renable this with a better check for static objects
        double tol = velocity_conv_tol;
        if (!is_velocity_conv_tol_abs) {
            tol *= problem_ptr->world_bbox_diagonal();
        }

        spdlog::info(
            "solver={} iter={:d} step_max_speed={:g} tol={:g}", //
            name(), iteration_number, step_max_speed, tol);

        double newton_step_energy =
            sqrt(abs(grad_constrained.dot(direction)));
        double gradient_step_energy =
            sqrt(abs(grad_constrained.dot(grad_constrained)));
        double gradient_mass_step_energy = sqrt(
            abs(grad_constrained.transpose() * problem_ptr->mass_matrix()
                * grad_constrained));

        // GREP_NCONV,1,0.130871,1.02885e-05,1.15864e-06,2.46467e-06,0.00434816,0.00399746,9.82746e-06
        /*
        std::cout << fmt::format(
                         "{},{:d},{:g},{:g},{:g},{:g},{:g},{:g},{:g}",
                         step_max_speed <= tol ? "GREP_CONV" : "GREP_NCONV",
                         iteration_number, step_max_speed,
                         newton_step_energy, gradient_step_energy,
                         gradient_mass_step_energy, direction_free.norm(),
                         direction_free.lpNorm<Eigen::Infinity>(),
                         sqrt(
                             abs(direction.transpose()
                                 * problem_ptr->mass_matrix() * direction)))
                  << std::endl;
        */
        is_energy_converged = step_max_speed <= tol;
        break;
    }
    case ConvergenceCriteria::ENERGY: {
        double step_energy = abs(grad_constrained.dot(direction));
        // double step_energy = abs(gradient_free.dot(gradient_free));
        double tol = energy_conv_tol;
        spdlog::info(
            "solver={} iter={:d} step_energy={:g} tol={:g}", //
            name(), iteration_number, step_energy, tol);
        is_energy_converged = step_energy <= tol;
        break;
    }
    }

    return is_energy_converged
        && problem_ptr->are_equality_constraints_satisfied(x);
}

OptimizationResults NewtonSolver::solve(const Eigen::VectorXd& x0)
{
    if (problem_ptr->has_linear_constraints()) {
        return solve_w_linear_constraints(x0);
    }

    PROFILE_POINT("NewtonSolver::solve");
    PROFILE_START();

    assert(problem_ptr != nullptr);
    // Initialize the working variables
    x_prev = x0;
    x = x0;

    double step_length = 1.0;
    double regulariztion_coeff = 0;

    spdlog::debug("solver={} action=BEGIN", name());

    std::string exit_reason = "exceeded the maximum allowable iterations";

    direction.setZero(problem_ptr->num_vars());
    // In this unconstrained case it will be equivalent to regularized gradient
    // it is used by converged() however.
    grad_constrained.setZero(problem_ptr->num_vars());

    is_energy_converged = false;
    bool success = false;

    for (iteration_number = 0; iteration_number < max_iterations;
         iteration_number++) {
        double fx = problem_ptr->compute_objective(x, gradient, hessian);
        // std::cout << "num Iter: " << iteration_number << " / " <<
        // max_iterations << std::endl;
        num_fx++;
        num_grad_fx++;
        num_hessian_fx++;

#ifdef USE_GRADIENT_DESCENT
        direction = -gradient;
#else
        grad_constrained = gradient;
        // Note that grad_constrained maybe modified here as regularization is applied
        bool solve_success = compute_regularized_direction(
            fx, grad_constrained, hessian, direction,
            regulariztion_coeff);
        if (!solve_success) {
            exit_reason = "regularization failed";
            break;
        }
#endif
        ///////////////////////////////////////////////////////////////////
        // Line search over newton direction

        // check for newton termination
        if (iteration_number > 0 && converged()) {
            exit_reason = "found a local optimum with newton dir";
            success = true;
            break;
        }

        step_length = 1;
        bool found_newton_step =
            line_search(x, direction, fx, grad_constrained, step_length);
        ///////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////
        // When newton direction fails, revert to gradient descent
        if (!found_newton_step) {
            // If I forced it to take a step when it should have converged
            if (iteration_number > 0) {
                num_newton_ls_fails++;
                spdlog::warn(
                    "solver={} iter={:d} failure=\"newton line-search\" "
                    "failsafe=\"gradient descent\"",
                    name(), iteration_number);
            }

            ///////////////////////////////////////////////////////////////
            // Line search over -gradient direction
            // replace delta_x by gradient direction
            direction = -gradient;

            // WARNING: Disable convergence using negative gradient
            // check for newton termination again
            // if (iteration_number > 0 && converged()) {
            //     exit_reason = "found a local optimum with -grad dir";
            //     success = true;
            //     break;
            // }
            step_length = 1;
            bool found_gradient_step =
                line_search(x, direction, fx, gradient, step_length);
            ///////////////////////////////////////////////////////////////

            // When gradient direction fails, exit
            if (!found_gradient_step) {
                // If I forced it to take a step when it should have
                // converged
                if (iteration_number == 0 && converged()) {
                    // Do not consider this a failure
                    spdlog::info(
                        "solver={} msg=\"converged without taking a step\"",
                        name());
                    exit_reason = "found a local optimum with -grad dir";
                    success = true;
                    break;
                }
                num_grad_ls_fails++;
                spdlog::error(
                    "solver={} iter={:d} failure=\"gradient line-search\" "
                    "failsafe=\"none\"",
                    name(), iteration_number);
                exit_reason = "line-search failed";
                break;
            }
        }
        ///////////////////////////////////////////////////////////////////

        spdlog::debug(
            "solver={} iter={:d} step_length={:g}", name(), iteration_number,
            step_length);

        x_prev = x;
        x += step_length * direction;
        assert(!problem_ptr->has_collisions(x_prev, x));
        newton_iterations++; // Only count complete steps

        post_step_update();
    } // end for loop

    spdlog::info(
        "solver={} action=END total_iter={:d} exit_reason=\"{}\"", name(),
        iteration_number, exit_reason);

    PROFILE_END();
    return OptimizationResults(
        x, problem_ptr->compute_objective(x), success, true, iteration_number);
}

OptimizationResults
NewtonSolver::solve_w_linear_constraints(const Eigen::VectorXd& x0)
{
    PROFILE_POINT("NewtonSolver::solve_w_linear_constraints");
    PROFILE_START();

    assert(problem_ptr != nullptr);
    // Initialize the working variables

    // Get the unconstrained basis (basis for subspace that is orthogonal to linear constraints)
    // the space spanned by this basis will be called Z_free from here on
    const Eigen::MatrixXd& Q_f = problem_ptr->unconstrained_basis();

    // IF HAS LINEAR CONSTRAINTS
    // Enforce constraints before start of optimization
    // to be sure
    x = problem_ptr->enforce_linear_constraint(x0);
    x_prev = x;

    double step_length = 1.0;
    double regulariztion_coeff = 0;

    spdlog::debug("solver={} action=BEGIN", name());

    std::string exit_reason = "exceeded the maximum allowable iterations";

    // Direction and grad_direction in X space
    direction.setZero(problem_ptr->num_vars());
    grad_constrained.setZero(problem_ptr->num_vars());

    // Update direction in Z_free coordinates
    Eigen::VectorXd direction_z_free;

    // Gradient wrt Z_free dofs (Grad_z(E(Q_f*z_free)) = Q_f.T * Grad_x(Ex))
    Eigen::VectorXd gradient_z_free;
    Eigen::SparseMatrix<double> hessian_z_free;

    Eigen::VectorXd grad_direction_z =
        Eigen::VectorXd::Zero(problem_ptr->num_vars());

    is_energy_converged = false;
    bool success = false;

    for (iteration_number = 0; iteration_number < max_iterations;
         iteration_number++) {

        // Calculate gradient and hessian in X space
        double fx = problem_ptr->compute_objective(x, gradient, hessian);

        num_fx++;
        num_grad_fx++;
        num_hessian_fx++;

        spdlog::trace(
            "solve_w_linear_constraints: num Iter: {} / {}", iteration_number,
            max_iterations);

        // Get the gradient and hessian in Z_free space
        // Grad_z(E(Uz)) = Q_f.T * Grad_x(E(x)))
        // Hess_z(E(Uz)) = Q_f.T * Hess_x(E(x)) * Q_f
        gradient_z_free = Q_f.transpose() * gradient;
        hessian_z_free = (Q_f.transpose() * hessian * Q_f).sparseView();

#ifdef USE_GRADIENT_DESCENT
        direction_free = -gradient_free;
#else
        bool solve_success = compute_regularized_direction(
            fx, gradient_z_free, hessian_z_free, direction_z_free,
            regulariztion_coeff);
        if (!solve_success) {
            exit_reason = "regularization failed";
            break;
        }
#endif

        ///////////////////////////////////////////////////////////////////
        // Line search over newton direction
        // get grad direction for lineseach

        // We need the gradient and direction in original problem space
        // for convergence check and line search
        grad_constrained = Q_f * gradient_z_free;
        direction      = Q_f * direction_z_free;

        // check for newton termination
        if (iteration_number > 0 && converged()) {
            exit_reason = "found a local optimum with newton dir";
            success = true;
            break;
        }

        step_length = 1;
        bool found_newton_step =
            line_search(x, direction, fx, grad_constrained, step_length);
        ///////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////
        // When newton direction fails, revert to gradient descent
        if (!found_newton_step) {
            // If I forced it to take a step when it should have converged
            if (iteration_number > 0) {
                num_newton_ls_fails++;
                spdlog::warn(
                    "solver={} iter={:d} failure=\"newton line-search\" "
                    "failsafe=\"gradient descent\"",
                    name(), iteration_number);
            }

            ///////////////////////////////////////////////////////////////
            // Line search over -gradient direction
            // replace delta_x by gradient direction
            direction = -grad_constrained;

            // WARNING: Disable convergence using negative gradient
            // check for newton termination again
            // if (iteration_number > 0 && converged()) {
            //     exit_reason = "found a local optimum with -grad dir";
            //     success = true;
            //     break;
            // }
            step_length = 1;
            bool found_gradient_step =
                line_search(x, direction, fx, grad_constrained, step_length);
            ///////////////////////////////////////////////////////////////

            // When gradient direction fails, exit
            if (!found_gradient_step) {
                // If I forced it to take a step when it should have
                // converged
                if (iteration_number == 0 && converged()) {
                    // Do not consider this a failure
                    spdlog::info(
                        "solver={} msg=\"converged without taking a step\"",
                        name());
                    exit_reason = "found a local optimum with -grad dir";
                    success = true;
                    break;
                }
                num_grad_ls_fails++;
                spdlog::error(
                    "solver={} iter={:d} failure=\"gradient line-search\" "
                    "failsafe=\"none\"",
                    name(), iteration_number);
                exit_reason = "line-search failed";
                break;
            }
        }
        ///////////////////////////////////////////////////////////////////

        spdlog::debug(
            "solver={} iter={:d} step_length={:g}", name(), iteration_number,
            step_length);

        x_prev = x;
        x += step_length * direction;
        assert(!problem_ptr->has_collisions(x_prev, x));
        newton_iterations++; // Only count complete steps

        post_step_update();
    } // end for loop

    spdlog::info(
        "solver={} action=END total_iter={:d} exit_reason=\"{}\"", name(),
        iteration_number, exit_reason);

    PROFILE_END();
    return OptimizationResults(
        x, problem_ptr->compute_objective(x), true/*success*/, true, iteration_number);
}

bool NewtonSolver::line_search(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& dir,
    const double fx,
    const Eigen::VectorXd& grad_fx,
    double& step_length)
{
    PROFILE_POINT("NewtonSolver::line_search");
    PROFILE_START();

    bool success = false;
    int num_it = 0;
    // double lower_bound = line_search_lower_bound() / -grad_fx.dot(dir);
    double lower_bound =
        line_search_lower_bound() / sqrt(abs(grad_fx.dot(dir)));
    lower_bound = std::min(lower_bound, 1e-1);
    // double lower_bound = line_search_lower_bound() / dir.squaredNorm();
    // double lower_bound = line_search_lower_bound();

    // Filter the step length so that x to x + α * Δx is collision free for
    // α ≤ step_length.
    num_collision_check++; // Count the number of collision checks
    bool is_ccd_aligned_with_newton_update =
        problem_ptr->is_ccd_aligned_with_newton_update();
    double max_step_size = 1;
    if (is_ccd_aligned_with_newton_update) {
        max_step_size =
            std::min(problem_ptr->compute_earliest_toi(x, x + dir), 1.0);
        step_length = std::min(step_length, max_step_size);
    }
    // #ifndef NDEBUG
    // while (problem_ptr->has_collisions(x, x + step_length * dir)) {
    //     spdlog::error(
    //         "max_step_size={:g} has collisions reducing the step!",
    //         step_length);
    //     step_length *= 0.8;
    // }
    // #endif
    if (step_length < lower_bound) {
        spdlog::error(
            "solver={} iter={:d} failure=\"initial step_length (α={:g}) is"
            " less than lower bound ({:g})\"",
            name(), iteration_number, step_length, lower_bound);
    }

    double fxi = std::numeric_limits<double>::infinity();
    while (std::isfinite(lower_bound) && step_length >= lower_bound) {
        num_it++;        // Count the number of iterations
        ls_iterations++; // Count the gloabal number of iterations

        // Compute the next variable
        Eigen::VectorXd xi = x + step_length * dir;

        // NOTE: We do not need to check for collisions because we filtered
        // the step length.
        // Check for collisions between newton updates
        if (is_ccd_aligned_with_newton_update
            || !problem_ptr->has_collisions(x, xi)) {
            fxi = problem_ptr->compute_objective(xi);
            num_fx++; // Count the number of objective computations
            if (fxi < fx) {
                success = true;
                break; // while loop
            }
        }

        // Try again with a smaller step_length
        step_length /= 2.0;
    }

    // PROFILE_MESSAGE(
    //    LINE_SEARCH, "success,it,dir",
    //    fmt::format("{},{:d},{:10e}", success, num_it, dir.norm()));
    // PROFILE_END(LINE_SEARCH);

    if (!success && iteration_number > 0 && max_step_size >= lower_bound) {
        if (!std::isfinite(lower_bound)) {
            spdlog::warn(
                "solver={} iter={:d} failure=\"line-search ∇f(x)⋅dir=0\"",
                name(), iteration_number);
        } else {
            spdlog::warn(
                "solver={} iter={:d} failure=\"line-search α ≤ {:g} / {:g} "
                "= {:g}; f(x + αΔx)-f(x)={:g}; α_max={:g}\"",
                name(), iteration_number, line_search_lower_bound(),
                sqrt(abs(grad_fx.dot(dir))), lower_bound, fxi - fx,
                max_step_size);
            sample_search_direction(
                x, dir,
                [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
                    double fx = problem_ptr->compute_objective(x, grad);
                    Eigen::VectorXi free_dof = problem_ptr->free_dof();
                    Eigen::VectorXd grad_free;
                    igl::slice(grad, free_dof, grad_free);
                    grad.setZero();
                    igl::slice_into(grad_free, free_dof, grad);
                    PROFILE_END();
                    return fx;
                },
                max_step_size);
        }
    }

    spdlog::info(
        "solver={} iter={:d} α_min={:g} α_max={:g} α={:g} ",
        // "d_min(x)={:g} d_min(x+αΔx)={:g}",
        name(), iteration_number, lower_bound, max_step_size, step_length
        // , problem_ptr->compute_min_distance(x),
        // problem_ptr->compute_min_distance(x + step_length * dir)
    );

    PROFILE_END();
    return success;
}

double norm_Linf(const Eigen::SparseMatrix<double>& M)
{
    double norm = 0;
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            norm = std::max(norm, abs(it.value()));
        }
    }
    return norm;
}

bool NewtonSolver::compute_regularized_direction(
    double& fx,
    Eigen::VectorXd& gradient,
    Eigen::SparseMatrix<double>& hessian,
    Eigen::VectorXd& direction,
    double& coeff)
{
    PROFILE_POINT("NewtonSolver::compute_regularized_direction");
    PROFILE_START();

    bool success = false;
    while (!success) {
        double regularized_fx = fx;
        Eigen::VectorXd regularized_gradient = gradient;
        Eigen::SparseMatrix<double> regularized_hessian = hessian;

        if (coeff > 0) {
            // regularized_fx += coeff / 2 * (x - x_prev).squaredNorm();
            // regularized_gradient += coeff * (x - x_prev);
            Eigen::SparseMatrix<double> I(hessian.rows(), hessian.cols());
            I.setIdentity();
            regularized_hessian += coeff * I;
            regularization_iterations++;
        }

        success = compute_direction(
            regularized_gradient, regularized_hessian, direction,
            /*make_psd=*/false);
        success = success && regularized_gradient.dot(direction) <= 0;

        // Update coefficient adaptivly when the solve fails
        if (success) {
            coeff /= 2;
            if (coeff < 1e-8) {
                coeff = 0;
            }
            fx = regularized_fx;
            gradient = regularized_gradient;
            hessian = regularized_hessian;
        } else {
            coeff = std::max(2 * coeff, 1e-8);
            if (!std::isfinite(coeff)) {
                spdlog::error(
                    "solver={} iter={:d} failure=\"regularization failed "
                    "(coeff={:g})\" failsafe=\"none\"",
                    name(), iteration_number, coeff);
                PROFILE_END();
                return false;
            }
            spdlog::warn(
                "solver={} iter={:d} failure=\"solve failed (∇f⋅Δx={:g}, "
                "||H||_inf={:g}); increasing regularization coeff={:g}\"",
                name(), iteration_number, gradient_free.dot(direction_free),
                norm_Linf(hessian), coeff);
        }
    }
    PROFILE_END();
    return success;
}

bool NewtonSolver::compute_direction(
    const Eigen::VectorXd& gradient,
    const Eigen::SparseMatrix<double>& hessian,
    Eigen::VectorXd& direction,
    bool make_psd)
{
    PROFILE_POINT("NewtonSolver::compute_direction");
    PROFILE_START();

    // Check if the hessian is positive semi-definite.
    // Eigen::LLT<Eigen::MatrixXd> LLT_H((Eigen::MatrixXd(hessian)));
    // if (LLT_H.info() == Eigen::NumericalIssue) {
    //     spdlog::warn(
    //         "solver={} iter={:d} failure=\"possibly non semi-positive "
    //         "definite Hessian\"",
    //         name(), iteration_number);
    // }

    // Solve for the Newton direction (Δx = -H⁻¹∇f).
    // Return true if the solve was successful.
    bool solve_success = false;
    bool parallel_solve_used = false;

    // if (hessian.rows() <= 1200) { // <= 200 bodies
    //     Eigen::MatrixXd dense_hessian(hessian);
    //     direction = dense_hessian.ldlt().solve(-gradient);
    //     solve_success = true;
    // } else {
    if (use_parallel_pcg && hessian.rows() >= 256) {
        direction = Eigen::VectorXd::Zero(gradient.size());
        if (parallel_bicgstab_solve(
                hessian, -gradient, direction,
                parallel_pcg_max_iterations, parallel_pcg_tolerance)) {
            parallel_pcg_solves++;
            parallel_solve_used = true;
            if (parallel_pcg_solves == 1) {
                spdlog::info(
                    "solver={} linear_solver=tbb_parallel_bicgstab "
                    "hessian_rows={} max_concurrency={}",
                    name(), hessian.rows(),
                    tbb::this_task_arena::max_concurrency());
            }
            solve_success = true;
        } else {
            parallel_pcg_fallbacks++;
            spdlog::debug(
                "solver={} parallel BiCGSTAB did not converge; falling back to {}",
                name(), linear_solver_settings.value("name", "configured direct solver"));
        }
    }
    if (!solve_success) {
    linear_solver->analyzePattern(hessian, hessian.rows());
    linear_solver->factorize(hessian);
    nlohmann::json info;
    linear_solver->getInfo(info);
    // TODO: This check only works for direct Eigen solvers
    if (!info.contains("solver_info") || info["solver_info"] == "Success") {
        // TODO: Do we have a better initial guess for iterative
        // solvers?
        direction = Eigen::VectorXd::Zero(gradient.size());
        linear_solver->solve(-gradient, direction);
        linear_solver->getInfo(info);
        if (!info.contains("solver_info") || info["solver_info"] == "Success") {
            solve_success = true;
        } else {
            spdlog::warn(
                "solver={} iter={:d} failure=\"sparse solve for newton "
                "direction\" failsafe=\"gradient descent\"",
                name(), iteration_number);
        }
    } else {
        spdlog::warn(
            "solver={} iter={:d} failure=\"sparse decomposition of the "
            "hessian\" failsafe=\"gradient descent\"",
            name(), iteration_number);
    }
    }
    // }

    // Check solve residual
    if (solve_success) {
        double solve_residual = (hessian * direction + gradient).norm();
        const double solve_residual_limit = parallel_solve_used
            ? std::max(
                  1e-8,
                  10.0 * parallel_pcg_tolerance
                      * std::max(gradient.norm(), 1.0))
            : 1e-8;
        if (solve_residual > solve_residual_limit) {
            spdlog::warn(
                "solver={} iter={:d} "
                "failure=\"linear solve residual ({:g}) > {:g}; "
                "||g||_{{L^inf}}={:g} ||H||_{{L^inf}}={:g}\"",
                name(), iteration_number, solve_residual, solve_residual_limit,
                gradient.lpNorm<Eigen::Infinity>(), norm_Linf(hessian));
        }
        solve_success = std::isfinite(solve_residual);
    }

    if (!solve_success) {
        direction = -gradient;
    }

    if (solve_success && make_psd && direction.dot(gradient) > 0) {
        // If delta_x is not a descent direction then we want to modify the
        // hessian to be diagonally dominant with positive elements on the
        // diagonal (positive definite). We do this by adding μI to the
        // hessian. This can result in doing a step of gradient descent.
        Eigen::SparseMatrix<double> psd_hessian = hessian;
        double mu = make_matrix_positive_definite(psd_hessian);
        spdlog::warn(
            "solver={} iter={:d} failure=\"newton direction not descent "
            "direction\" failsafe=\"H += μI\" μ={:g}",
            name(), iteration_number, mu);
        solve_success =
            compute_direction(gradient, psd_hessian, direction, false);
        double dir_dot_grad = direction.dot(gradient);
        if (dir_dot_grad > 0) {
            spdlog::error(
                "solver={} iter={:d} failure=\"adjusted newton "
                "direction not descent direction\" failsafe=\"gradient "
                "descent\" dir_dot_grad={:g}",
                name(), iteration_number, dir_dot_grad);
            direction = -gradient;
        }
    }

    PROFILE_END();
    return solve_success;
}

// Make the matrix positive definite (x^T A x > 0).
double make_matrix_positive_definite(Eigen::SparseMatrix<double>& A)
{
    // Conservative way of making A PSD by making it diagonally dominant
    // with all positive diagonal entries
    Eigen::SparseMatrix<double> I(A.rows(), A.rows());
    I.setIdentity();
    // Entries along the diagonal of A
    Eigen::VectorXd diag = Eigen::VectorXd::Zero(A.rows());
    // Sum of columns per row not including the diagonal entry
    // (∑_{i≠j}|a_{ij}|)
    Eigen::VectorXd sum_row = Eigen::VectorXd::Zero(A.rows());

    // Loop over elements adding them to the appropriate vector above
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            if (it.row() == it.col()) { // Diagonal element
                diag(it.row()) = it.value();
            } else { // Non-diagonal element
                sum_row(it.row()) += abs(it.value());
            }
        }
    }
    // Take max to ensure all diagonal elements are dominant
    double mu = std::max((sum_row - diag).maxCoeff(), 0.0);
    A += mu * I;
    return mu;
}

void NewtonSolver::post_step_update()
{
    if (is_energy_converged
        && !problem_ptr->are_equality_constraints_satisfied(x)) {
        problem_ptr->update_augmented_lagrangian(x);
    }
}

// Log samples along the search direction.
void sample_search_direction(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& dir,
    const std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)>&
        f_and_gradf,
    double max_step)
{
    Eigen::VectorXd grad_fx;

    Eigen::VectorXd sampling;
    int num_samples = 25;
    bool use_geometric_sampling = true;
    if (use_geometric_sampling) {
        sampling.resize(2 * num_samples + 1);
        double max_pow = log10(max_step);
        sampling.tail(num_samples) =
            pow(10, Eigen::ArrayXd::LinSpaced(num_samples, -16, max_pow));
        sampling(num_samples) = 0;
        sampling.head(num_samples) = -sampling.tail(num_samples).reverse();
    } else {
        sampling = Eigen::VectorXd::LinSpaced(
            2 * num_samples + 1, -max_step, max_step);
    }

    double fx0 = f_and_gradf(x, grad_fx);
    for (int i = 0; i < sampling.size(); i++) {
        double step_length = sampling(i);
        double fx = f_and_gradf(x + step_length * dir, grad_fx);
        spdlog::log(
            step_length < 0 ? spdlog::level::debug : spdlog::level::info,
            "method=line_search step_length={:+.1e} obj={:.16g} "
            "(obj_i-obj_0)={:.16g} grad_L∞norm={:g}",
            step_length, fx, fx - fx0, grad_fx.lpNorm<Eigen::Infinity>());
    }
}

} // namespace ipc::rigid
