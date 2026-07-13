#include "rigid_body.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <autodiff/autodiff_types.hpp>
#include <finitediff.hpp>
#include <logger.hpp>
#include <physics/mass.hpp>
#include <profiler.hpp>
#include <utils/eigen_ext.hpp>
#include <utils/flatten.hpp>
#include <utils/not_implemented_error.hpp>

namespace ipc::rigid {

VectorMax3d center_vertices(
    Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    PoseD& pose,
    double& mass,
    VectorMax3d& center_of_mass,
    MatrixMax3d& inertia_tensor)
{
    int dim = vertices.cols();
    // Center of mass in input coordinates
    VectorMax3d in_origin;
    in_origin.resize(dim);
    in_origin.setZero();
    in_origin += pose.position;

    vertices.rowwise() += pose.position.transpose();

    // compute the center of mass several times to get more accurate
    double temp_mass;
    VectorMax3d com;
    MatrixMax3d inertia;
    pose.position.setZero(dim);
    for (int i = 0; i < 10; i++) {
        com.setZero();
        inertia.setZero();
        compute_mass_properties(
            vertices, dim == 2 || faces.size() == 0 ? edges : faces, temp_mass,
            com, inertia);

        // Breaking only after we've calculated the latest mass properties
        // following our updates on vertices and pose
        if (com.squaredNorm() < 1e-8) {
            break;
        }

        vertices.rowwise() -= com.transpose();
        in_origin -= com;
        pose.position += com;
    }

    mass = temp_mass;
    center_of_mass = com;
    inertia_tensor = inertia;

    return in_origin;
}

RigidBody::RigidBody(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    const PoseD& pose,
    const PoseD& velocity,
    const PoseD& force,
    const double density,
    const bool oriented,
    const int group_id,
    const RigidBodyType type,
    const double kinematic_max_time,
    const std::deque<PoseD>& kinematic_poses)
    : group_id(group_id)
    , type(type)
    , vertices(vertices)
    , edges(edges)
    , faces(faces)
    , is_oriented(oriented)
    , mesh_selector(vertices.rows(), edges, faces)
    , pose(pose)
    , velocity(velocity)
    , force(force)
    , kinematic_max_time(kinematic_max_time)
    , kinematic_poses(kinematic_poses)
{
    is_dof_fixed = VectorXb::Zero(12);
    assert(dim() == pose.dim());
    assert(dim() == velocity.dim());
    assert(dim() == force.dim());
    assert(edges.size() == 0 || edges.cols() == 2);
    assert(faces.size() == 0 || faces.cols() == 3);

    // Store initial transformation matrix as it is needed later
    auto R_initial = pose.transform;

    VectorMax3d center_of_mass;
    MatrixMax3d I;
    this->in_origin = center_vertices(
        this->vertices, edges, faces, this->pose, mass, center_of_mass, I);
    volume = mass;
    // Mass above is actually volume in m³ and density is Kg/m³
    mass = volume * density;
    if (dim() == 3) {
        compute_principle_axes_3D(I, R0);
    } else {
        compute_principle_axes_2D(I, R0);
    }
    // R = RᵢR₀
    this->pose.transform = this->pose.transform * R0;
    // v = Rv₀ + p = RᵢR₀v₀ + p
    this->vertices = this->vertices * R0; // R₀ᵀ * V₀ᵀ = V₀ * R₀
    this->in_origin = R0.transpose() * this->in_origin;
    // Rdot^world_body_old = [w_world]_x * R^world_body_old
    // https://arxiv.org/pdf/1609.06088.pdf
    // Rdot^world_body_new = Rdot^world_body_old * R^world_body_old.T *
    // R^world_body_new
    this->velocity.transform =
        this->velocity.transform * R_initial.transpose() * R0;

    // Update the previous pose and velocity to reflect the changes made
    // here
    this->pose_prev = this->pose;
    this->velocity_prev = this->velocity;

    this->acceleration = PoseD::Zero(dim());

    // Compute and construct some useful constants
    // Recompute mass moments required to construct the generalized mass matrix
    VectorMax3d second_moments;
    if (dim() == 3 && faces.size() > 0) {
        // This method works for closed triangular meshes
        Eigen::Matrix<double, 6, 1> moments;
        compute_mass_moments_3D(this->vertices, faces, moments);
        second_moments = moments.middleRows(0, 3);
        // Since the body is centred and using principal axes as basis
        // Check that integrals(xy dm, xz dm, yz dm) is almost zero
        assert(moments.tail(3).norm() < 1E-7);
    } else if (dim() == 3 && faces.size() == 0) {
        // This method works for wire bodies in 3D space
        Eigen::Matrix<double, 6, 1> moments;
        compute_mass_moments_3D_no_faces(this->vertices, edges, moments);
        second_moments = moments.middleRows(0, 3);
        // Since the body is centred and using principal axes as basis
        // Check that integrals(xy dm, xz dm, yz dm) is almost zero
        assert(moments.tail(3).norm() < 1E-7);

    } else {
        Eigen::Matrix<double, 3, 1> moments;
        compute_mass_moments_2D(this->vertices, edges, moments);
        second_moments = moments.head(2);
        // Since the body is centred and using principal axes as basis
        // Check that integral(xy dm) is almost zero
        assert(abs(moments[2]) < 1E-7);
    }

    mass_matrix.resize(ndof());
    mass_matrix.setZero();
    mass_matrix.diagonal().head(pos_ndof()).setConstant(mass);

    int block_size = pos_ndof();
    for (int block_i = 1; block_i < pos_ndof() + 1; block_i++) {
        mass_matrix.diagonal().middleRows(block_i * block_size, block_size) =
            density * second_moments;
    }

    // Set follower forces jacobian to zero at the start
    this->follower_force_jacobian =
        MatrixMax12d::Zero(this->ndof(), this->ndof());

    r_max = this->vertices.rowwise().norm().maxCoeff();

    average_edge_length = 0;
    for (long i = 0; i < edges.rows(); i++) {
        average_edge_length +=
            (this->vertices.row(edges(i, 0)) - this->vertices.row(edges(i, 1)))
                .norm();
    }
    if (edges.rows() > 0) {
        average_edge_length /= edges.rows();
    }
    assert(std::isfinite(average_edge_length));

    init_bvh();
}

void RigidBody::init_bvh()
{
    PROFILE_POINT("RigidBody::init_bvh");
    PROFILE_START();

    // heterogenous bounding boxes
    std::vector<std::array<Eigen::Vector3d, 2>> aabbs(
        num_codim_vertices() + num_codim_edges() + num_faces());

    for (size_t i = 0; i < num_codim_vertices(); i++) {
        size_t vi = mesh_selector.codim_vertices_to_vertices(i);
        if (dim() == 2) {
            aabbs[i][0][2] = 0;
            aabbs[i][1][2] = 0;
        }
        aabbs[i][0].head(dim()) = vertices.row(i);
        aabbs[i][1].head(dim()) = vertices.row(i);
    }

    size_t start_i = num_codim_vertices();
    for (size_t i = 0; i < num_codim_edges(); i++) {
        size_t ei = mesh_selector.codim_edges_to_edges(i);
        const auto& e0 = vertices.row(edges(ei, 0));
        const auto& e1 = vertices.row(edges(ei, 1));

        if (dim() == 2) {
            aabbs[start_i + i][0][2] = 0;
            aabbs[start_i + i][1][2] = 0;
        }
        aabbs[start_i + i][0].head(dim()) = e0.cwiseMin(e1);
        aabbs[start_i + i][1].head(dim()) = e0.cwiseMax(e1);
    }

    start_i += num_codim_edges();
    for (size_t i = 0; i < num_faces(); i++) {
        assert(dim() == 3);
        const auto& f0 = vertices.row(faces(i, 0));
        const auto& f1 = vertices.row(faces(i, 1));
        const auto& f2 = vertices.row(faces(i, 2));
        aabbs[start_i + i][0] = f0.cwiseMin(f1).cwiseMin(f2);
        aabbs[start_i + i][1] = f0.cwiseMax(f1).cwiseMax(f2);
    }

    bvh.init(aabbs);

    PROFILE_END();
}

Eigen::MatrixXd RigidBody::world_vertices_jacobian() const
{
    int n_verts = vertices.rows();
    int n_dims = vertices.cols();

    Eigen::MatrixXd world_vertices_jac;
    world_vertices_jac.resize(n_verts * n_dims, ndof());
    world_vertices_jac.setZero();

    for (int v_i = 0; v_i < n_verts; v_i++) {
        if (n_dims == 2) {
            assert(ndof(n_dims) == 6);
            Eigen::Block<Eigen::MatrixXd, 2, 6> block =
                world_vertices_jac.block<2, 6>(v_i * n_dims, 0);

            J_matrix(vertices.row(v_i), block);
        } else if (n_dims == 3) {
            assert(ndof(n_dims) == 12);
            Eigen::Block<Eigen::MatrixXd, 3, 12> block =
                world_vertices_jac.block<3, 12>(v_i * n_dims, 0);

            J_matrix(vertices.row(v_i), block);
        } else {
            throw std::runtime_error("Invalid number of dimensions: " + n_dims);
        }
    }

    return world_vertices_jac;
}

void RigidBody::world_vertices_jacobian(
    long rb_v0_i, Eigen::MatrixXd& jac) const
{
    assert(rb_v0_i <= jac.rows() - vertices.size());
    assert(jac.cols() == ndof());

    jac.block(rb_v0_i * dim(), 0, dim() * vertices.rows(), ndof()) =
        world_vertices_jacobian();
}

Eigen::MatrixXd RigidBody::world_vertices(
    const PoseD& pose, long rb_v0_i, Eigen::MatrixXd& V) const
{
    assert(rb_v0_i >= 0 && rb_v0_i <= V.rows() - vertices.rows());
    assert(V.cols() == dim());

    V.block(rb_v0_i, 0, vertices.rows(), vertices.cols()) =
        world_vertices(pose);

    return V;
}

Eigen::MatrixXd RigidBody::world_vertices_diff(
    const PoseD& pose,
    long rb_v0_i,
    Eigen::MatrixXd& V,
    Eigen::MatrixXd& jac) const
{
    assert(rb_v0_i >= 0 && rb_v0_i <= V.rows() - vertices.rows());
    assert(V.cols() == dim());
    assert(rb_v0_i <= jac.rows() - vertices.size());
    assert(jac.cols() == ndof());

    V.block(rb_v0_i, 0, vertices.rows(), vertices.cols()) =
        world_vertices(pose);

    jac.block(dim() * rb_v0_i, 0, dim() * vertices.rows(), ndof()) =
        world_vertices_jacobian();

    return V;
}

Eigen::MatrixXd RigidBody::world_velocities() const
{
    // compute ẋ = Ȧ * x_bar + ṗ
    if (dim() == 2) {
        return (vertices * velocity.transform.transpose()).rowwise()
            + velocity.position.transpose();
    }
    return (vertices * velocity.transform.transpose()).rowwise()
        + velocity.position.transpose();
}

void RigidBody::compute_bounding_box(
    const PoseD& pose_t0,
    const PoseD& pose_t1,
    VectorMax3d& box_min,
    VectorMax3d& box_max) const
{
    //PROFILE_POINT("RigidBody::compute_bounding_box");
    //PROFILE_START();

    // If the body is not rotating then just use the linearized
    // trajectory
    if (type == RigidBodyType::STATIC
        || (pose_t0.transform.array() == pose_t1.transform.array()).all()) {
        Eigen::MatrixXd V0 = world_vertices(pose_t0);
        box_min = V0.colwise().minCoeff();
        box_max = V0.colwise().maxCoeff();
        Eigen::MatrixXd V1 = world_vertices(pose_t1);
        box_min = box_min.cwiseMin(V1.colwise().minCoeff().transpose());
        box_max = box_max.cwiseMax(V1.colwise().maxCoeff().transpose());
    } else {
        // TODO: AFFINE: This bound is not guaranteed when using affine bodies
        // Use the maximum radius of the body to bound all rotations
        box_min = pose_t0.position.cwiseMin(pose_t1.position).array() - r_max;
        box_max = pose_t0.position.cwiseMax(pose_t1.position).array() + r_max;
    }

    //PROFILE_END();
}

void RigidBody::add_force(VectorMax3d material_point, VectorMax3d point_force)
{
    assert(point_force.rows() == this->dim());
    assert(material_point.rows() == this->dim());
    this->force += J_matrix(material_point).transpose() * point_force;
}

void RigidBody::add_follower_force(
    VectorMax3d material_point, VectorMax3d point_force)
{
    MatrixMax<double, 3, 12> G;
    int d = this->dim();
    G.resize(d, ndof(d));
    G.setZero();
    for (int i = 0; i < d; i++) {
        G.block(i, (i + 1) * d, 1, d) = point_force.transpose();
    }

    this->follower_force_jacobian += J_matrix(material_point).transpose() * G;
}

void RigidBody::add_torque(VectorMax3d torque)
{
    int dim = this->dim();

    // Get magnitude of torque
    double tor_mag = torque.norm();
    if (tor_mag < 1E-13)
        return;

    VectorMax3d tor_dir = torque / tor_mag;

    // Torque matrix if torque was pointing in Z
    // integral(f(θ)r^T(θ))dθ
    MatrixMax3d S = MatrixMax3d::Zero(dim, dim);
    S(0, 1) = -tor_mag / 2.0;
    S(1, 0) = tor_mag / 2.0;

    // U is torque matrix when torque is pointing the desired direction
    MatrixMax3d U = MatrixMax3d::Zero(dim, dim);

    // For 3D case we need to deal with the 3D orientation of the torque
    if (dim == 3) {
        // Get rotation matrix from frame where torque is pointing in Z-axis
        // to the body frame
        // We'll do this using spherical coordinates then reorthogonalizing for
        // accuracy
        double theta = atan2(tor_dir[0], tor_dir.tail(2).norm());
        double phi = atan2(-tor_dir[1], tor_dir[2]);
        Eigen::AngleAxisd int_x_rot(phi, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd int_y_rot(theta, Eigen::Vector3d::UnitY());
        Eigen::Matrix3d R_tor_to_b = (int_x_rot * int_y_rot).toRotationMatrix();

        // Reset last column to torque direction and reorthogonalize
        R_tor_to_b.col(2) = tor_dir;
        R_tor_to_b.col(0) = R_tor_to_b.col(1).cross(R_tor_to_b.col(2));
        R_tor_to_b.col(1) = R_tor_to_b.col(2).cross(R_tor_to_b.col(0));

        U = R_tor_to_b * S * R_tor_to_b.transpose();
    } else if (dim == 2) {
        Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
        // Get sign of torque
        double sign = (0 < torque[0]) - (torque[0] < 0);
        R(0, 0) = 1.0;
        R(1, 1) = sign;
        U = R * S * R.transpose();
    }

    MatrixMax12d jac = MatrixMax12d::Zero((dim + 1) * dim, (dim + 1) * dim);
    for (int i = 1; i < dim + 1; i++) {
        jac.block(i * dim, i * dim, dim, dim) = U.transpose();
    }

    this->follower_force_jacobian += jac;
}

// Affine body broad-phase
void RigidBody::compute_spatial_rotation_bbox(
    const PoseD& pose_t0,
    const PoseD& pose_t1,
    VectorMax3d& box_min,
    VectorMax3d& box_max) const
{
    box_min = pose_t0.position.cwiseMin(pose_t1.position).array() - r_max;
    box_max = pose_t0.position.cwiseMax(pose_t1.position).array() + r_max;
}

} // namespace ipc::rigid
