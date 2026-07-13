#pragma once

#include <deque>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <physics/pose.hpp>
#include <utils/eigen_ext.hpp>

#include <BVH.hpp>
#include <utils/mesh_selector.hpp>

namespace ipc::rigid {

enum RigidBodyType { STATIC, KINEMATIC, DYNAMIC };

NLOHMANN_JSON_SERIALIZE_ENUM(
    RigidBodyType,
    { { STATIC, "static" },
      { KINEMATIC, "kinematic" },
      { DYNAMIC, "dynamic" } });

class RigidBody {
public:
    /**
     * @brief Create rigid body with center of mass at \f$\vec{0}\f$.
     *
     * @param vertices  Vertices of the rigid body in body space
     * @param faces     Vertices pairs defining the topology of the rigid
     *                  body
     */
    RigidBody(
        const Eigen::MatrixXd& vertices,
        const Eigen::MatrixXi& edges,
        const Eigen::MatrixXi& faces,
        const PoseD& pose,
        const PoseD& velocity,
        const PoseD& force,
        const double density,
        const bool oriented,
        const int group_id,
        const RigidBodyType type = RigidBodyType::DYNAMIC,
        const double kinematic_max_time =
            std::numeric_limits<double>::infinity(),
        const std::deque<PoseD>& kinematic_poses = std::deque<PoseD>());

    // Faceless version for convienence (useful for 2D)
    RigidBody(
        const Eigen::MatrixXd& vertices,
        const Eigen::MatrixXi& edges,
        const PoseD& pose,
        const PoseD& velocity,
        const PoseD& force,
        const double density,
        const bool oriented,
        const int group_id,
        const RigidBodyType type = RigidBodyType::DYNAMIC,
        const double kinematic_max_time =
            std::numeric_limits<double>::infinity(),
        const std::deque<PoseD>& kinematic_poses = std::deque<PoseD>())
        : RigidBody(
              vertices,
              edges,
              Eigen::MatrixXi(),
              pose,
              velocity,
              force,
              density,
              oriented,
              group_id,
              type,
              kinematic_max_time,
              kinematic_poses)
    {
    }

    // --------------------------------------------------------------------
    // State Functions
    // --------------------------------------------------------------------

    enum Step { PREVIOUS_STEP = 0, CURRENT_STEP };
    /// @brief: computes vertices position for current or previous state
    Eigen::MatrixXd world_vertices(const Step step = CURRENT_STEP) const
    {
        return world_vertices(step == PREVIOUS_STEP ? pose_prev : pose);
    }
    Eigen::MatrixXd world_vertices_t0() const
    {
        return world_vertices(PREVIOUS_STEP);
    }
    Eigen::MatrixXd world_vertices_t1() const
    {
        return world_vertices(CURRENT_STEP);
    }

    Eigen::MatrixXd world_velocities() const;

    // --------------------------------------------------------------------
    // CCD Functions
    // --------------------------------------------------------------------

    /// @brief Computes vertices position for given state.
    /// @return The positions of all vertices in 'world space',
    ///         taking into account the given body's position.
    template <typename T>
    MatrixX<T>
    world_vertices(const MatrixMax3<T>& R, const VectorMax3<T>& p) const;
    template <typename T> MatrixX<T> world_vertices(const Pose<T>& _pose) const
    {
        return world_vertices<T>(
            _pose.transform, _pose.position);
    }
    template <typename T>
    MatrixX<T> world_vertices(const VectorMax6<T>& dof) const
    {
        return world_vertices(Pose<T>(dof));
    }

    template <typename T>
    VectorMax3<T> world_vertex(
        const MatrixMax3<T>& A,
        const VectorMax3<T>& p,
        const int vertex_idx) const;
    template <typename T>
    VectorMax3<T> world_vertex(const Pose<T>& _pose, const int vertex_idx) const
    {
        return world_vertex<T>(
            _pose.construct_transformation_matrix(), _pose.position, vertex_idx);
    }
    template <typename T>
    VectorMax3<T>
    world_vertex(const VectorMax6<T>& dof, const int vertex_idx) const
    {
        return world_vertex<T>(Pose<T>(dof), vertex_idx);
    }

    template <typename T>
    static MatrixMax<T, 3, 12> J_matrix(VectorMax3<T> x_bar);

    template <typename T, typename MatType, int Dim, int Ndof>
    static void J_matrix(const VectorMax3<T>& x_bar, Eigen::Block<MatType, Dim, Ndof> sub_matrix);

    template <typename MatType1, typename MatType2, int Dim, int Ndof>
    static void J_matrix(const Eigen::Block<MatType1, 1, -1> x_bar, Eigen::Block<MatType2, Dim, Ndof> sub_matrix);

    /// @warning Will not resize jac, so make sure it is large
    /// enough.
    void world_vertices_jacobian(
        long rb_v0_i,
        Eigen::MatrixXd& jac) const;

    Eigen::MatrixXd world_vertices(
        const PoseD& pose,
        long rb_v0_i,
        Eigen::MatrixXd& V) const;

    // Transforms point from input space to material space
    VectorMax3d input_to_material_point(VectorMax3d input_point) const
    {
        return R0.transpose() * input_point  + in_origin;
    }

    // Transforms point from world space to material space
    VectorMax3d material_point(VectorMax3d world_point) const
    {
        return pose.transform.transpose() * (world_point - pose.position);
    }

    // Transforms point from material space to world space
    VectorMax3d world_point(VectorMax3d material_point) const
    {
        return J_matrix(material_point) * pose.dof();
    }

    /// @warning Will not resize jac, so make sure it is large
    /// enough.
    /// Legacy method! With affine body, we store jacobian in assembler as it is constant wrt pose
    /// This method recalculates the jacobian everytime. Use the two methods above instead
    Eigen::MatrixXd world_vertices_diff(
        const PoseD& pose,
        long rb_v0_i,
        Eigen::MatrixXd& V,
        Eigen::MatrixXd& jac) const;


    double edge_length(int edge_id) const
    {
        return (vertices.row(edges(edge_id, 1))
                - vertices.row(edges(edge_id, 0)))
            .norm();
    }

    long num_vertices() const { return vertices.rows(); }
    long num_edges() const { return edges.rows(); }
    long num_faces() const { return faces.rows(); }
    long num_codim_vertices() const
    {
        return mesh_selector.num_codim_vertices();
    }
    long num_codim_edges() const { return mesh_selector.num_codim_edges(); }
    int dim() const { return vertices.cols(); }
    int ndof() const { return pose.ndof(); }
    static int ndof(int dim) { return dim == 2 ? 6 : 12; }
    int pos_ndof() const { return pose.pos_ndof(); }
    int trans_ndof() const { return pose.trans_ndof(); }
    long bvh_size() const
    {
        return num_codim_vertices() + num_codim_edges() + num_faces();
    }

    void compute_bounding_box(
        const PoseD& pose_t0,
        const PoseD& pose_t1,
        VectorMax3d& box_min,
        VectorMax3d& box_max) const;
    void compute_bounding_box(
        const PoseD& pose, VectorMax3d& box_min, VectorMax3d& box_max) const
    {
        return compute_bounding_box(pose, pose, box_min, box_max);
    }
    void compute_spatial_rotation_bbox(
        const PoseD& pose_t0,
        const PoseD& pose_t1,
        VectorMax3d& box_min,
        VectorMax3d& box_max) const;

    void convert_to_static()
    {
        type = RigidBodyType::STATIC;
        velocity = PoseD::Zero(dim());
        force = PoseD::Zero(dim());
    }

    // --------------------------------------------------------------------
    // Forces
    // --------------------------------------------------------------------
    void add_force(VectorMax3d material_point, VectorMax3d force);
    void add_follower_force(VectorMax3d material_point, VectorMax3d force);
    void add_torque(VectorMax3d torque);


    // --------------------------------------------------------------------
    // Properties
    // --------------------------------------------------------------------

    std::string name = "AffineBody";

    /// @brief Group id of this body
    int group_id;

    /// @brief Dyanmic type of rigid body
    RigidBodyType type;

    // --------------------------------------------------------------------
    // Geometry
    // --------------------------------------------------------------------
    Eigen::MatrixXd vertices; ///< Vertices positions in body space
    Eigen::MatrixXi edges;    ///< Vertices connectivity
    Eigen::MatrixXi faces;    ///< Vertices connectivity

    double average_edge_length; ///< Average edge length

    /// @brief total mass (M) of the rigid body
    double mass;
    /// @brief total volume (m^3) of the rigid body
    double volume;
    /// @brief moment of inertia measured with respect to the principal axes
    //@javidf: this is defined here but never used. Also, in RigidBody(), we define local variables
    // I, center_of_mass, etc. I wonder why not defining them as members. 
    VectorMax3d moment_of_inertia;
    /// @brief rotation from the principal axes to the input orientation
    MatrixMax3d R0;
    /// @brief position of input mesh origin expressed in principal coordinates
    VectorMax3d in_origin;
    /// @brief maximum distance from CM to a vertex
    double r_max;
    /// @brief the generalized mass matrix of the affine body
    DiagonalMatrixMax12d mass_matrix;
    // @javidf: added to make the  code runnign while transferring from rigid to ABD
    VectorMax12b is_dof_fixed;
    
    /// @brief Use edge orientation for normal in 2D restitution
    bool is_oriented;

    /// @brief Local space BVH initalized at construction
    BVH::BVH bvh;
    MeshSelector mesh_selector;

    // --------------------------------------------------------------------
    // State
    // --------------------------------------------------------------------
    /// @brief current timestep position and rotation of the center of mass
    PoseD pose;
    /// @brief previous timestep position and rotation of the center of mass
    PoseD pose_prev;

    /// @brief current timestep velocity of the center of mass
    PoseD velocity;
    /// @brief previous timestep velocity of the center of mass
    PoseD velocity_prev;

    PoseD acceleration;

    /// @brief external force acting on the body
    PoseD force;

    /// @brief Provides the follower force when matmultiplied with q (F_follower = F_follower_jac * q)
    MatrixMax12d follower_force_jacobian;

    // --------------------------------------------------------------------
    // Scripted kinematic motion
    // --------------------------------------------------------------------
    double kinematic_max_time;
    std::deque<PoseD> kinematic_poses;

protected:
    void init_bvh();
    Eigen::MatrixXd world_vertices_jacobian() const;
};

} // namespace ipc::rigid

#include "rigid_body.tpp"
