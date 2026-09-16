#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include <utils/eigen_ext.hpp>

namespace ipc::rigid {

template <typename T> class Pose;
template <typename T> using Poses = std::vector<Pose<T>>;

/// @brief The position and linear transformation of an object.
template <typename T> class Pose {
public:
    Pose();
    Pose(const VectorMax3<T>& position, const MatrixMax3<T>& transformation);
    Pose(const VectorMax12<T>& dof);

    // Made into a static method instead of a constructor to avoid ambiguity since MatrixMax3 and VectorMax3 cannot be distinguised by the compiler
    static Pose<T> FromRigidPose(const VectorMax3<T>& position, const VectorMax3<T>& rotation);

    // Converts velocities in linear velocity + angular velocity vector to linear velocity + angular velocity tensor
    static Pose<T> FromRigidVelocity(const VectorMax3<T>& linear_velocity, const VectorMax3<T>& angular_velocity, const VectorMax3<T>& rotation);

    Pose(const T& x, const T& y, const T& theta);
    Pose(
        const T& x,
        const T& y,
        const T& z,
        const T& theta_x,
        const T& theta_y,
        const T& theta_z);

    static Pose<T> Zero(int dim);
    static Pose<T> Eye(int dim);

    static Poses<T> dofs_to_poses(const VectorX<T>& dofs, int dim);
    static VectorX<T> poses_to_dofs(const Poses<T>& poses);

    static int dim_to_ndof(const int dim) { return dim == 2 ? 6 : 12; }
    static int dim_to_pos_ndof(const int dim) { return dim; }
    static int dim_to_trans_ndof(const int dim)
    {
        return dim_to_ndof(dim) - dim_to_pos_ndof(dim);
    }
    int dim() const { return position.size(); }
    //@javidf: make sure that pos_ndof() and rot_ndof() are needed. Since it is a linear transformation we often do not need 
    // to separate them and they might not be required. 
    int pos_ndof() const { return position.size(); }
    int trans_ndof() const { return transform.size(); }
    int ndof() const { return pos_ndof() + trans_ndof(); }

    VectorMax12<T> dof() const;

    void set_rotation(const VectorMax3<T>& rotation_vector);
    void set_rotation(T angle);

    /// @brief Given a rotation matrix taking vectors in a new basis to the old basis,
    /// the pose is tranformed into the new basis
    void change_basis(const MatrixMax3<T>& rotation_mat);
    //@javidf: what is the usage of this? I don't get it over Pose::Zero(dim)
    //@javidf: set_zero() and set_eye() should be used on size fixed Poses. 
    void set_zero();
    void set_eye();

    MatrixMax3<T> construct_transformation_matrix() const { return transform;}
    
    Eigen::Quaternion<T> construct_quaternion() const;

    static Pose<T> interpolate(const Pose<T>& pose0, const Pose<T>& pose1, T t);

    bool operator==(const Pose<T>& other) const;

    friend Pose<T> operator*(const Pose<T>& pose, const T& x)
    {
        return Pose<T>(pose.position * x, pose.transform * x);
    }
    friend Pose<T> operator*(const T& x, const Pose<T>& pose)
    {
        return Pose<T>(x * pose.position, x * pose.transform);
    }
    inline Pose<T>& operator*=(const T& x);
    Pose<T> operator/(const T& x) const;
    Pose<T> operator-(const Pose<T>& other) const;
    Pose<T> operator+(const Pose<T>& other) const;
    inline Pose<T>& operator+=(const VectorMax12<T>& vec);
    inline Pose<T>& operator=(const VectorMax12<T>& vec);

    template <typename T1> Pose<T1> cast() const
    {
        MatrixMax3<T1> transform_casted = transform.template cast<T1>();
        return Pose<T1>(
            position.template cast<T1>(), transform_casted);
    }

    /// Position dof (either 2D or 3D)
    VectorMax3<T> position;
    /// Linear transformation dof expressed as either a 2x2 (2D case) or 3x3 matrix (3D case)
    MatrixMax3<T> transform;
};

typedef Pose<double> PoseD;
typedef Poses<double> PosesD;

template <typename T>
Poses<T> interpolate(const Poses<T>& pose0, const Poses<T>& pose1, T t);
template <typename T> Poses<T> operator*(const Poses<T>& poses, const T& x);
/// @brief Cast poses element-wise.
template <typename T, typename U> Poses<T> cast(const Poses<U>& poses);

template <typename T>
MatrixMax3<T> construct_rotation_matrix(const VectorMax3<T>& r);
template <typename Derived, typename T = typename Derived::Scalar>
Eigen::Quaternion<T> construct_quaternion(const Eigen::MatrixBase<Derived>& r);
template <typename T>
Eigen::Quaternion<T> construct_quaternion_from_transform(const MatrixMax3<T>& transform);

template <typename T> Matrix3<T> rotate_to_z(Vector3<T> n);
template <typename T> Matrix3<T> rotate_around_z(const T& theta);

} // namespace ipc::rigid

#include "pose.tpp"
