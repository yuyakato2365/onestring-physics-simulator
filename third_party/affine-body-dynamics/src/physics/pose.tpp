#include "pose.hpp"

#include <typeinfo> // operator typeid

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <tbb/parallel_for.h>

// #include <autodiff/autodiff.h>
#include <logger.hpp>
#include <profiler.hpp>
#include <utils/is_zero.hpp>
#include <utils/not_implemented_error.hpp>
#include <utils/sinc.hpp>
#include <utils/type_name.hpp>

namespace ipc::rigid {

template <typename T>
Pose<T>::Pose()
    : position()
    , transform()
{
}

template <typename T>
Pose<T>::Pose(const VectorMax3<T>& position, const MatrixMax3<T>& transformation)
    : position(position)
    , transform(transformation)
{
}

template <typename T>
Pose<T> Pose<T>::FromRigidPose(const VectorMax3<T>& position, const VectorMax3<T>& rotation)
{
    return Pose<T>(position, construct_rotation_matrix(rotation));
}

template <typename T>
Pose<T> Pose<T>::FromRigidVelocity(const VectorMax3<T>& linear_velocity, const VectorMax3<T>& angular_velocity, const VectorMax3<T>& rotation)
{
    // Note: we will calculate angular velocity tensor in 3D for both 2D and 3D
    // Then for the 2D case, we will resize the final tensor

    // Linear velocity is in world frame
    // Angular velocity is in world frame
    // Rotation represents a rotation of a vector in body frame to world frame
    int dim = linear_velocity.size();
    // Rdot^world_body = [w_world]_x * R^world_body
    // https://arxiv.org/pdf/1609.06088.pdf
    MatrixMax3<T> w_x(3, 3);

    MatrixMax3<T> angular_vel_tensor(3, 3);

    if (dim == 3){
        w_x <<   0.0                , -angular_velocity(2),  angular_velocity(1),
                angular_velocity(2),  0.0                , -angular_velocity(0),
                -angular_velocity(1),  angular_velocity(0),  0.0                ;

        angular_vel_tensor = w_x * construct_rotation_matrix(rotation);
    }
    else {
        w_x <<   0.0                , -angular_velocity(0),  0.0,
                 angular_velocity(0),  0.0                ,  0.0,
                 0.0                ,  0.0                ,  0.0;

        VectorMax3d rot_vec;
        rot_vec.resize(3);
        rot_vec << 0.0, 0.0, rotation(0);
        angular_vel_tensor = w_x * construct_rotation_matrix(rot_vec);
    }

    angular_vel_tensor.resize(dim, dim);

    return Pose<T>(linear_velocity, angular_vel_tensor);
}

template <typename T> Pose<T>::Pose(const VectorMax12<T>& dof)
{
    if (dof.size() == dim_to_ndof(2)) {
        position = dof.head(dim_to_pos_ndof(2));
        transform = dof.tail(dim_to_trans_ndof(2)).reshaped(2, 2).transpose();
    } else if (dof.size() == dim_to_ndof(3)) {
        position = dof.head(dim_to_pos_ndof(3));
        transform = dof.tail(dim_to_trans_ndof(3)).reshaped(3, 3).transpose();
    } else {
        throw NotImplementedError("Unknown pose conversion for given ndof");
    }
}

template <typename T>
Pose<T>::Pose(const T& x, const T& y, const T& theta)
    : Pose(Vector2<T>(x, y), Matrix2<T>::Identity())
{
    VectorMax3<T> r;
    r.resize(1);
    r[0] = theta;
    transform = construct_rotation_matrix(r);
}

template <typename T>
Pose<T>::Pose(
    const T& x,
    const T& y,
    const T& z,
    const T& theta_x,
    const T& theta_y,
    const T& theta_z)
    : Pose(Vector3<T>(x, y, z), Vector3<T>(theta_x, theta_y, theta_z))
{
}

template <typename T> Pose<T> Pose<T>::Zero(int dim)
{
    assert(dim == 2 || dim == 3);
    return Pose(
        VectorX<T>::Zero(Pose<T>::dim_to_pos_ndof(dim)),
        MatrixX<T>::Zero(Pose<T>::dim_to_pos_ndof(dim),(Pose<T>::dim_to_pos_ndof(dim))));
}

template <typename T> Pose<T> Pose<T>::Eye(int dim)
{
    assert(dim == 2 || dim == 3);
    return Pose(
        VectorX<T>::Zero(Pose<T>::dim_to_pos_ndof(dim)),
        MatrixX<T>::Identity(Pose<T>::dim_to_pos_ndof(dim),(Pose<T>::dim_to_pos_ndof(dim))));
}

template <typename T>
Poses<T> Pose<T>::dofs_to_poses(const VectorX<T>& dofs, int dim)
{
    int ndof = dim_to_ndof(dim);
    int num_poses = dofs.size() / ndof;
    assert(dofs.size() % ndof == 0);
    Poses<T> poses;
    poses.reserve(num_poses);
    for (int i = 0; i < num_poses; i++) {
        poses.emplace_back(dofs.segment(i * ndof, ndof));
    }
    return poses;
}

template <typename T> VectorX<T> Pose<T>::poses_to_dofs(const Poses<T>& poses)
{
    const int ndof = poses.size() ? poses[0].ndof() : 0;
    VectorX<T> dofs(poses.size() * ndof);
    for (size_t i = 0; i < poses.size(); i++) {
        assert(poses[i].ndof() == ndof);
        dofs.segment(i * ndof, ndof) = poses[i].dof();
    }
    return dofs;
}

template <typename T> VectorMax12<T> Pose<T>::dof() const
{
    VectorMax12<T> pose_dof(ndof());
    pose_dof.head(pos_ndof()) = position;
    // We transpose the transform matrix first to have row-major order when reshaped
    pose_dof.tail(trans_ndof()) = transform.transpose().reshaped();
    return pose_dof;
}

template <typename T> void Pose<T>::set_rotation(const VectorMax3<T>& rotation_vector){
    this->transform = construct_rotation_matrix(rotation_vector);
}

template <typename T> void Pose<T>::set_rotation(T angle){
    assert(dim() == 2);
    VectorMax3d rot(1);
    rot << angle;
    this->transform = construct_rotation_matrix(rot);
}

template <typename T> void Pose<T>::change_basis(const MatrixMax3<T>& rotation_mat){
    transform = transform * rotation_mat;
}

template <typename T> void Pose<T>::set_zero(){
    position.setZero();
    transform.setZero();
    // transform.setIdentity(); 
}

template <typename T> void Pose<T>::set_eye(){
    position.setZero();
    transform.setZero();
    transform.setIdentity(); 
}

template <typename T> Eigen::Quaternion<T> Pose<T>::construct_quaternion() const
{
    return construct_quaternion_from_transform(transform);
}

template <typename T>
Pose<T> Pose<T>::interpolate(const Pose<T>& pose0, const Pose<T>& pose1, T t)
{
    assert(pose0.dim() == pose1.dim());
    return Pose<T>(
        (pose1.position - pose0.position) * t + pose0.position,
        (pose1.transform - pose0.transform) * t + pose0.transform);
}

template <typename T> bool Pose<T>::operator==(const Pose<T>& other) const
{
    return this->position == other.position && this->transform == other.transform;
}

template <typename T> Pose<T>& Pose<T>::operator*=(const T& x)
{
    this->position *= x;
    this->transform *= x;
    return *this;
}

template <typename T> Pose<T> Pose<T>::operator/(const T& x) const
{
    return Pose<T>(this->position / x, this->transform / x);
}

template <typename T> Pose<T> Pose<T>::operator-(const Pose<T>& other) const
{
    int d = this->dim();
    int trans_d = this->dim_to_trans_ndof(d);
    return Pose<T>(this->position - other.position, this->transform - other.transform);
}

template <typename T> Pose<T> Pose<T>::operator+(const Pose<T>& other) const
{
    int d = this->dim();
    int trans_d = this->dim_to_trans_ndof(d);
    return Pose<T>(this->position + other.position, this->transform + other.transform);
}


template <typename T> Pose<T>& Pose<T>::operator+=(const VectorMax12<T>& vec)
{
    int d = this->dim();
    int trans_d = this->dim_to_trans_ndof(d);
    this->position += vec.head(d);
    this->transform += vec.tail(trans_d).reshaped(d,d).transpose();
    return *this;
}

template <typename T> Pose<T>& Pose<T>::operator=(const VectorMax12<T>& vec)
{
    int d = this->dim();
    int trans_d = this->dim_to_trans_ndof(d);
    this->position = vec.head(d);
    this->transform = vec.tail(trans_d).reshaped(d,d).transpose();
    return *this;
}

///////////////////////////////////////////////////////////////////////////
// Operations on vector of Poses

template <typename T>
Poses<T> interpolate(const Poses<T>& poses0, const Poses<T>& poses1, T t)
{    
    Poses<T> poses(poses0.size());
    for (size_t i = 0; i < poses.size(); i++) {
        poses[i] = Pose<T>::interpolate(poses0[i], poses1[i], t);
    } 
    return poses;
}

template <typename T> Poses<T> operator*(const Poses<T>& poses, const T& x)
{
    Poses<T> product = poses;
    for (size_t i = 0; i < product.size(); i++) {
        product[i] *= x;
    }
    return product;
}

template <typename T, typename U> Poses<T> cast(const Poses<U>& poses)
{
    Poses<T> poses_T;
    poses_T.reserve(poses_T.size());
    for (int i = 0; i < poses.size(); i++) {
        poses_T.push_back(poses[i].template cast<T>());
    }
    return poses_T;
}

template <typename T>
MatrixMax3<T> construct_rotation_matrix(const VectorMax3<T>& r)
{
    if (r.size() == 1) {
        return Eigen::Rotation2D<T>(r(0)).toRotationMatrix();
    } else {
        assert(r.size() == 3);
        T sinc_angle = sinc_normx(r);
        T sinc_half_angle = sinc_normx((r / T(2.0)).eval());
        Matrix3<T> K = Hat(r);
        Matrix3<T> K2 = K * K;
        Matrix3<T> R =
            sinc_angle * K + 0.5 * sinc_half_angle * sinc_half_angle * K2;
        R.diagonal().array() += T(1.0);
        return R;
    }
}

template <typename Derived, typename T>
Eigen::Quaternion<T> construct_quaternion(const Eigen::MatrixBase<Derived>& r)
{
    assert(r.size() == 3 && (r.rows() == 3 || r.cols() == 3));
    T angle = r.norm();
    if (angle == 0) {
        return Eigen::Quaternion<T>::Identity();
    }
    return Eigen::Quaternion<T>(Eigen::AngleAxis<T>(angle, r / angle));
}

template <typename T>
Eigen::Quaternion<T> construct_quaternion_from_transform(const MatrixMax3<T>& transform)
{
    Eigen::JacobiSVD<MatrixMax3<T>> svd(transform, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixX<T> U = svd.matrixU();
    Eigen::MatrixX<T> V = svd.matrixV();
    Eigen::Matrix3<T> rot = U * V.transpose();    
    return Eigen::Quaternion<T>(rot);
}

template <typename T> Matrix3<T> rotate_to_z(Vector3<T> n)
{
    if (n.norm() == T(0)) {
        return Matrix3<T>::Identity();
    }
    return Eigen::Quaternion<T>::FromTwoVectors(n, Vector3<T>::UnitZ())
        .toRotationMatrix();
}

template <typename T> Matrix3<T> rotate_around_z(const T& theta)
{
    Matrix3<T> R;
    R.row(0) << cos(theta), -sin(theta), T(0);
    R.row(1) << sin(theta), cos(theta), T(0);
    R.row(2) << T(0), T(0), T(1);
    return R;
}

} // namespace ipc::rigid
