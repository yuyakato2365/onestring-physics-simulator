#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <utils/eigen_ext.hpp>

namespace ipc::rigid {

/// @brief Compute the total mass, center of mass, and moment of intertia
void compute_mass_properties(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& facets,
    double& total_mass,
    VectorMax3d& center_of_mass,
    MatrixMax3d& moment_of_inertia);

/// @brief Compute the 2D total mass, center of mass, and moment of intertia
void compute_mass_properties_2D(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& edges,
    double& mass,
    VectorMax3d& center,
    MatrixMax3d& intertia);

/// @brief Compute the 3D total mass, center of mass, and moment of intertia
void compute_mass_properties_3D(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    double& mass,
    VectorMax3d& center,
    MatrixMax3d& intertia);

/// @brief Compute the 3D mass moments
//  Integrals of:  x^2, y^2, z^2, xy, yz, zx  over entire volume of body
void compute_mass_moments_3D(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    Eigen::Matrix<double, 6, 1>& moments);

/// @brief Compute the 2D mass moments
//  Integrals of:  x^2, y^2, xy over entire area of the 2D body
void compute_mass_moments_2D(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& edges,
    Eigen::Matrix<double, 3, 1>& moments);

/// @brief Compute the 3D mass moments for bodies with no faces
//  Integrals of:  x^2 dm, y^2 dm, z^2 dm, xy dm, yz dm, zx dm over entire body
void compute_mass_moments_3D_no_faces(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& edges,
    Eigen::Matrix<double, 6, 1>& moments);

/// @brief Find the orientation of the principle axes of a 3D body
// R0 takes a vector in principal axes basis to the input basis
void compute_principle_axes_3D(
    const MatrixMax3d& I,
    MatrixMax3d& R0);

/// @brief Find the orientation of the principle axes of a 2D body
// Note that as input, we consider 2D body in 3D space but with zero thickness along Z axis
// Thus I is a 3x3 Matrix, but R0 is a 2x2 matrix as we ignore the Z axis
// R0 takes a vector in principal axes basis to the input basis
void compute_principle_axes_2D(
    const MatrixMax3d& I,
    MatrixMax3d& R0);

/// @brief Construct the sparse mass matrix for the given mesh (V, E).
void construct_mass_matrix(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& facets,
    Eigen::SparseMatrix<double>& mass_matrix);

/// @brief Computes the total mass for the given mesh
double compute_total_mass(
    const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& facets);
/// @brief Computes the total mass from the mass matrix
double compute_total_mass(const Eigen::SparseMatrix<double>& mass_matrix);

VectorMax3d compute_center_of_mass(
    const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& facets);
VectorMax3d compute_center_of_mass(
    const Eigen::MatrixXd& vertices,
    const Eigen::SparseMatrix<double>& mass_matrix);

/**
 * @brief Computes the moment of intertia
 *
 * Assumes vertices are given in body space (i.e centered of mass at 0,0).
 */
MatrixMax3d compute_moment_of_inertia(
    const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& facets);
MatrixMax3d compute_moment_of_inertia(
    const Eigen::MatrixXd& vertices,
    const Eigen::SparseMatrix<double>& mass_matrix);

} // namespace ipc::rigid
