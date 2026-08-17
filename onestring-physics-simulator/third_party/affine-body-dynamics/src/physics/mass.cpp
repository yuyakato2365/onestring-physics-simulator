#include "mass.hpp"

#include <igl/massmatrix.h>
#include <ipc/utils/eigen_ext.hpp>
#include <logger.hpp>
#include <utils/eigen_ext.hpp>
#include <utils/not_implemented_error.hpp>

namespace ipc::rigid {

void compute_mass_properties(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& facets,
    double& mass,
    VectorMax3d& center,
    MatrixMax3d& inertia)
{
    if (vertices.cols() == 2) {
        compute_mass_properties_2D(vertices, facets, mass, center, inertia);
    } else {
        // TODO: Fix this temporary version
        try {
            compute_mass_properties_3D(vertices, facets, mass, center, inertia);
        } catch (NotImplementedError err) {
            compute_mass_properties_2D(vertices, facets, mass, center, inertia);
        }
    }
}

void compute_mass_properties_2D(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& edges,
    double& mass,
    VectorMax3d& center,
    MatrixMax3d& inertia)
{
    Eigen::SparseMatrix<double> M;
    construct_mass_matrix(vertices, edges, M);
    mass = compute_total_mass(M);
    center = compute_center_of_mass(vertices, M);
    inertia = compute_moment_of_inertia(vertices, M);
    // std::cout << "mass: " << mass << ", inertia : " << inertia << std::endl;
}

void mass_integrals_3D(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    Eigen::Matrix<double, 10, 1>& integral)
{
    integral.setZero();

    for (int i = 0; i < faces.rows(); i++) {
        // Get vertices of triangle i.
        const Eigen::Vector3d& v0 = vertices.row(faces(i, 0));
        const Eigen::Vector3d& v1 = vertices.row(faces(i, 1));
        const Eigen::Vector3d& v2 = vertices.row(faces(i, 2));

        // Get cross product of edges and normal vector.
        const Eigen::Vector3d& V1mV0 = v1 - v0;
        const Eigen::Vector3d& V2mV0 = v2 - v0;
        const Eigen::Vector3d& N = V1mV0.cross(V2mV0);

        // Compute integral terms.
        double tmp0, tmp1, tmp2;
        double f1x, f2x, f3x, g0x, g1x, g2x;
        tmp0 = v0.x() + v1.x();
        f1x = tmp0 + v2.x();
        tmp1 = v0.x() * v0.x();
        tmp2 = tmp1 + v1.x() * tmp0;
        f2x = tmp2 + v2.x() * f1x;
        f3x = v0.x() * tmp1 + v1.x() * tmp2 + v2.x() * f2x;
        g0x = f2x + v0.x() * (f1x + v0.x());
        g1x = f2x + v1.x() * (f1x + v1.x());
        g2x = f2x + v2.x() * (f1x + v2.x());

        double f1y, f2y, f3y, g0y, g1y, g2y;
        tmp0 = v0.y() + v1.y();
        f1y = tmp0 + v2.y();
        tmp1 = v0.y() * v0.y();
        tmp2 = tmp1 + v1.y() * tmp0;
        f2y = tmp2 + v2.y() * f1y;
        f3y = v0.y() * tmp1 + v1.y() * tmp2 + v2.y() * f2y;
        g0y = f2y + v0.y() * (f1y + v0.y());
        g1y = f2y + v1.y() * (f1y + v1.y());
        g2y = f2y + v2.y() * (f1y + v2.y());

        double f1z, f2z, f3z, g0z, g1z, g2z;
        tmp0 = v0.z() + v1.z();
        f1z = tmp0 + v2.z();
        tmp1 = v0.z() * v0.z();
        tmp2 = tmp1 + v1.z() * tmp0;
        f2z = tmp2 + v2.z() * f1z;
        f3z = v0.z() * tmp1 + v1.z() * tmp2 + v2.z() * f2z;
        g0z = f2z + v0.z() * (f1z + v0.z());
        g1z = f2z + v1.z() * (f1z + v1.z());
        g2z = f2z + v2.z() * (f1z + v2.z());

        // Update integrals.
        integral[0] += N.x() * f1x;
        integral[1] += N.x() * f2x;
        integral[2] += N.y() * f2y;
        integral[3] += N.z() * f2z;
        integral[4] += N.x() * f3x;
        integral[5] += N.y() * f3y;
        integral[6] += N.z() * f3z;
        integral[7] += N.x() * (v0.y() * g0x + v1.y() * g1x + v2.y() * g2x);
        integral[8] += N.y() * (v0.z() * g0y + v1.z() * g1y + v2.z() * g2y);
        integral[9] += N.z() * (v0.x() * g0z + v1.x() * g1z + v2.x() * g2z);
    }

    integral[0] /= 6;
    integral[1] /= 24;
    integral[2] /= 24;
    integral[3] /= 24;
    integral[4] /= 60;
    integral[5] /= 60;
    integral[6] /= 60;
    integral[7] /= 120;
    integral[8] /= 120;
    integral[9] /= 120;
}

// Based on ChTriangleMeshConnected.cpp::ComputeMassProperties from Chrono:
// Copyright (c) 2016, Project Chrono Development Team
// All rights reserved.
// https://github.com/projectchrono/chrono/blob/develop/LICENSE
//
// This requires the mesh to be closed, watertight, with proper triangle
// orientation.
void compute_mass_properties_3D(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    double& mass,
    VectorMax3d& center,
    MatrixMax3d& inertia)
{
    if (faces.size() == 0 || faces.cols() != 3) {
        throw NotImplementedError("compute_mass_properties_3D() not "
                                  "implemented for faceless meshes!");
    }
    assert(vertices.cols() == 3);
    assert(faces.cols() == 3);

    // order:  1, x, y, z, x^2, y^2, z^2, xy, yz, zx
    Eigen::Matrix<double, 10, 1> integral =
        Eigen::Matrix<double, 10, 1>::Zero();

    mass_integrals_3D(vertices, faces, integral);

    // mass
    mass = integral[0];
    if (mass <= 0 || !std::isfinite(mass)) {
        // mass = 0.0;
        throw NotImplementedError(
            "3D mass computation only works for closed meshes!");
    }
    assert(mass > 0);

    // center of mass
    center = Eigen::Vector3d(integral[1], integral[2], integral[3]) / mass;

    // inertia relative to world origin
    inertia.resize(3, 3);
    inertia(0, 0) = integral[5] + integral[6];
    inertia(0, 1) = -integral[7];
    inertia(0, 2) = -integral[9];
    inertia(1, 0) = inertia(0, 1);
    inertia(1, 1) = integral[4] + integral[6];
    inertia(1, 2) = -integral[8];
    inertia(2, 0) = inertia(0, 2);
    inertia(2, 1) = inertia(1, 2);
    inertia(2, 2) = integral[4] + integral[5];

    // inertia relative to center of mass
    inertia(0, 0) -= mass * (center.y() * center.y() + center.z() * center.z());
    inertia(0, 1) += mass * center.x() * center.y();
    inertia(0, 2) += mass * center.z() * center.x();
    inertia(1, 0) = inertia(0, 1);
    inertia(1, 1) -= mass * (center.z() * center.z() + center.x() * center.x());
    inertia(1, 2) += mass * center.y() * center.z();
    inertia(2, 0) = inertia(0, 2);
    inertia(2, 1) = inertia(1, 2);
    inertia(2, 2) -= mass * (center.x() * center.x() + center.y() * center.y());
}

/// @brief Compute the 3D second order mass moments
//  Integrals of:  x^2 dm, y^2 dm, z^2 dm, xy dm, yz dm, zx dm over entire volume of body
void compute_mass_moments_3D(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    Eigen::Matrix<double, 6, 1>& moments)
{
    if (faces.size() == 0 || faces.cols() != 3) {
        throw NotImplementedError("compute_mass_moments_3D() not "
                                  "implemented for faceless meshes!");
    }
    assert(vertices.cols() == 3);
    assert(faces.cols() == 3);

    // order:  1, x, y, z, x^2, y^2, z^2, xy, yz, zx
    Eigen::Matrix<double, 10, 1> integral =
        Eigen::Matrix<double, 10, 1>::Zero();

    mass_integrals_3D(vertices, faces, integral);

    // mass
    double mass = integral[0];
    if (mass <= 0 || !std::isfinite(mass)) {
        // mass = 0;
        // moments.resize(6, 1);
        // moments[0] = 0.0;
        // moments[1] = 0.0;
        // moments[2] = 0.0;
        // moments[3] = 0.0;
        // moments[4] = 0.0;
        // moments[5] = 0.0;
        // return;
        throw NotImplementedError(
            "3D mass computation only works for closed meshes!");
    }
    assert(mass > 0);

    // center of mass
    auto center = Eigen::Vector3d(integral[1], integral[2], integral[3]) / mass;
    
    // moments relative to world origin
    // order:  1, x, y, z, x^2, y^2, z^2, xy, yz, zx
    moments.resize(6, 1);
    moments[0] = integral[4];
    moments[1] = integral[5];
    moments[2] = integral[6];
    moments[3] = integral[7];
    moments[4] = integral[8];
    moments[5] = integral[9];

    // moments relative to center of mass
    moments[0] -= mass * (center.x() * center.x());
    moments[1] -= mass * (center.y() * center.y());
    moments[2] -= mass * (center.z() * center.z());
    moments[3] -= mass * center.x() * center.y();
    moments[4] -= mass * center.y() * center.z();
    moments[5] -= mass * center.z() * center.z();
}

/// @brief Find the orientation of the principle axes of a 3D body
// R0 takes a vector in principal axes basis to the input basis
void compute_principle_axes_3D(
    const MatrixMax3d& I,
    MatrixMax3d& R0)
{

    // Get principal moment via eigen solve
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
    double threshold = I.lpNorm<Eigen::Infinity>() * 1e-16;
    Eigen::Matrix3d I_new = (threshold < I.array().abs()).select(I, 0.0);
    es.compute(I_new);
    if (es.info() != Eigen::Success) {
        spdlog::error("Eigen decompostion of the inertia tensor failed!");
    }
    R0 = es.eigenvectors();
    // Ensure that we have an orientation preserving transform
    if (R0.determinant() < 0.0) {
        R0.col(0) *= -1.0;
    }
    assert(R0.isUnitary(1e-9));
    assert(fabs(R0.determinant() - 1.0) <= 1.0e-9);
}

/// @brief Find the orientation of the principle axes of a 2D body
// R0 takes a vector in principal axes basis to the input basis
void compute_principle_axes_2D(
    const MatrixMax3d& I,
    MatrixMax3d& R0)
{
    // Check that 2D body is parallel to the xy-plane
    assert(I.topRightCorner(2,1).isZero());
    assert(I.bottomLeftCorner(1,2).isZero());

    compute_principle_axes_3D(I, R0);

    // Find which column has the z-axis
    Eigen::Vector3d z_vec = {0.0, 0.0, 1.0};
    Eigen::Vector3d cols_dot_z_mag = R0.transpose() * z_vec;
    cols_dot_z_mag = cols_dot_z_mag.cwiseAbs();

    int col_id;
    double max_val = cols_dot_z_mag.maxCoeff(&col_id);
    // Check that it does point in Z-direction
    assert(abs(max_val - 1.0) < 1e-10 && "2D body is not parallel to XY-plane");

    std::vector<int> princ_col_ids = {0,1,2};
    princ_col_ids.erase(princ_col_ids.begin() + col_id);

    // Check other 2 axes are perpendicular to Z
    assert(cols_dot_z_mag(princ_col_ids).isZero() && "2D body is not parallel to XY-plane");
    
    MatrixMax3d temp_R0 = R0;

    // Remove the axis that points in the Z-direction
    // and truncate the other 2 vector (they should have zero Z component anyway)
    // To get a 2x2 rotation matrix
    R0.conservativeResize(2,2);
    R0.setZero();
    R0 = temp_R0(Eigen::seq(0,1), princ_col_ids);
}

/// @brief Compute the 2D mass moments
//  Integrals of:  x^2 dm, y^2 dm, xy dm 
void compute_mass_moments_2D(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& facets,
    Eigen::Matrix<double, 3, 1>& moments)
{
    Eigen::SparseMatrix<double> M;
    construct_mass_matrix(vertices, facets, M);

    // ∑ mᵢ * xᵢ ⋅ xᵢ
    // ∑ mᵢ * yᵢ ⋅ yᵢ
    // ∑ mᵢ * xᵢ ⋅ yᵢ this should be zero when in principal coordinates
    // This only work with 2D objects in 2D ambient space
    assert(vertices.cols() == 2);
    Eigen::MatrixXd squares = vertices.array().square().matrix();
    moments.head(2) = (M * squares).colwise().sum();
    moments[2] = (M * vertices.col(0).cwiseProduct(vertices.col(1))).sum();
}

/// @brief Compute the 3D mass moments for bodies with no faces
//  Integrals of:  x^2 dm, y^2 dm, z^2 dm, xy dm, yz dm, zx dm over entire body
void compute_mass_moments_3D_no_faces(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& edges,
    Eigen::Matrix<double, 6, 1>& moments)
{
    Eigen::SparseMatrix<double> M;
    construct_mass_matrix(vertices, edges, M);

    Eigen::VectorXd vertex_masses = M.diagonal();

    moments.setZero();
    for (long i = 0; i < vertices.rows(); i++) {
        const VectorMax3d& v = vertices.row(i);
        const VectorMax3d v_sqr = v.array().pow(2);

        moments[0] += vertex_masses(i) * v_sqr(0);
        moments[0] += vertex_masses(i) * v_sqr(1);
        moments[0] += vertex_masses(i) * v_sqr(2);
        moments[0] += vertex_masses(i) * v(0)*v(1);
        moments[0] += vertex_masses(i) * v(1)*v(2);
        moments[0] += vertex_masses(i) * v(2)*v(0);
    }
}

// Construct the sparse mass matrix for the given mesh (V, F).
void construct_mass_matrix(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& facets,
    Eigen::SparseMatrix<double>& mass_matrix)
{
    if (vertices.cols() == 2 || facets.cols() == 2) {
        assert(facets.cols() == 2);
        Eigen::VectorXd vertex_masses = Eigen::VectorXd::Zero(vertices.rows());
        for (long i = 0; i < facets.rows(); i++) {
            double edge_length =
                (vertices.row(facets(i, 1)) - vertices.row(facets(i, 0)))
                    .norm();
            // Add vornoi areas to the vertex weight
            vertex_masses(facets(i, 0)) += edge_length / 2;
            vertex_masses(facets(i, 1)) += edge_length / 2;
        }
        mass_matrix = SparseDiagonal<double>(vertex_masses);
    } else if (facets.cols() == 3) {
        assert(vertices.cols() == 3); // Only use triangles in 3D
        igl::massmatrix(
            vertices, facets, igl::MassMatrixType::MASSMATRIX_TYPE_VORONOI,
            mass_matrix);
    } else {
        // Probably a point cloud
        mass_matrix.resize(vertices.rows(), vertices.rows());
        mass_matrix.setIdentity();
    }
}

double compute_total_mass(
    const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& facets)
{
    Eigen::SparseMatrix<double> M;
    construct_mass_matrix(vertices, facets, M);
    return compute_total_mass(M);
}

double compute_total_mass(const Eigen::SparseMatrix<double>& mass_matrix)
{
    return mass_matrix.sum();
}

VectorMax3d compute_center_of_mass(
    const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& facets)
{
    Eigen::SparseMatrix<double> M;
    construct_mass_matrix(vertices, facets, M);
    return compute_center_of_mass(vertices, M);
}

VectorMax3d compute_center_of_mass(
    const Eigen::MatrixXd& vertices,
    const Eigen::SparseMatrix<double>& mass_matrix)
{
    double total_mass = mass_matrix.sum();
    if (total_mass == 0) {
        return VectorMax3d::Zero(vertices.cols());
    }
    return (mass_matrix * vertices).colwise().sum() / mass_matrix.sum();
}

MatrixMax3d compute_moment_of_inertia(
    const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& facets)
{
    MatrixMax3d inertia;
    // NOTE: this assumes center of mass is at 0,0
    if (vertices.cols() == 2) {
        Eigen::SparseMatrix<double> M;
        construct_mass_matrix(vertices, facets, M);
        inertia = compute_moment_of_inertia(vertices, M);
    } else {
        double mass;
        VectorMax3d center;
        compute_mass_properties_3D(vertices, facets, mass, center, inertia);
    }
    return inertia;
}

MatrixMax3d compute_moment_of_inertia(
    const Eigen::MatrixXd& vertices,
    const Eigen::SparseMatrix<double>& mass_matrix)
{
    int dim = vertices.cols();
    // https://i.ytimg.com/vi/9jOrDufoO50/hqdefault.jpg
    Eigen::VectorXd vertex_masses = mass_matrix.diagonal();

    Eigen::Matrix<double, 3, 3> I = Eigen::Matrix<double, 3, 3>::Zero();
    for (long i = 0; i < vertices.rows(); i++) {
        const VectorMax3d& v = vertices.row(i);
        const VectorMax3d v_sqr = v.array().pow(2);

        // If only 2 dims provided, assume third dim is zero
        double v_2, v_2_sqr;
        if (dim > 2){
            v_2 = v(2);
            v_2_sqr = v_sqr(2);
        }
        else{
            v_2 = 0.0;
            v_2_sqr = 0.0;
        }

        I(0, 0) += vertex_masses(i) * (v_sqr(1) + v_2_sqr);
        I(0, 1) += -vertex_masses(i) * v(0) * v(1);
        I(0, 2) += -vertex_masses(i) * v(0) * v_2;
        I(1, 1) += vertex_masses(i) * (v_sqr(0) + v_2_sqr);
        I(1, 2) += -vertex_masses(i) * v(1) * v_2;
        I(2, 2) += vertex_masses(i) * (v_sqr(0) + v_sqr(1));
    }
    I(1, 0) = I(0, 1);
    I(2, 0) = I(0, 2);
    I(2, 1) = I(1, 2);

    return I;
}

} // namespace ipc::rigid
