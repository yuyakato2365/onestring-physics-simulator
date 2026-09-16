#pragma once
#include "rigid_body.hpp"

#include <Eigen/Geometry>

#include <autodiff/autodiff_types.hpp>
#include <utils/not_implemented_error.hpp>

namespace ipc::rigid {

template <typename T>
MatrixX<T>
RigidBody::world_vertices(const MatrixMax3<T>& A, const VectorMax3<T>& p) const
{
    return (vertices * A.transpose()).rowwise() + p.transpose();
}

template <typename T>
VectorMax3<T> RigidBody::world_vertex(
    const MatrixMax3<T>& A, const VectorMax3<T>& p, const int vertex_idx) const
{
    // compute X[i] = A * x̄ + p
    return (vertices.row(vertex_idx) * A.transpose()) + p.transpose();
}

template <typename T>
MatrixMax<T, 3, 12> RigidBody::J_matrix(VectorMax3<T> x_bar){
    MatrixMax<T, 3, 12> J;
    int dim = x_bar.size();
    J.resize(dim, ndof(dim));
    J.setZero();
    J.diagonal().setConstant(1.0);

    for (int i = 0; i < dim; i++){
        J.block(i, (i+1) * dim, 1, dim) = x_bar.transpose();
    }

    return J;
}

template <typename T, typename MatType, int Dim, int Ndof>
void RigidBody::J_matrix(const VectorMax3<T>& x_bar, Eigen::Block<MatType, Dim, Ndof> sub_matrix){
    sub_matrix.setConstant(0.0);
    sub_matrix.diagonal().setConstant(1.0);
    for (int b = 0; b < Dim; b++){
        for (int i = 0; i < Dim; i++){
            sub_matrix(b, (b + 1) * Dim + i) = x_bar(i);
        }
    }
}

template <typename MatType1, typename MatType2, int Dim, int Ndof>
void RigidBody::J_matrix(const Eigen::Block<MatType1, 1, -1> x_bar, Eigen::Block<MatType2, Dim, Ndof> sub_matrix){
    sub_matrix.setConstant(0.0);
    sub_matrix.diagonal().setConstant(1.0);
    for (int b = 0; b < Dim; b++){
        for (int i = 0; i < Dim; i++){
            sub_matrix(b, (b + 1) * Dim + i) = x_bar(i);
        }
    }
}


} // namespace ipc::rigid
