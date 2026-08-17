#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <ipc/utils/eigen_ext.hpp>

namespace ipc {

typedef Eigen::DiagonalMatrix<double, 3> DiagonalMatrix3d;

template <typename T>
Eigen::SparseMatrix<T> SparseDiagonal(const VectorX<T>& x);

template <typename T> inline Matrix2<T> Hat(T x);
template <typename T> inline Matrix3<T> Hat(Vector3<T> x);
template <typename T> inline MatrixMax3<T> Hat(VectorMax3<T> x);
/// @brief A dynamic size diagonal matrix with a fixed maximum size of 12×12
using DiagonalMatrixMax12d = Eigen::DiagonalMatrix<double, Eigen::Dynamic, 12>;
/// @brief A dynamic size matrix with a fixed maximum size of 12×1
using VectorMax12b = VectorMax12<bool>;
using Matrix3d = Eigen::Matrix<double, 3, 3>;
/// @brief A dynamic size matrix with a fixed maximum size of 24×1
template <typename T> using VectorMax24 = Vector<T, Eigen::Dynamic, 24>;
/// @brief A dynamic size matrix with a fixed maximum size of 24×1
using VectorMax24d = VectorMax24<double>;
/// @brief A dynamic size matrix with a fixed maximum size of 24x24
template <typename T> using MatrixMax24 = MatrixMax<T, 24, 24>;
/// @brief A dynamic size matrix with a fixed maximum size of 24x24
using MatrixMax24d = MatrixMax24<double>;

} // namespace ipc

#include "eigen_ext.tpp"
