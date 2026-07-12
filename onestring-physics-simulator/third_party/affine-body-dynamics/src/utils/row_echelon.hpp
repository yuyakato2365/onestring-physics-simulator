#pragma once

#include <Eigen/Core>

namespace ipc {

// Transforming a matrix to row echelon form NOT reduced row echelon. (may be done more efficiently)
void row_echelon(Eigen::MatrixXd& mat);

std::vector<int> find_pivot_cols(const Eigen::MatrixXd& row_echelon);

} // namespace ipc