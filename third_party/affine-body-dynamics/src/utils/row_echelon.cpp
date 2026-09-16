#include "row_echelon.hpp"
#include <iostream>

namespace ipc {

void row_echelon(Eigen::MatrixXd& mat){
    // Get the dimensions for convenience
    int m = mat.rows();
    int n = mat.cols();

    // First we normalise each row
    // To minimise likelihood of floating point errors
    for (int c_r = 0; c_r < m; c_r++){
        mat.row(c_r) /= mat.row(c_r).norm();
    }

    // Loop over each row
    int curr_col = 0;
    for (int curr_row = 0; curr_row < m; curr_row++){

        // Find row  with largest value at curr_column to be the new pivot
        Eigen::Index largest_row_id = curr_row;
        Eigen::VectorXd pivot_col = mat.col(curr_col).tail(m - curr_row);
        pivot_col.cwiseAbs().maxCoeff(&largest_row_id);
        largest_row_id += curr_row;

        // If all rows under pivot col is zero, check if pivot is in next col
        if (abs(mat(largest_row_id, curr_col)) < std::numeric_limits<double>::epsilon()){
            // Stop curr_row from moving down
            curr_row--;
            // Move curr_column to right
            curr_col++;
            if (curr_col + 1 >= n) return;
            continue;
        }

        // Otherwise, swap the curr_row with the row with biggest value
        if (curr_row != largest_row_id){
            mat.row(curr_row).swap(mat.row(largest_row_id));
        }

        // Now zero out the lower part of pivot column
        for (int lower_r = curr_row + 1; lower_r < m; lower_r++){
            if (abs(mat(lower_r, curr_col)) > std::numeric_limits<double>::epsilon()){
                double frac = mat(lower_r, curr_col) / mat(curr_row, curr_col);
                mat.row(lower_r) = mat.row(lower_r) - frac * mat.row(curr_row);
            }
        }
        // Ensure we zero out the column under pivot (not including pivot)
        mat.col(curr_col).tail(m - curr_row - 1).setZero();

        // Move to next column
        curr_col++;
        if (curr_col + 1 >= n) return;
    }

}

std::vector<int> find_pivot_cols(const Eigen::MatrixXd& row_echelon){
    std::vector<int> piv_cols;
    piv_cols.reserve(row_echelon.rows());

    for (int r = 0; r < row_echelon.rows(); r++){
        for (int c = 0; c < row_echelon.cols(); c++){
            // Found non-zero column in row
            if (abs(row_echelon(r, c)) > std::numeric_limits<double>::epsilon()){
                piv_cols.push_back(c);
                break;
            }
        }
    }

    return piv_cols;
}

} // namespace ipc