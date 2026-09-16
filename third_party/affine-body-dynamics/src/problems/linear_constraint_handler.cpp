#include "linear_constraint_handler.hpp"
#include <utils/row_echelon.hpp>
#include <io/serialize_json.hpp>
#include <logger.hpp>
#include <iostream>

namespace ipc::rigid {

bool LinearConstraintHandler::settings(const nlohmann::json& json_linear_cons, const RigidBodyAssembler& bodies)
{
    for (auto& jcon : json_linear_cons) {
        if (jcon["type"] == "pin_world"){
            add_pin_world_constraint(jcon, bodies);
        }
        else if (jcon["type"] == "pin_joint"){
            add_pin_joint_constraint(jcon, bodies);
        }
    }

    num_bodies = bodies.num_bodies();
    dim = bodies.dim();

    return true;
}

void LinearConstraintHandler::init()
{
    if (num_constraints() <= 0){
        return;
    }

    if (!assembled){
        assemble_constraints();
    }

    // // Debug:
    // std::cout << "constraint mat:\n" << m_constraint_mat << std::endl;
    // std::cout << std::endl;
    // std::cout << "bcs vec:\n" << m_bcs << std::endl;

    // --------------------------------------------------------------------------------------
    // Using QR decomposition with column-pivoting on constraint matrix

    // .. Step 1. Performing QR decomposition
    // .. C^T * P = Q * R
    // .. P^T * C = R^T * Q^T

    Eigen::MatrixXd m_constraint_mat_T = m_constraint_mat.transpose();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr = m_constraint_mat_T.colPivHouseholderQr();

    m_rank = qr.rank();
    // std::cout << "C rank: " << m_rank << std::endl;

    Eigen::MatrixXd P = qr.colsPermutation();
    Eigen::MatrixXd Q = qr.householderQ();
    Eigen::MatrixXd R = qr.matrixR();

    // std::cout << "P matrix:" << std::endl << P << std::endl;
    // std::cout << "Q matrix:" << std::endl << Q << std::endl;
    // std::cout << "R matrix:" << std::endl << R << std::endl;

    // .. Step 2. Truncate constraints (Keeping only full-rank)
    // .. If C was not full-rank, we can truncate the constraint matrix
    // .. after permutating it with P^T. 
    // .. (P^T * C)[:rank,:] * x = (P^T * b)[:rank,:]
    // .. Bx = d, nrows = rank
    Eigen::MatrixXd constraint_trunc = (P.transpose() * m_constraint_mat).topRows(m_rank);
    Eigen::VectorXd bcs_trunc        = (P.transpose() * m_bcs).topRows(m_rank);

    // .. Step 2. Partition R and Q
    // .. Additionally, we partition R and Q into parts related to constraints
    // .. and parts that are not
    // .. B = R_c^T * Q_c^T
    // .. where R^T = [[R_c^T, 0],[R_f^T, 0]], R_c^T is rank x rank
    // .. and   Q^T = [Q_c, Q_f]^T, Q_c^T is rank x numdofs

    Eigen::MatrixXd Q_c = Q.leftCols(m_rank);
    Eigen::MatrixXd Q_f = Q.rightCols(Q.cols() - m_rank);
    Eigen::MatrixXd R_c = R.topLeftCorner(m_rank, m_rank).triangularView<Eigen::Upper>();

    // std::cout << "Q_c matrix:" << std::endl << Q_c << std::endl;
    // std::cout << "R_c matrix:" << std::endl << R_c << std::endl;
    // std::cout << "Q_f matrix:" << std::endl << Q_f << std::endl;

    // .. Step 3. Get final form of constraint matrix
    // .. Bx = d
    // .. R_c^T * Q_c^T * x = d
    // .. Q_c^T * x = R_c^-T * d = p
    // .. Q_c^T now becomes our constraint matrix
    // .. Q_c^-T * d = p is now our bcs vector

    // .. R_c^-T * d = p equivalent to R_c^T * p = d
    m_bcs_qr = R_c.transpose().triangularView<Eigen::Lower>().solve(bcs_trunc);
    m_constraint_qr = Q_c.transpose();


    // .. Q is a full-rank transformation matrix
    // .. we define a constraint space Z (R^ndofs)
    // .. where z = Qx and since Q is orthonormal,
    // .. x = Q^T * z

    // .. part of the tranformation matrix Q
    // .. which spans the unconstrained space (subspace of Z)
    // .. i.e. z_free = Q_f^T * x. z_free is not constrained by any prescribed values
    m_z_free_basis = Q_f;

    initialized = true;
}

void LinearConstraintHandler::assemble_constraints(){

    int ndof = PoseD::dim_to_ndof(dim);

    // Get the number of rows in constraint matrix
    int n_rows = 0;
    n_rows += m_pin_world_cons.size() * PinWorldConstraint::rows(dim);
    n_rows += m_pin_joint_cons.size() * PinJointConstraint::rows(dim);

    m_constraint_mat.resize(n_rows, num_bodies * ndof);
    m_constraint_mat.setZero();
    m_bcs.resize(n_rows);
    m_bcs.setZero();

    // Start filling in constraint matrix and bcs
    int curr_row = 0;
    for (int ci = 0; ci < this->num_constraints(); ci++){
        auto& con = (*this)[ci];
        assert(dim == con.dim());

        con.constraint_matrix(m_constraint_mat, curr_row, dim);
        con.boundary_conditions(m_bcs, curr_row, dim);
        curr_row += con.rows();
    }

    assembled = true;
    initialized = false;
}

const Eigen::MatrixXd& LinearConstraintHandler::constraint_matrix()
{
    if (!assembled){
        throw std::runtime_error("Linear constraints have not been assembled!");
    }
    return m_constraint_mat;
}

const Eigen::VectorXd& LinearConstraintHandler::boundary_conditions()
{
    if (!assembled){
        throw std::runtime_error("Linear constraints have not been assembled!");
    }
    return m_bcs;
}

bool LinearConstraintHandler::add_pin_world_constraint(const nlohmann::json& json_constraint, const RigidBodyAssembler& bodies)
{
    auto body_name = json_constraint["body_name"].get<std::string>();
    long body_id = bodies.body_name_to_id(body_name);
    
    if (body_id == -1){
        throw "Could not find body named " + body_name;
        return false;
    }

    VectorMax3d b_point;
    from_json(json_constraint["body_point"], b_point);

    this->m_pin_world_cons.emplace_back(bodies, body_id, b_point);

    assembled = false;
    initialized = false;

    return true;
}

bool LinearConstraintHandler::add_pin_joint_constraint(const nlohmann::json& json_constraint, const RigidBodyAssembler& bodies)
{
    auto bodyA_name = json_constraint["bodyA_name"].get<std::string>();
    long bodyA_id = bodies.body_name_to_id(bodyA_name);
    
    if (bodyA_id == -1){
        assert(false);
        throw "Could not find body named " + bodyA_name;
        return false;
    }

    auto bodyB_name = json_constraint["bodyB_name"].get<std::string>();
    long bodyB_id = bodies.body_name_to_id(bodyB_name);
    
    if (bodyB_id == -1){
        assert(false);
        throw "Could not find body named " + bodyB_name;
        return false;
    }

    VectorMax3d bA_point;
    from_json(json_constraint["bodyA_point"], bA_point);

    this->m_pin_joint_cons.emplace_back(bodies, bodyA_id, bodyB_id, bA_point);

    assembled = false;
    initialized = false;

    return true;
}

Eigen::VectorXd LinearConstraintHandler::enforce_constraints(const Eigen::Ref<const Eigen::VectorXd> x){
    // We make use of the orthonormal form of the qr decomped constraint matrix C (i.e. C * C^T == I)
    // We set x_new = x - C^T(Cx - b)
    // Essentially, x_new = x - reprojection_of_error

    // This makes x_new satify Cx_new - b = 0 
    // Because C * (x - C^T(Cx - b)) - b = Cx - C * C^T * Cx + C * C^Tb - b = 0

    Eigen::VectorXd correction = m_constraint_qr.transpose() * (m_constraint_qr * x - m_bcs_qr);

    Eigen::VectorXd x_enforced = x - correction;

    return x_enforced;
}

Eigen::VectorXd LinearConstraintHandler::calc_error(const Eigen::Ref<const Eigen::VectorXd> x){
    return m_constraint_mat * x - m_bcs;
}


const LinearConstraint& LinearConstraintHandler::operator[](size_t ci) const
{
    if (ci < m_pin_world_cons.size()) {
        return m_pin_world_cons[ci];
    }
    ci -= m_pin_world_cons.size();
    if (ci < m_pin_joint_cons.size()) {
        return m_pin_joint_cons[ci];
    }
    assert(false);
    throw "Invalid constraint index!";
};

LinearConstraint& LinearConstraintHandler::operator[](size_t ci)
{
    assembled = false;

    if (ci < m_pin_world_cons.size()) {
        return m_pin_world_cons[ci];
    }
    ci -= m_pin_world_cons.size();
    if (ci < m_pin_joint_cons.size()) {
        return m_pin_joint_cons[ci];
    }
    assert(false);
    throw "Invalid constraint index!";
};

const int LinearConstraintHandler::num_constraints() const
{
    int sum = 0;
    sum += m_pin_world_cons.size();
    sum += m_pin_joint_cons.size();
    return sum;
}

int LinearConstraintHandler::rank() const
{
    if (!initialized) {
        throw std::logic_error(
            "Linear constraint handler must be initialized before checking "
            "rank");
    }
    return m_rank;
}

} // namespace ipc::rigid