#pragma once

#include <Eigen/Core>
#include <physics/rigid_body_assembler.hpp>
#include <problems/linear_constraint.hpp>
#include <utils/eigen_ext.hpp>

namespace ipc::rigid {

class LinearConstraintHandler {
public:
    bool settings(const nlohmann::json& params, const RigidBodyAssembler& bodies);

    bool add_pin_world_constraint(const nlohmann::json& params, const RigidBodyAssembler& bodies);
    bool add_pin_joint_constraint(const nlohmann::json& params, const RigidBodyAssembler& bodies);

    void init();
    void assemble_constraints();

    const Eigen::MatrixXd& constraint_matrix();
    const Eigen::VectorXd& boundary_conditions();

    // Part of the tranformation matrix Q, z = Qx
    // which takes z_free to x, x = Q_f * z_free
    // z_free is the part of z that doesn't have prescribed values
    const Eigen::MatrixXd& z_free_basis() {return m_z_free_basis; };

    Eigen::VectorXd enforce_constraints(const Eigen::Ref<const Eigen::VectorXd> x);
    Eigen::VectorXd calc_error(const Eigen::Ref<const Eigen::VectorXd> x);

    const LinearConstraint& operator[](size_t ci) const;
    LinearConstraint& operator[](size_t ci);
    const int num_constraints() const;

    int rank() const;
    // const std::vector<int>& constraint_rows() const;
    // Eigen::VectorXi free_dof() const;

    const std::vector<PinWorldConstraint>& pin_world_cons() const { return m_pin_world_cons;};
    const std::vector<PinJointConstraint>& pin_joint_cons() const { return m_pin_joint_cons;};
    std::vector<PinWorldConstraint>& pin_world_cons(){ assembled = false, initialized = false; return m_pin_world_cons;};
    std::vector<PinJointConstraint>& pin_joint_cons(){ assembled = false; initialized = false; return m_pin_joint_cons;};

protected:
    // Storage for the constraints
    std::vector<PinWorldConstraint> m_pin_world_cons;
    std::vector<PinJointConstraint> m_pin_joint_cons;

    // Flags to mark complete steps
    bool assembled = false;
    bool initialized = false;

    // System dimensions
    int num_bodies;
    int dim;

    // Assembled constraint matrix
    Eigen::MatrixXd m_constraint_mat;
    Eigen::VectorXd m_bcs;
    int m_rank;

    // Constraint matrix after being processed using QR decomp
    // to be full rank and orthonormal
    Eigen::MatrixXd m_constraint_qr;
    Eigen::VectorXd m_bcs_qr;

    // Part of the tranformation matrix Q, z = Qx
    // which takes z_free to x, x = Q_f * z_free
    // z_free is the part of z that doesn't have prescribed values
    Eigen::MatrixXd m_z_free_basis;
};

} // namespace ipc::rigid