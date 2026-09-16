#pragma once

#include <Eigen/Core>
#include <physics/rigid_body_assembler.hpp>
#include <utils/eigen_ext.hpp>

namespace ipc::rigid {

class LinearConstraint {
public:
    virtual ~LinearConstraint() = default;
    virtual void constraint_matrix(Eigen::MatrixXd& constraint_mat, int start_row, int dim) = 0;
    virtual void boundary_conditions(Eigen::VectorXd& bcs, int start_row, int dim) = 0;
    virtual int rows() = 0;
    virtual int dim() = 0;
};

// Pin a body point to a world point
class PinWorldConstraint : public LinearConstraint {
public:
    // This pins the body material point to it's position in the world (according to the body's pose at initialization)
    // We will take in the body point in input coordinates.
    PinWorldConstraint(const RigidBodyAssembler& bodies, size_t body_id, VectorMax3d body_input_point);

    // This pins the body material point to a world point
    // This constructor should be avoided as we don't check if constraints are consistent or overconstraint
    PinWorldConstraint(size_t body_id, VectorMax3d body_material_point, VectorMax3d world_point);

    void constraint_matrix(Eigen::MatrixXd& constraint_mat, int start_row, int dim) override;
    void boundary_conditions(Eigen::VectorXd& bcs, int start_row, int dim) override;

    static int rows(int dim) {return dim;}
    int rows() override {return this->dim();}
    int dim() {return this->body_point.rows();}

    // Index of body in rigid assembler
    long body_id;
    // body_point in material coordinates
    VectorMax3d body_point;
    VectorMax3d world_point;
};

// Pin two bodies together at a point
class PinJointConstraint : public LinearConstraint {
public:
    // This pins bodyA to bodyB
    // bodyA is pinned at material_pointA(bodyA_input_point)
    // bodyB is pinned at material_pointB(world_point(bodyA_input_point))
    PinJointConstraint(const RigidBodyAssembler& bodies, size_t bodyA_id, size_t bodyB_id, VectorMax3d bodyA_input_point);

    void constraint_matrix(Eigen::MatrixXd& constraint_mat, int start_row, int dim) override;
    void boundary_conditions(Eigen::VectorXd& bcs, int start_row, int dim) override;

    static int rows(int dim) {return dim;}
    int rows() override {return this->dim();}
    int dim() {return this->bodyA_point.rows();}

    // Index of bodyA and bodyB in rigid assembler
    long bodyA_id;
    long bodyB_id;
    // bodyA_point in material coordinates of bodyB
    VectorMax3d bodyA_point;
    // bodyB_point in material coordinates of bodyB
    VectorMax3d bodyB_point;
};

} // namespace ipc::rigid