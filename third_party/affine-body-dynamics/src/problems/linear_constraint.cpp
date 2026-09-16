#pragma once

#include "linear_constraint.hpp"

#include <physics/rigid_body.hpp>

namespace ipc::rigid {

PinWorldConstraint::PinWorldConstraint(
    const RigidBodyAssembler& bodies, size_t body_id, VectorMax3d body_input_point)
    : body_id(body_id)
{
    int dim = body_input_point.rows();
    assert(dim == 2 || dim == 3);
    body_point = bodies[body_id].input_to_material_point(body_input_point);
    world_point = bodies[body_id].world_point(body_point);
};

PinWorldConstraint::PinWorldConstraint(
    size_t body_id, VectorMax3d body_material_point, VectorMax3d world_point)
    : body_id(body_id)
    , body_point(body_material_point)
    , world_point(world_point)
{
    int dim = body_point.rows();
    assert(dim == world_point.size());
    assert(dim == 2 || dim == 3);
};

void PinWorldConstraint::constraint_matrix(
    Eigen::MatrixXd& constraint_mat, int start_row, int dim)
{
    assert(dim == body_point.rows());
    int ndof = PoseD::dim_to_ndof(dim);
    assert((body_id + 1) * ndof <= constraint_mat.cols());
    constraint_mat.middleRows(start_row,dim).setZero();
    constraint_mat.block(start_row, body_id * ndof, dim, ndof) = RigidBody::J_matrix(body_point);
};

void PinWorldConstraint::boundary_conditions(Eigen::VectorXd& bcs, int start_row, int dim)
{
    assert(dim == body_point.rows());
    assert(start_row + dim <= bcs.size());
    bcs.middleRows(start_row, dim) = world_point;
};

PinJointConstraint::PinJointConstraint(
    const RigidBodyAssembler& bodies, size_t bodyA_id, size_t bodyB_id, VectorMax3d bodyA_input_point)
    : bodyA_id(bodyA_id)
    , bodyB_id(bodyB_id)
{
    int dim = bodyA_input_point.rows();
    assert(dim == 2 || dim == 3);
    const RigidBody& bodyA = bodies[bodyA_id];
    const RigidBody& bodyB = bodies[bodyB_id];
    bodyA_point = bodyA.input_to_material_point(bodyA_input_point);
    bodyB_point = bodyB.material_point(bodyA.world_point(bodyA_point));
};

void PinJointConstraint::constraint_matrix(
    Eigen::MatrixXd& constraint_mat, int start_row, int dim)
{
    assert(dim == bodyA_point.rows());
    int ndof = RigidBody::ndof(dim);
    assert((bodyA_id + 1) * ndof <= constraint_mat.cols());
    assert((bodyB_id + 1) * ndof <= constraint_mat.cols());

    constraint_mat.middleRows(start_row,dim).setZero();

    constraint_mat.block(start_row, bodyA_id * ndof, dim, ndof) = RigidBody::J_matrix(bodyA_point);
    constraint_mat.block(start_row, bodyB_id * ndof, dim, ndof) = -RigidBody::J_matrix(bodyB_point);
};

void PinJointConstraint::boundary_conditions(Eigen::VectorXd& bcs, int start_row, int dim)
{
    assert(dim == bodyA_point.rows());
    assert(start_row + dim <= bcs.size());
    bcs.middleRows(start_row, dim).setZero();
};



} // namespace ipc::rigid