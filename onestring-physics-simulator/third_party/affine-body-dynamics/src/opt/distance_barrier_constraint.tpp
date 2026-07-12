#pragma once
#include "distance_barrier_constraint.hpp"

#include <ipc/barrier/barrier.hpp>
// #include <geometry/distance.hpp>
#include <logger.hpp>

namespace ipc::rigid {

template <typename T>
T DistanceBarrierConstraint::distance_barrier(
    const T& distance, const double dhat) const
{
    const double& dmin = minimum_separation_distance;
    return barrier(
        distance - dmin * dmin, 2 * dmin * dhat + dhat * dhat);
}

} // namespace ipc::rigid
