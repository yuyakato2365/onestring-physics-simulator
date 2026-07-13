#include "problem_factory.hpp"

#include <problems/distance_barrier_rb_problem.hpp>
// #include <problems/volume_rb_problem.hpp>

namespace ipc::rigid {

const ProblemFactory& ProblemFactory::factory()
{
    static ProblemFactory instance;

    return instance;
}

ProblemFactory::ProblemFactory()
{
    problems_.emplace(
        DistanceBarrierRBProblem::problem_name(),
        std::make_shared<DistanceBarrierRBProblem>());
    // problems_.emplace(
    //     "volume_rb_problem", std::make_shared<VolumeRBProblem>());
}

std::shared_ptr<SimulationProblem>
ProblemFactory::get_problem(const std::string& name) const
{
    auto it = problems_.find(name);
    if (it == problems_.end()) {
        return nullptr;
    }
    return it->second;
}

} // namespace ipc::rigid
