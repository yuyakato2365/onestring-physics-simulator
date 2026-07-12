// Detection collisions between different geometry.
// Includes continuous collision detection to compute the time of impact.
// Supported geometry: point vs edge

#include "broad_phase.hpp"

#include <tbb/parallel_invoke.h>

#include <ccd/rigid/broad_phase.hpp>
#include <ccd/rigid/rigid_body_bvh.hpp>
#include <logger.hpp>
#include <profiler.hpp>

namespace ipc::rigid {

///////////////////////////////////////////////////////////////////////////////
// Broad-Phase CCD
///////////////////////////////////////////////////////////////////////////////

void detect_collision_candidates_linear(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const int collision_types,
    Candidates& candidates,
    DetectionMethod method,
    const double inflation_radius)
{
    if (bodies.m_rbs.size() <= 1) {
        return;
    }

    PROFILE_POINT("detect_collision_candidates (Two Poses)");
    PROFILE_START();

    assert(bodies.m_edges.size() == 0 || bodies.m_edges.cols() == 2);
    assert(bodies.m_faces.size() == 0 || bodies.m_faces.cols() == 3);

    switch (method) {
    case BRUTE_FORCE: {
        Eigen::MatrixXd V_t0 = bodies.world_vertices(poses_t0);
        detect_collision_candidates_brute_force(
            V_t0, bodies.m_edges, bodies.m_faces, bodies.group_ids(),
            collision_types, candidates);
        break;
    }
    // case HASH_GRID: {
    //     Eigen::MatrixXd V_t0 = bodies.world_vertices(poses_t0);
    //     Eigen::MatrixXd V_t1 = bodies.world_vertices(poses_t1);
    //     detect_collision_candidates_hash_grid(
    //         V_t0, V_t1, bodies.m_edges, bodies.m_faces, bodies.group_ids(),
    //         collision_types, candidates, inflation_radius);
    //     break;
    // }
    case BVH:
        detect_collision_candidates_linear_bvh(
            bodies, poses_t0, poses_t1, collision_types, candidates,
            inflation_radius);
        break;
    case IAABB:
        // detect_collision_candidates_iaabb(
        //     bodies, poses_t0, poses_t1, collision_types, candidates,
        //     inflation_radius);
        break;
    }

    PROFILE_END();
}

void detect_collision_candidates_linear(
    const RigidBodyAssembler& bodies,
    const PosesD& poses,
    const int collision_types,
    Candidates& candidates,
    DetectionMethod method,
    const double inflation_radius)
{
    if (bodies.m_rbs.size() <= 1) {
        return;
    }

    PROFILE_POINT("detect_collision_candidates (Single Pose)");
    PROFILE_START();
    switch (method) {
    case BRUTE_FORCE: {
        detect_collision_candidates_brute_force(
            bodies.world_vertices(poses), bodies.m_edges, bodies.m_faces,
            bodies.group_ids(), collision_types, candidates);
        break;
    }
    case HASH_GRID:
        break;
    case BVH:
        detect_collision_candidates_linear_bvh(
            bodies, poses, collision_types, candidates, inflation_radius);
        break;
    case IAABB:
        break;
    }

    PROFILE_END();
}

// void detect_collision_candidates(
//     const Eigen::MatrixXd& vertices_t0,
//     const Eigen::MatrixXd& vertices_t1,
//     const Eigen::MatrixXi& edges,
//     const Eigen::MatrixXi& faces,
//     const Eigen::VectorXi& group_ids,
//     const int collision_types,
//     Candidates& candidates,
//     DetectionMethod method,
//     const double inflation_radius)
// {
//     assert(edges.size() == 0 || edges.cols() == 2);
//     assert(faces.size() == 0 || faces.cols() == 3);

//     PROFILE_POINT("detect_collision_candidates(vertices...)");
//     PROFILE_START();

//     switch (method) {
//     case BRUTE_FORCE:
//         detect_collision_candidates_brute_force(
//             vertices_t0, edges, faces, group_ids, collision_types,
//             candidates);
//         break;
//     case HASH_GRID:
//         detect_collision_candidates_hash_grid(
//             vertices_t0, vertices_t1, edges, faces, group_ids,
//             collision_types, candidates, inflation_radius);
//         break;
//     default:
//         throw NotImplementedError(
//             "detect_collision_candidates(vertices...) is only implemented for
//             " "BRUTE_FORCE and HASH_GRID!");
//     }

//     PROFILE_END();
// }

void detect_edge_vertex_collision_candidates_brute_force(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& edges,
    const Eigen::VectorXi& group_ids,
    std::vector<EdgeVertexCandidate>& ev_candidates)
{
    const bool check_group = group_ids.size() > 0;
    for (int ei = 0; ei < edges.rows(); ei++) {
        // Loop over all vertices
        for (int vi = 0; vi < vertices.rows(); vi++) {
            // Check that the vertex is not an endpoint of the edge
            bool is_endpoint = vi == edges(ei, 0) || vi == edges(ei, 1);
            bool same_group = check_group
                && (group_ids(vi) == group_ids(edges(ei, 0))
                    || group_ids(vi) == group_ids(edges(ei, 1)));
            if (!is_endpoint && !same_group) {
                ev_candidates.emplace_back(ei, vi);
            }
        }
    }
}

void detect_edge_edge_collision_candidates_brute_force(
    const Eigen::MatrixXi& edges,
    const Eigen::VectorXi& group_ids,
    std::vector<EdgeEdgeCandidate>& ee_candidates)
{
    const bool check_group = group_ids.size() > 0;
    for (int ei = 0; ei < edges.rows(); ei++) {
        // Loop over all remaining edges
        for (int ej = ei + 1; ej < edges.rows(); ej++) {
            bool has_common_endpoint = edges(ei, 0) == edges(ej, 0)
                || edges(ei, 0) == edges(ej, 1) || edges(ei, 1) == edges(ej, 0)
                || edges(ei, 1) == edges(ej, 1);
            bool same_group = check_group
                && (group_ids(edges(ei, 0)) == group_ids(edges(ej, 0))
                    || group_ids(edges(ei, 0)) == group_ids(edges(ej, 1))
                    || group_ids(edges(ei, 1)) == group_ids(edges(ej, 0))
                    || group_ids(edges(ei, 1)) == group_ids(edges(ej, 1)));
            if (!has_common_endpoint && !same_group) {
                ee_candidates.emplace_back(ei, ej);
            }
        }
    }
}

void detect_face_vertex_collision_candidates_brute_force(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& faces,
    const Eigen::VectorXi& group_ids,
    std::vector<FaceVertexCandidate>& fv_candidates)
{
    const bool check_group = group_ids.size() > 0;
    // Loop over all faces
    for (int fi = 0; fi < faces.rows(); fi++) {
        // Loop over all vertices
        for (int vi = 0; vi < vertices.rows(); vi++) {
            // Check that the vertex is not an endpoint of the edge
            bool is_endpoint =
                vi == faces(fi, 0) || vi == faces(fi, 1) || vi == faces(fi, 2);
            bool same_group = check_group
                && (group_ids(vi) == group_ids(faces(fi, 0))
                    || group_ids(vi) == group_ids(faces(fi, 1))
                    || group_ids(vi) == group_ids(faces(fi, 2)));
            if (!is_endpoint && !same_group) {
                fv_candidates.emplace_back(fi, vi);
            }
        }
    }
}

// Find all edge-vertex collisions in one time step using brute-force
// comparisons of all edges and all vertices.
void detect_collision_candidates_brute_force(
    const Eigen::MatrixXd& vertices,
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    const Eigen::VectorXi& group_ids,
    const int collision_types,
    Candidates& candidates)
{
    PROFILE_POINT("detect_collision_candidates_brute_force");
    PROFILE_START();

    assert(edges.size() == 0 || edges.cols() == 2);
    assert(faces.size() == 0 || faces.cols() == 3);

    // Loop over all edges
    tbb::parallel_invoke(
        [&] {
            if (collision_types & CollisionType::EDGE_VERTEX) {
                detect_edge_vertex_collision_candidates_brute_force(
                    vertices, edges, group_ids, candidates.ev_candidates);
            }
        },
        [&] {
            if (collision_types & CollisionType::EDGE_EDGE) {
                detect_edge_edge_collision_candidates_brute_force(
                    edges, group_ids, candidates.ee_candidates);
            }
        },
        [&] {
            if (collision_types & CollisionType::FACE_VERTEX) {
                detect_face_vertex_collision_candidates_brute_force(
                    vertices, faces, group_ids, candidates.fv_candidates);
            }
        });

    PROFILE_END();
}

// // Find all edge-vertex collisions in one time step using spatial-hashing to
// // only compare points and edge in the same cells.
// void detect_collision_candidates_hash_grid(
//     const Eigen::MatrixXd& vertices_t0,
//     const Eigen::MatrixXd& vertices_t1,
//     const Eigen::MatrixXi& edges,
//     const Eigen::MatrixXi& faces,
//     const Eigen::VectorXi& group_ids,
//     const int collision_types,
//     Candidates& candidates,
//     const double inflation_radius)
// {
//     using namespace CollisionType;
//     HashGrid hashgrid;
//     assert(edges.size()); // Even face-vertex need the edges
//     std::runtime_error("ahmed removed HashGrid.resize");
//     // hashgrid.resize(vertices_t0, vertices_t1, edges, inflation_radius);

//     if (collision_types & (EDGE_VERTEX | FACE_VERTEX)) {
//         std::runtime_error("ahmed removed HashGrid.addVertices");
//         // hashgrid.addVertices(vertices_t0, vertices_t1, inflation_radius);
//     }

//     if (collision_types & (EDGE_VERTEX | EDGE_EDGE)) {
//         std::runtime_error("ahmed removed HashGrid.addEdges");
//         // hashgrid.addEdges(vertices_t0, vertices_t1, edges, inflation_radius);
//     }

//     if (collision_types & FACE_VERTEX) {
//         std::runtime_error("ahmed removed HashGrid.addFaces");
//         // hashgrid.addFaces(vertices_t0, vertices_t1, faces, inflation_radius);
//     }

//     auto can_vertices_collide = [&group_ids](size_t vi, size_t vj) {
//         return group_ids[vi] != group_ids[vj];
//     };

//     // Assume checking if vertex is and end-point of the edge is done by
//     // `hashgrid.getVertexEdgePairs(...)`.
//     if (collision_types & EDGE_VERTEX) {
//         std::runtime_error("ahmed removed HashGrid.getVertexEdgePairs");
//         // hashgrid.getVertexEdgePairs(
//         //    edges, candidates.ev_candidates, can_vertices_collide);
//     }
//     if (collision_types & EDGE_EDGE) {
//         std::runtime_error("ahmed removed HashGrid.getEdgeEdgePairs");
//         // hashgrid.getEdgeEdgePairs(
//         //    edges, candidates.ee_candidates, can_vertices_collide);
//     }
//     if (collision_types & FACE_VERTEX) {
//         std::runtime_error("ahmed removed HashGrid.getFaceVertexPairs");
//         // hashgrid.getFaceVertexPairs(
//         //    faces, candidates.fv_candidates, can_vertices_collide);
//     }
// }

void detect_collision_candidates_linear_bvh(
    const RigidBodyAssembler& bodies,
    const PosesD& poses_t0,
    const PosesD& poses_t1,
    const int collision_types,
    Candidates& candidates,
    const double inflation_radius)
{
    std::vector<std::pair<int, int>> body_pairs =
        bodies.close_bodies(poses_t0, poses_t1, inflation_radius);

    typedef tbb::enumerable_thread_specific<Candidates> LocalStorage;
    LocalStorage storages;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), body_pairs.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            LocalStorage::reference loc_storage_candidates = storages.local();
            for (long i = range.begin(); i != range.end(); ++i) {
                int bodyA_id = body_pairs[i].first;
                int bodyB_id = body_pairs[i].second;
                sort_body_pair(bodies, bodyA_id, bodyB_id);

                const auto& bodyA = bodies[bodyA_id];
                const auto& bodyB = bodies[bodyB_id];

                // PROFILE_POINT("detect_collision_candidates_linear_bvh:compute_vertices");
                // PROFILE_START();
                // Compute the smaller body's vertices in the larger body's
                // local coordinates.
                const auto& pA_t0 = poses_t0[bodyA_id].position;
                const auto& pB_t0 = poses_t0[bodyB_id].position;
                const auto& pA_t1 = poses_t1[bodyA_id].position;
                const auto& pB_t1 = poses_t1[bodyB_id].position;
                const auto RA_t0 =
                    poses_t0[bodyA_id].construct_transformation_matrix();
                const auto RB_t0 =
                    poses_t0[bodyB_id].construct_transformation_matrix();
                const auto RA_t1 =
                    poses_t1[bodyA_id].construct_transformation_matrix();
                const auto RB_t1 =
                    poses_t1[bodyB_id].construct_transformation_matrix();
                const Eigen::MatrixXd VA_t0 =
                    ((bodyA.vertices * RA_t0.transpose()).rowwise()
                     + (pA_t0 - pB_t0).transpose())
                    * RB_t0;
                const Eigen::MatrixXd VA_t1 =
                    ((bodyA.vertices * RA_t1.transpose()).rowwise()
                     + (pA_t1 - pB_t1).transpose())
                    * RB_t1;

                std::vector<AABB> VA_aabbs;
                VA_aabbs.reserve(bodyA.num_vertices());
                for (int i = 0; i < bodyA.num_vertices(); i++) {
                    const auto& v_t0 = VA_t0.row(i);
                    const auto& v_t1 = VA_t1.row(i);
                    VA_aabbs.emplace_back(
                        v_t0.cwiseMin(v_t1).array() - inflation_radius,
                        v_t0.cwiseMax(v_t1).array() + inflation_radius);
                }
                // PROFILE_END();

                detect_body_pair_collision_candidates_from_aabbs(
                    bodies, VA_aabbs, bodyA_id, bodyB_id, collision_types,
                    loc_storage_candidates, inflation_radius);
            }
        });

    merge_local_candidates(storages, candidates);
}

// void detect_collision_candidates_iaabb(
//     const RigidBodyAssembler& bodies,
//     const PosesD& poses_t0,
//     const PosesD& poses_t1,
//     const int collision_types,
//     Candidates& candidates,
//     const double inflation_radius)
// {
//     std::vector<std::pair<int, int>> body_pairs =
//         bodies.close_bodies_iaabb(poses_t0, poses_t1, inflation_radius);

//     typedef tbb::enumerable_thread_specific<Candidates> LocalStorage;
//     LocalStorage storages;
//     tbb::parallel_for(
//         tbb::blocked_range<size_t>(size_t(0), body_pairs.size()),
//         [&](const tbb::blocked_range<size_t>& range) {
//             LocalStorage::reference loc_storage_candidates =
//             storages.local(); const int dim = bodies.dim(); for (long i =
//             range.begin(); i != range.end(); ++i) {
//                 int bodyA_id = body_pairs[i].first;
//                 int bodyB_id = body_pairs[i].second;
//                 sort_body_pair(bodies, bodyA_id, bodyB_id);

//                 const auto& bodyA = bodies[bodyA_id];
//                 const auto& bodyB = bodies[bodyB_id];
//                 // Body A must be presented in Body B coordinate system.
//                 // Body A must be presented in Body B coordinate system.

//                 const auto& pA_t0 = poses_t0[bodyA_id].position;
//                 const auto& pB_t0 = poses_t0[bodyB_id].position;
//                 const auto& pA_t1 = poses_t1[bodyA_id].position;
//                 const auto& pB_t1 = poses_t1[bodyB_id].position;
//                 const auto RA_t0 =
//                     poses_t0[bodyA_id].construct_transformation_matrix();
//                 const auto RB_t0 =
//                     poses_t0[bodyB_id].construct_transformation_matrix();
//                 const auto RA_t1 =
//                     poses_t1[bodyA_id].construct_transformation_matrix();
//                 const auto RB_t1 =
//                     poses_t1[bodyB_id].construct_transformation_matrix();
//                 // Body A must be presented in Body B coordinate system.
//                 VectorMax3d minA, maxA, minB, maxB;
//                 // bodyA.compute_spatial_rotation_bbox(poses_t0[bodyA_id],
//                 // poses_t1[bodyA_id], minA, maxA);
//                 minA = (RB_t0 * (pA_t0 - pB_t0))
//                            .cwiseMin(RB_t1 * (pA_t1 - pB_t1))
//                            .array()
//                     - bodyA.r_max;
//                 maxA = (RB_t0 * (pA_t0 - pB_t0))
//                            .cwiseMax(RB_t1 * (pA_t1 - pB_t1))
//                            .array()
//                     + bodyA.r_max;
//                 minA.array() -= inflation_radius;
//                 maxA.array() += inflation_radius;
//                 // bodyB.compute_spatial_rotation_bbox(poses_t0[bodyB_id],
//                 // poses_t1[bodyB_id], minB, maxB); minB.array() -=
//                 // inflation_radius; maxB.array() += inflation_radius;
//                 minB = -Eigen::Vector3d::Constant(bodyB.r_max);
//                 maxB = Eigen::Vector3d::Constant(bodyB.r_max);
//                 // Eigen::Vector3d min_box = Eigen::Vector3d::Zero(),
//                 //                 max_box = Eigen::Vector3d::Zero();
//                 // min_box.head(dim) = minA.cwiseMin(minB);
//                 // max_box.head(dim) = maxA.cwiseMax(maxB);

//                 const Eigen::MatrixXd VA_t0 =
//                     ((bodyA.vertices * RA_t0.transpose()).rowwise()
//                      + (pA_t0 - pB_t0).transpose())
//                     * RB_t0;
//                 const Eigen::MatrixXd VA_t1 =
//                     ((bodyA.vertices * RA_t1.transpose()).rowwise()
//                      + (pA_t1 - pB_t1).transpose())
//                     * RB_t1;
//                 std::vector<AABB> VA_aabbs;
//                 VA_aabbs.reserve(bodyA.num_vertices());
//                 for (int i = 0; i < bodyA.num_vertices(); i++) {
//                     const auto& v_t0 = VA_t0.row(i);
//                     const auto& v_t1 = VA_t1.row(i);
//                     VA_aabbs.emplace_back(
//                         v_t0.cwiseMin(v_t1).array() - inflation_radius,
//                         v_t0.cwiseMax(v_t1).array() + inflation_radius);
//                 }
//                 std::set<int> iVA, iEA, iFA, iVB, iEB, iFB;
//                 // @javidf : DO we need to inflate bodyA points before
//                 checking
//                 // for intersections?
//                 for (int vi = 0; vi < bodyA.num_vertices(); vi++) {
//                     Eigen::Vector3d v_t0, v_t1;
//                     v_t0.head(dim) = VA_t0.row(vi);
//                     v_t1.head(dim) = VA_t1.row(vi);
//                     if ((v_t0.cwiseMax(minB) == v_t0
//                          && v_t0.cwiseMin(maxB) == v_t0)
//                         || (v_t1.cwiseMax(minB) == v_t1
//                             && v_t1.cwiseMin(maxB) == v_t1)) {
//                         iVA.insert(vi);
//                     }
//                 }
//                 for (int vi = 0; vi < bodyB.num_vertices(); vi++) {
//                     Eigen::Vector3d v_t;
//                     v_t.head(dim) = bodyB.vertices.row(vi);
//                     if ((v_t.cwiseMax(minA) == v_t
//                          && v_t.cwiseMin(maxA) == v_t)) {
//                         iVB.insert(vi);
//                     }
//                 }
//                 for (int ei = 0; ei < bodyA.num_edges(); ei++) {
//                     int vi = bodyA.edges(ei, 0);
//                     if (std::find(iVA.begin(), iVA.end(), vi) != iVA.end()) {
//                         iEA.insert(ei);
//                         // iVA.insert(bodyA.edges(ei, 1));
//                         continue;
//                     }
//                     vi = bodyA.edges(ei, 1);
//                     if (std::find(iVA.begin(), iVA.end(), vi) != iVA.end()) {
//                         iEA.insert(ei);
//                         // iVA.insert(bodyA.edges(ei, 0));
//                     }
//                 }
//                 for (int fi = 0; fi < bodyA.num_faces(); fi++) {
//                     int vi = bodyA.faces(fi, 0);
//                     if (std::find(iVA.begin(), iVA.end(), vi) != iVA.end()) {
//                         iFA.insert(fi);
//                         // iVA.insert(bodyA.faces(fi, 1));
//                         // iVA.insert(bodyA.faces(fi, 2));
//                         continue;
//                     }
//                     vi = bodyA.faces(fi, 1);
//                     if (std::find(iVA.begin(), iVA.end(), vi) != iVA.end()) {
//                         iFA.insert(fi);
//                         // iVA.insert(bodyA.faces(fi, 0));
//                         // iVA.insert(bodyA.faces(fi, 2));
//                         continue;
//                     }
//                     vi = bodyA.faces(fi, 2);
//                     if (std::find(iVA.begin(), iVA.end(), vi) != iVA.end()) {
//                         iFA.insert(fi);
//                         // iVA.insert(bodyA.faces(fi, 0));
//                         // iVA.insert(bodyA.faces(fi, 1));
//                     }
//                 }
//                 for (int ei = 0; ei < bodyB.num_edges(); ei++) {
//                     int vi = bodyB.edges(ei, 0);
//                     if (std::find(iVB.begin(), iVB.end(), vi) != iVB.end()) {
//                         iEB.insert(ei);
//                         // iVB.insert(bodyB.edges(ei, 1));
//                         continue;
//                     }
//                     vi = bodyB.edges(ei, 1);
//                     if (std::find(iVB.begin(), iVB.end(), vi) != iVB.end()) {
//                         iEB.insert(ei);
//                         // iVB.insert(bodyB.edges(ei, 0));
//                     }
//                 }
//                 for (int fi = 0; fi < bodyB.num_faces(); fi++) {
//                     int vi = bodyB.faces(fi, 0);
//                     if (std::find(iVB.begin(), iVB.end(), vi) != iVB.end()) {
//                         iFB.insert(fi);
//                         // iVB.insert(bodyB.faces(fi, 1));
//                         // iVB.insert(bodyB.faces(fi, 2));
//                         continue;
//                     }
//                     vi = bodyB.faces(fi, 1);
//                     if (std::find(iVB.begin(), iVB.end(), vi) != iVB.end()) {
//                         iFB.insert(fi);
//                         // iVB.insert(bodyB.faces(fi, 0));
//                         // iVB.insert(bodyB.faces(fi, 2));
//                         continue;
//                     }
//                     vi = bodyB.faces(fi, 2);
//                     if (std::find(iVB.begin(), iVB.end(), vi) != iVB.end()) {
//                         iFB.insert(fi);
//                         // iVB.insert(bodyB.faces(fi, 0));
//                         // iVB.insert(bodyB.faces(fi, 1));
//                     }
//                 }
//                 //
//                 detect_body_pair_collision_candidates_intersection_region(bodies,
//                 // bodyA_id, bodyB_id, collision_types
//                 //     , loc_storage_candidates, inflation_radius
//                 //     , iVA, iEA, iFA, iVB, iEB, iFB, VA_aabbs);
//                 detect_body_pair_collision_candidates_from_aabbs(
//                     bodies, VA_aabbs, bodyA_id, bodyB_id, collision_types,
//                     candidates, inflation_radius);
//             }
//         });
//     merge_local_candidates(storages, candidates);
// }

void detect_collision_candidates_linear_bvh(
    const RigidBodyAssembler& bodies,
    const PosesD& poses,
    const int collision_types,
    Candidates& candidates,
    const double inflation_radius)
{
    PROFILE_POINT("detect_collision_candidates_linear_bvh");
    PROFILE_START();

    // std::vector<std::pair<int, int>> body_pairs =
    //     bodies.close_bodies_iaabb(poses, inflation_radius);
    std::vector<std::pair<int, int>> body_pairs =
        bodies.close_bodies(poses, poses, inflation_radius);

    // Use interval arithmetic to conservativly capture all distance candidates
    // auto posesI = cast<Interval>(poses);

    const int dim = bodies.dim();

    typedef tbb::enumerable_thread_specific<Candidates> LocalStorage;
    LocalStorage storages;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), body_pairs.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            LocalStorage::reference local_storage_candidates = storages.local();
            for (long i = range.begin(); i != range.end(); ++i) {
                int bodyA_id = body_pairs[i].first;
                int bodyB_id = body_pairs[i].second;
                sort_body_pair(bodies, bodyA_id, bodyB_id);

                const auto& bodyA = bodies[bodyA_id];
                const auto& bodyB = bodies[bodyB_id];

                // Body A must be presented in Body B coordinate system.
                const auto& pA_t = poses[bodyA_id].position;
                const auto& pB_t = poses[bodyB_id].position;
                const auto RA_t =
                    poses[bodyA_id].construct_transformation_matrix();
                const auto RB_t =
                    poses[bodyB_id].construct_transformation_matrix();
                // Body A must be presented in Body B coordinate system.
                VectorMax3d minA, maxA, minB, maxB;
                // bodyA.compute_spatial_rotation_bbox(poses_t0[bodyA_id],
                // poses_t1[bodyA_id], minA, maxA);
                minA = (RB_t * (pA_t - pB_t)).array() - bodyA.r_max;
                maxA = (RB_t * (pA_t - pB_t)).array() + bodyA.r_max;
                minA.array() -= inflation_radius;
                maxA.array() += inflation_radius;
                // bodyB.compute_spatial_rotation_bbox(poses_t0[bodyB_id],
                // poses_t1[bodyB_id], minB, maxB); minB.array() -=
                // inflation_radius; maxB.array() += inflation_radius;
                minB = -Eigen::Vector3d::Constant(bodyB.r_max);
                maxB = Eigen::Vector3d::Constant(bodyB.r_max);
                // Eigen::Vector3d min_box = Eigen::Vector3d::Zero(),
                //                 max_box = Eigen::Vector3d::Zero();
                // min_box.head(dim) = minA.cwiseMin(minB);
                // max_box.head(dim) = maxA.cwiseMax(maxB);

                const Eigen::MatrixXd VA =
                    ((bodyA.vertices * RA_t.transpose()).rowwise()
                     + (pA_t - pB_t).transpose())
                    * RB_t;
                std::vector<AABB> VA_aabbs;
                VA_aabbs.reserve(bodyA.num_vertices());
                for (int i = 0; i < bodyA.num_vertices(); i++) {
                    const auto& v_t = VA.row(i);
                    VA_aabbs.emplace_back(
                        v_t.array() - inflation_radius,
                        v_t.array() + inflation_radius);
                }
                detect_body_pair_collision_candidates_from_aabbs(
                    bodies, VA_aabbs, bodyA_id, bodyB_id, collision_types,
                    local_storage_candidates, inflation_radius);
            }
        });
    merge_local_candidates(storages, candidates);

    PROFILE_END();
}

// typedef tbb::enumerable_thread_specific<std::vector<EdgeFaceCandidate>>
//     ThreadSpecificEFCandidates;

// void merge_local_candidates(
//     const ThreadSpecificEFCandidates& storages,
//     std::vector<EdgeFaceCandidate>& ef_candidates)
// {
//     PROFILE_POINT("merge_local_ef_candidates");
//     PROFILE_START();
//     // size up the candidates
//     size_t size = 0;
//     for (const auto& local_candidates : storages) {
//         size += local_candidates.size();
//     }
//     // serial merge
//     ef_candidates.reserve(size);
//     for (const auto& local_candidates : storages) {
//         ef_candidates.insert(
//             ef_candidates.end(), //
//             local_candidates.begin(), local_candidates.end());
//     }
//     PROFILE_END();
// }

// // Use a BVH to create a set of all candidate intersections.
// void detect_intersection_candidates_linear_bvh(
//     const RigidBodyAssembler& bodies,
//     const PosesD& poses,
//     std::vector<EdgeFaceCandidate>& ef_candidates)
// {
//     std::vector<std::pair<int, int>> body_pairs =
//         bodies.close_bodies(poses, poses, /*inflation_radius=*/0);

//     // auto posesI = cast<Interval>(poses);

//     typedef tbb::enumerable_thread_specific<Candidates> LocalStorage;
//     LocalStorage storages;
//     tbb::parallel_for(
//         tbb::blocked_range<size_t>(size_t(0), body_pairs.size()),
//         [&](const tbb::blocked_range<size_t>& range) {
//             LocalStorage::reference loc_storage_candidates =
//             storages.local(); for (long i = range.begin(); i != range.end();
//             ++i) {
//                 int bodyA_id = body_pairs[i].first;
//                 int bodyB_id = body_pairs[i].second;
//                 sort_body_pair(bodies, bodyA_id, bodyB_id);

//                 const auto RA =
//                 poses[bodyA_id].construct_transformation_matrix(); const auto
//                 RB = poses[bodyB_id].construct_transformation_matrix(); const
//                 auto& pA = poses[bodyA_id].position; const auto& pB =
//                 poses[bodyB_id].position; const MatrixXI VA =
//                     ((bodies[bodyA_id].vertices * RA.transpose()).rowwise()
//                      + (pA - pB).transpose())
//                     * RB;

//                 std::vector<AABB> aabbs = vertex_aabbs(VA);

//                 detect_body_pair_intersection_candidates_from_aabbs(
//                     bodies, aabbs, bodyA_id, bodyB_id,
//                     local_storage_candidates);
//             }
//         });

//     merge_local_candidates(storages, ef_candidates);
// }

} // namespace ipc::rigid
