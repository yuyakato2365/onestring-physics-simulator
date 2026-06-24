# Physics Model

## Rigid Tile Model

Each quad tile is represented by four corner particles. Tile rigidity is enforced by PBD-style distance constraints over the four edges and two diagonals. This keeps each tile close to a rigid quad while keeping the implementation small and inspectable.

The `RigidTile` class is included as a lightweight rigid body container with mass, inertia-like radius, position, quaternion orientation, velocity, local corner coordinates, and force/torque accumulation. The active v0.1 world uses corner constraints as the primary rigid tile mechanism because it works naturally with hinge and rope constraints.

## Hinge Constraint Model

Neighboring tiles are connected with `PointHingeConstraint` objects. For each shared edge, two point constraints keep the corresponding edge endpoints together. This is not a full revolute joint, but it captures hinge-connected mechanical linkage behavior for small qualitative demos.

Optional placeholders such as `AngularHingeLimit` and `HingeSpring` are present for later expansion.

## Rope Particle Model

The rope is represented by `RopeParticle` objects connected by `DistanceConstraint` objects. Rope particles have current position, previous position, velocity, mass, and a pinned flag. The rope can be attached to tile corners through `RopeTileAttachment`.

## Rope Pulling Model

The world includes a pull handle that moves from below the flat structure toward a point above the center. Tendon-like constraints connect tile centers to the handle and shrink their effective rest lengths according to `rope_rest_length_scale`. This approximates a string pull that injects tension into the linkage.

The boundary rope remains visible and mechanically coupled to tile boundary points. Friction and detailed string routing are simplified in v0.1.

## XPBD/PBD Solver

Each time step performs:

1. Gravity and damped Verlet-style position prediction.
2. Rope distance constraint solving.
3. Rope-tile attachment solving.
4. Tile edge and diagonal rigidity solving.
5. Hinge point constraint solving.
6. Pull-tendon constraint solving.
7. Optional debug goal attraction, off by default.
8. Velocity reconstruction from position differences.

The implementation is PBD/XPBD-style rather than a strict full XPBD formulation. It is built for clarity and small examples.

## Gravity and Damping

Gravity is applied during position prediction. Damping reduces velocity-like displacement between current and previous positions. These parameters are exposed in the Streamlit UI.

## Limitations of the Physical Model

The model is qualitative. It is useful for making rigid tile, hinge, rope, and pull-handle interactions visible, but it is not quantitatively validated against real fabricated mechanisms.
