# Concept

`onestring-physics-simulator` is a compact research prototype for exploring the workflow behind rope-driven assembly of curved structures from flat quad linkages.

The v0.1 pipeline is:

1. Choose or load a height-field-like target shape.
2. Generate a small quad-tile grid.
3. Optimize an assembled state against the target surface.
4. Generate a corresponding flat initial state.
5. Convert tiles and links into a lightweight physical model.
6. Pull a rope handle and simulate deployment with gravity, damping, rigid tile constraints, hinge constraints, and rope constraints.
7. Compare the physical final state against the optimized assembled state.

The project intentionally separates design fitting from physical deployment. The design optimizer may fit the target directly. The physical simulator does not use direct shape interpolation by default.
