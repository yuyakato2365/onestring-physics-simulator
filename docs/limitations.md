# Limitations

- v0.1 uses a lightweight custom XPBD/PBD-style simulator.
- Rope contact and friction are simplified.
- Collision is approximate or absent.
- The hinge model is simplified.
- There is no fabrication guarantee.
- There is no exact OneString string path optimization.
- There is no 8-vertex frustum tile yet.
- Closed shapes such as bunny are not fully supported.
- Physical simulation results are qualitative, not quantitatively validated.

The warning shown for closed imported meshes is:

> v0.1 works best for open height-field-like surfaces. Closed shapes such as bunny are not fully supported yet.
