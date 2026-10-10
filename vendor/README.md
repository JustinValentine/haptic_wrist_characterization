# Vendored code

## barrett_puck

`MockP4` (an in-process emulated Barrett P4 drive) and the modules it imports,
copied unmodified from the `barrett-puck` package (`~/Projects/barrett/barrett-puck`,
`src/barrett_puck/`, pyproject version 0.1.0, MIT license) on 2026-10-10.
barrett-puck is not on PyPI or in version control, so the files live here.

| File | SHA-256 at copy |
| --- | --- |
| exceptions.py | 20ceddd6dc0455936038c10ba1d47e87b6aad1397dc0b2991da4edb45e729463 |
| frames.py | 09258927ebbb9df2be265817207c4ca2d82632e99a5e295810e188f516d1b450 |
| mock.py | 9b3f114a309131538b0fcdc709ba4a1bc0395c45b99772ded54f9df1c2e95395 |
| objects.py | 2753cfe399dbe7b1e0cae92a871a99b76d629e40a5bd2452072a20baeeb2a53a |
| pdo.py | aa2a3515ab91d5f037736671e7f79ab0934ea076d2cea63bd2628e208c7f0760 |

`__init__.py` is new: the original imports the controller and python-can
transport, which the simulation does not need. The wrist-specific behaviour
(physics, CST torque ramp, heartbeats, SYNC loss, firmware 4.3.2 quirks) is
added by subclassing in `p4_sim.py`, so these files stay as copied. To pick up
upstream fixes, copy the files again and update the hashes.
