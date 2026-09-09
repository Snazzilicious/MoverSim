
### FixedWingMover

* force models incomplete
    * need damping model
    * body forces function of attitude relative to velocity
* Autopilot is just a stub
Tests
    All methods

### Grand TODO

* Redesign Aircraft movers
    * Standard damping & equilibrium dynamics for control axes
        * (restoring forces)
    * lift, drag, slip forces update
        * function of attitude relative to velocity
    * Fixed wing mover
    * Fixed wing autopilot
        * Intelligence to pick the appropriate inputs
    * Rocket mover
    * Rocket guidance
    * Ballistic mover
    * Remove old aircraft mover(s)
        * Rewrite scenarios
        * Update AircraftSplineMover to use fru instead of quaternions 
    * Remove quaternions everywhere
        * redesign AircraftSplineMover
    * Update logging
        * mainly just ...
            * change orientation from quaternions to fru
            * remove body rates
    * Update plotting
    * Update example scenarios
* Future features (need not be added yet, but ideally not precluded)
    * despawn platform (e.g. if crashes into ground or something)
        * update ballistic missile scenario
    * collision evaluator
        * including with the ground
    * line of sight evaluator
