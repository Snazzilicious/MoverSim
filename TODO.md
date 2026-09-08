
### FixedWingMover

* Correction event
    * replace renormalize with project to rotation?
* undefined references in compute_state_derivative
* force models missing
    * need damping model
* Autopilot is just a stub


### Grand TODO

* Redesign Aircraft movers
    * Coriolis update
        * must include orientation change (just subtract earth's rotation from A')
    * Standard damping & equilibrium dynamics for control axes
        * (restoring forces)
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
