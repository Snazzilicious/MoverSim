
* Redesign Aircraft movers
    * Need Torque / rotational math
        * review in math.orientation
        * use forward, right, up
        * need renormalization routine
    * Standard damping & equilibrium dynamics for control axes
    * Coriolis update
        * must include orientation change
    * Restoring forces
    * Fixed wing mover
    * Fixed wing autopilot
        * Intelligence to pick the appropriate inputs
    * Rocket mover
    * Rocket guidance
    * Ballistic mover
    * Remove old aircraft mover(s)
        * Rewrite scenarios
    * Remove quaternions everywhere
        * redesign AircraftSplineMover
    * Update logging
    * Update plotting
* Future features (need not be added yet, but ideally not precluded)
    * despawn platform (e.g. if crashes into ground or something)
        * update ballistic missile scenario
    * collision evaluator
        * including with the ground
    * line of sight evaluator
