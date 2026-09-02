
* Redesign Aircraft movers
    * Need Torque / rotational math
        * review in math.orientation
        * use forward, right, up
        * need to renormalization routine
    * Coriolis update
        * must include orientation change
    * Fixed wing mover
    * Fixed wing autopilot
    * Rocket mover
    * Rocket guidance
    * Remove old aircraft mover(s)
        * Rewrite scenarios
    * Remove quaternions everywhere
* Future features (need not be added yet, but ideally not precluded)
    * despawn platform (e.g. if crashes into ground or something)
        * update ballistic missile scenario
    * collision evaluator
        * including with the ground
    * line of sight evaluator
