
### FixedWingMover

* Autopilot is just a stub
Tests
    All methods

### Grand TODO

* Redesign Aircraft movers
    * Fixed wing mover
        * tests for physical behavior
    * Fixed wing autopilot
        * tests for tracking a route
        * Intelligence to pick the appropriate inputs
    * Rocket mover
    * Rocket guidance
    * Ballistic mover
    * Remove old aircraft mover(s)
        * Aircraft6DOFMover, Aircraft6DOFAutopilot
        * Update AircraftSplineMover to use fru instead of quaternions 
        * Rewrite scenarios
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
