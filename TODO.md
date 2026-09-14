

### Grand TODO

* Redesign Aircraft movers
    * Rocket mover
        * mass model
            * fuel consumption
            * stage jettison
        * control variables
        * restoring model
    * Rocket guidance
        * Waypoints or whatever
        * Trigger stage separation
    * Ballistic mover
        * for expended stages
    * Remove old aircraft mover(s)
        * Aircraft6DOFMover, Aircraft6DOFAutopilot
        * Update AircraftSplineMover to use fru instead of quaternions 
        * Rewrite scenarios
        * remove unused imports
    * Remove quaternions everywhere
        * redesign AircraftSplineMover
    * Update logging
        * mainly just ...
            * change orientation from quaternions to fru
            * remove body rates
    * Update plotting
    * Update example scenarios
    * Revisit FixedWingMover
        * add helpers and properties and setters like RocketMover has
        * update Autopilot to use them
        * (make them consistent)
* Future features (need not be added yet, but ideally not precluded)
    * despawn platform (e.g. if crashes into ground or something)
        * update ballistic missile scenario
    * collision evaluator
        * including with the ground
    * line of sight evaluator
