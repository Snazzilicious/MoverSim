
### Grand TODO

* Redesign Aircraft movers
    * Rewrite scenarios
        * re-spec these
        * merge scenario and example run
        * Keep the signature, but replace everything else
        * don't de-spawn yet
        * Update tests
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
        * Just do 3D Trajectory plotting
    * remove unused imports and other dead code
        * h5 utils
    * Revisit FixedWingMover
        * add helpers and properties and setters like RocketMover has
        * update Autopilot to use them
        * (make them consistent)
    * Maybe separate out FixedWingMover and RocketMover
* Future features (need not be added yet, but ideally not precluded)
    * despawn platform (e.g. if crashes into ground or something)
        * update ballistic missile scenario
    * collision evaluator
        * including with the ground
    * line of sight evaluator
