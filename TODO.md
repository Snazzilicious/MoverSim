

### Grand TODO

* Redesign Aircraft movers
    * Remove old aircraft mover(s)
        * Update AircraftSplineMover to use fru instead of quaternions 
    * Remove quaternions everywhere
        * redesign AircraftSplineMover
    * Update plotting
        * Just do 3D Trajectory plotting
    * remove unused imports and other dead code
        * h5 utils
    * Revisit FixedWingMover
        * add helpers and properties and setters like RocketMover has
        * update Autopilot to use them
        * (make them consistent)
    * Maybe separate out FixedWingMover and RocketMover
* Identify duplicated functions
    * local_enu_basis
* Future features (need not be added yet, but ideally not precluded)
    * despawn platform (e.g. if crashes into ground or something)
        * disable controller and don't reschedule when it comes up
        * consider re-registering all movers if that is easier
        * update ballistic missile scenario
    * collision evaluator
        * including with the ground
    * line of sight evaluator
