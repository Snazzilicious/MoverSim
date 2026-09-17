
* We are rewriting the scenarios in mover_sim/
* We are moving them to examples/ and merging them with the existing driver scripts
* We are updating them to use the new FixedWingMover and RocketMover
* I have sketched out the new designs:
    * They largely have the same structure as to originals, but with different implementation.
    * Some TODO items are left
    * May need to revise the arguments to the run_* functions to be logically consistent with constructors
* Known supporting steps include
    * Add registration of ExpendedStage in RocketMover.separate_stage (or something similar)
    * Add add_waypoint to FixedWingAutopilot which will put the controller back in tracking mode if no waypoints currently exist
* Will not be despawning platforms yet, just let them fall through the ground
* Will need to update scenario tests to run the new scenarios
* Will then remove old scenarios mover_sim/scenario_*
* Will then remove any dead code
    * mover_sim/hdf5_utils.py
    * others?

ballistic

### Grand TODO

* Redesign Aircraft movers
    * Rewrite scenarios
        * re-spec these
            * Provided general structure
            * Keep the general structure, but replace everything else
            * Signatures can be modified a bit to be more logically consistent with Movers and Controllers
            * Ensure FixedWing startup w/out waypoints tracks existing heading, alt, and speed
        * Add expended stage jettison to RocketMover
        * Add add_waypoint to FixedWingAutopilot
        * merge scenario and example run
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
        * disable controller and don't reschedule when it comes up
        * consider re-registering all movers if that is easier
        * update ballistic missile scenario
    * collision evaluator
        * including with the ground
    * line of sight evaluator
