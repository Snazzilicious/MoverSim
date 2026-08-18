
# Mover Simulation Requirements

## Overview

Intent is to design a simulator for moving platforms in a global coordinate frame.
Should be general-purpose for setting up a variet of different scenarios.
Should provide geometric and geographic routines.
Should support logical control in addition to purely physics-based movement.
Should provide templates/examples for common platforms and behaviors.


## Requirements

* Support for movement governed by analytical functions of time
* Support for movement governed by newtonian motion
    * object provides (some of) the forces acting on it
* Support for dynamically spawned platforms
    * for e.g. missile launch
* Support for logical control
    * Must regularly poll platforms for changes to behavior
    * Must not require movement to be strictly smooth
    * e.g. missile tracking target
    * e.g. aircraft following waypoints
* Support various observers
    * e.g. event handlers
    * e.g. position loggers
* Provide global physics
    * coriolis force
    * gravity
    * airspeed and wind drag
* Provide global geography and geometry
    * coordinate transformations e.g. ENU, ECEI
    * 6 DOF poisition and velocity
* Should provide template mover models
    * Spline follower (does not use ode solver)
        * Waypoint follower subclass
    * Aircraft
        * thrust, lift, drag
* May provide
    * Line of sight between platforms evaluator
    * Collision detector
        * including with the ground


## Catalog of libraries, design patterns, etc

* Probably should be written in Python
    * allows very generic mover and behavior models
    * can integrate with other tools
* Scipy RK45
    * adaptive step size
    * does _not_ require strictly smooth derivatives
    * can set max step size and final time
* AFSIM design
    * Platforms with "Mover" member
    * min step size
    * User-created Events
        * One time & recurring
    * Sim-created events
        * position updated
        * platform created