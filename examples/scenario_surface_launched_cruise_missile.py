import os
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mover_sim.math.coordinates import lla_to_ecef


class SurfaceLaunchedCruiseMissileMover(FixedWingMover):

    BOOST_PHASE = 1
    CRUISE_PHASE = 2

    def __init__( self, ... ):
        self.stage = BOOST_PHASE
    
    def _thrust_vector( self, forward ):
        if self.stage == BOOST_PHASE:
            return self.boost_thrust * forward
        else:
            return super()._thrust_vector( forward )

class SurfaceLaunchedCruiseMissileController(FixedWingAutopilot):

    def __init__( self, ..., guidance_delay ):
        self.t_launch = None

    def initialize(self, engine):
        self.t_launch = engine.t # save time of launch to begin delay timer
        super().initialize(engine)
        if not self._boost_start_published:
            engine.broker.publish("boost_start", self.platform)
            self._boost_start_published = True

    def update( self, t, engine ):
        if t - self.t_launch < self.guidance_delay:
            return
        self.platform.mover.stage = CRUISE_PHASE
        if not self._boost_end_published:
            engine.broker.publish("boost_end", self.platform)
            self._boost_end_published = True
        
        super().update( t, engine )
    


def _initial_orientation_from_heading_pitch(initial_position_ecef, heading, pitch_angle):
    """Returns forward, right, up basis matrix in ECEF
    """
    # TODO


SCENARIO_EVENT_TOPICS = [
    "platform_registered",
    "boost_start",
    "boost_end",
]


def run_surface_launched_cruise_missile_scenario(
    initial_position_ecef,
    cruise_speed,
    cruise_altitude,
    cruise_heading,
    boost_duration,
    boost_acceleration,
    launch_pitch_angle,
    t_end,
    sample_interval,
    output_group,
):
    """Run surface-launched cruise missile tracking a given heading and save HDF5 telemetry.

    Args:
        initial_position_ecef: Initial missile ECEF position vector in meters.
        cruise_speed: Commanded cruise speed in meters/second.
        cruise_altitude: Commanded cruise altitude above the WGS-84 ellipsoid in meters.
        cruise_heading: Commanded local-ENU azimuth heading in radians.
        boost_duration: Duration of the boost phase in seconds.
        boost_acceleration: Forward boost acceleration command in meters/second^2.
        launch_pitch_angle: Initial launch pitch angle in radians.
        t_end: Maximum scenario run time in seconds.
        sample_interval: HDF5 logging sample interval in seconds.
        output_group: Caller-created `h5py.Group` used as the root for this scenario run.

    Returns:
        A dictionary containing the simulation engine, platform, logger, and output group.
    """

    # XXX If we need to validate all inputs here, can re-include that from old scenario
    # But I think constructors decently cover all that

    initial_orientation = _initial_orientation_from_heading_pitch(
        initial_position_ecef,
        cruise_heading,
        launch_pitch_angle,
    )
    initial_velocity = np.zeros(3)
    initial_body_rates = np.zeros(3)

    engine = SimulationEngine()
    mover = SurfaceLaunchedCruiseMissileMover(
        initial_position=initial_position_ecef,
        initial_velocity=initial_velocity,
        initial_orientation=initial_orientation,
        initial_body_rates=initial_body_rates,
    )
    # TODO Need to decide where to translate the user-provided arguments into Autopilot constructor arguments
    # here? in SurfaceLaunchedCruiseMissileController constructor?
    controller = SurfaceLaunchedCruiseMissileController(
        cruise_speed=cruise_speed,
        cruise_altitude=cruise_altitude,
        cruise_heading=cruise_heading,
        boost_duration=boost_duration,
        boost_acceleration=boost_acceleration,
        launch_pitch_angle=launch_pitch_angle,
    )
    platform = Platform("surface_cruise_missile", mover, controller)
    engine.register_platform(platform)

    logger = HDF5Logger(
        engine,
        output_group,
        sample_interval=sample_interval,
        include_state=True,
        include_lla=True,
        include_events=True,
        event_topics=SCENARIO_EVENT_TOPICS,
    )

    engine.run(t_end)

    return {
        "engine": engine,
        "platform": platform,
        "logger": logger,
        "output_group": output_group,
    }




def run_example():
    print("=== Running Scenario: Surface-Launched Cruise Missile ===")

    os.makedirs("output", exist_ok=True)
    output_path = Path("output/scenario_surface_launched_cruise_missile.h5")

    with h5py.File(output_path, "w") as h5:
        result = run_surface_launched_cruise_missile_scenario(
            initial_position_ecef=lla_to_ecef(37.6193, -122.3750, 10.0),
            cruise_speed=250.0,
            cruise_altitude=1200.0,
            cruise_heading=0.0,
            boost_duration=2.0,
            boost_acceleration=30.0,
            launch_pitch_angle=0.35,
            t_end=20.0,
            sample_interval=0.1,
            output_group=h5.create_group("surface_run_001"),
        )

    print(f"Simulation ended at t = {result['engine'].t:.2f}s. Telemetry written to {output_path}")


if __name__ == "__main__":
    run_example()
