class Controller:
    """
    Base class for guidance systems, autopilots, and behavioral controllers.
    """
    def __init__(self, update_interval=None):
        """
        Parameters:
            update_interval: The periodic rate in seconds at which the controller 
                             should run. If None, it does not poll automatically.
        """
        self.platform = None
        self.update_interval = update_interval
        self._initialized = False

    def initialize(self, engine):
        """
        Initializes the controller and schedules the first periodic update event.
        """
        if self._initialized:
            return
        if self.update_interval is not None and self.update_interval > 0:
            engine.schedule(
                engine.t, 
                self._update_recurring, 
                name=f"Controller_{self.platform.id}", 
                interval=self.update_interval
            )
        self._initialized = True

    def _update_recurring(self, engine):
        """
        Wrapper to run the controller's update loop and trigger solver resets if necessary.
        """
        self.update(engine.t, engine)

    def update(self, t, engine):
        """
        User-defined control logic. Overriden by subclasses to adjust forces/kinematics.
        
        Parameters:
            t: Current simulation time.
            engine: The SimulationEngine instance.
        """
        pass
