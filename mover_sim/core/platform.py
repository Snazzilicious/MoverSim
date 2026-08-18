class Platform:
    """
    Represents an entity in the simulation.
    A Platform contains a Mover (for movement dynamics) and an optional Controller (for logic).
    """
    def __init__(self, platform_id: str, mover, controller=None, properties=None):
        """
        Parameters:
            platform_id: Unique identifier for the platform.
            mover: A Mover instance governing its position and kinematics/dynamics.
            controller: An optional Controller instance governing behavior.
            properties: An optional dict containing physical/config properties.
        """
        self.id = platform_id
        self.mover = mover
        self.controller = controller
        self.properties = properties or {}
        
        # Link children back to parent
        self.mover.platform = self
        if self.controller:
            self.controller.platform = self
            
    def __repr__(self):
        return f"Platform(id={self.id}, mover={self.mover.__class__.__name__})"
