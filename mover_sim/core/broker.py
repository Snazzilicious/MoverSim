class EventBroker:
    """
    A simple Publish-Subscribe broker to decouple components (like platforms, 
    engines, and observers) in the simulation.
    """
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, topic: str, callback):
        """
        Subscribe a callable to a specific topic.
        
        Parameters:
            topic: The topic name string.
            callback: A callable to be invoked when the topic is published to.
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        if callback not in self._subscribers[topic]:
            self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback):
        """
        Unsubscribe a callable from a topic.
        """
        if topic in self._subscribers:
            try:
                self._subscribers[topic].remove(callback)
            except ValueError:
                pass

    def publish(self, topic: str, *args, **kwargs):
        """
        Publish an event to all subscribers of a topic.
        
        Parameters:
            topic: The topic name string.
            *args, **kwargs: Arguments passed directly to the callback.
        """
        if topic in self._subscribers:
            # Iterate over a copy of the list in case subscribers unsubscribe during callback
            for callback in list(self._subscribers[topic]):
                callback(*args, **kwargs)
