import queue
import itertools


class EventQueueEmpty(queue.Empty):
    pass


class EventQueue(queue.PriorityQueue):
    """
    Collection of events in a stable and prioritized queue.
    Events with a higher priority are fetched first in a FIFO manner.
    """

    def __init__(self, events=[]):
        super().__init__()
        self._counter = itertools.count()
        for event in events:
            self.put(event)

    def put(self, event):
        super().put((-event.priority, next(self._counter), event))

    def get(self, *args, **kwargs):
        try:
            return super().get(*args, **kwargs)[2]
        except queue.Empty as e:
            raise EventQueueEmpty(e)
