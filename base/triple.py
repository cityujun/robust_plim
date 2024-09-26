from dataclasses import dataclass
import heapq


@dataclass
class UtilityTriple:
    __slots__ = ('index', 'cost', 'value')
    index: int
    cost: float
    value: float


@dataclass
class PriorityQueue:
    _queue = []
    _index = 0

    def put(self, item):
        """
        queue composed of (priority, index, item)
        add - since smallest heap by default
        add index to sort by order 
        """
        priority = - item.value
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def get(self):
        return heapq.heappop(self._queue)[-1]

    def qsize(self):
        return len(self._queue)

    def empty(self):
        return True if not self._queue else False

    def top_priority(self):
        if not self._queue:
            return - self._queue[0][0]
        return 0
