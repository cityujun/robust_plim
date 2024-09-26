from dataclasses import dataclass

@dataclass
class Node:
    __slots__ = ('index', 'pp')
    index: int
    pp: float
