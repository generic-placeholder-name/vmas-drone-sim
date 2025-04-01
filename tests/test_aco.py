import pytest
from aco import *

def test_heuristic():
    assert heuristic(3, 1) == 1/3
    assert heuristic(3, 2) == (1/3) * (1/2)
    assert heuristic(4, 0) == (1/4) * (1/0.1)
    
