import sys

def test_environment():
    import numpy
    import pandas
    assert sys.version_info >= (3, 11)