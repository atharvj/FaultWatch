from importlib.metadata import version

import faultwatch


def test_package_is_installed():
    assert version("faultwatch")
    assert faultwatch.__doc__
