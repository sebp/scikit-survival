import re

import sksurv


def test_show_versions(capsys):
    sksurv.show_versions()
    captured = capsys.readouterr()

    assert "SYSTEM" in captured.out
    assert "DEPENDENCIES" in captured.out

    # check required dependency
    assert re.search(r"numpy\s*:\s([0-9\.\+a-f]|dev)+\n", captured.out)
    assert re.search(r"pandas\s*:\s([0-9\.\+a-f]|dev)+\n", captured.out)
    assert re.search(r"sklearn\s*:\s([0-9\.\+a-f]|dev)+\n", captured.out)

    # check optional dependency
    assert re.search(r"matplotlib\s*:\s([0-9\.]+|None)\n", captured.out)
