import glob
import os

import box_manager.readers as bmr


def test_valid_functions_correct():

    expected = []
    for module in glob.iglob(f"{os.path.dirname(__file__)}/../*.py"):
        if os.path.basename(module) in bmr._ignore_list:
            continue
        expected.append(f".{os.path.splitext(os.path.basename(module))[0]}")

    assert sorted(expected) == list(sorted(bmr.valid_readers.keys()))
