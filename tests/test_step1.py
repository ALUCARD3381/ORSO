import orso_core


def test_core_import():
    assert orso_core.hello() == "ORSO core OK"


if __name__ == "__main__":
    test_core_import()
    print("PASS: orso_core import/teste OK")
