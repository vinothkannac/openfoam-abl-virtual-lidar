import subprocess, sys, pathlib

def test_self_checks():
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("test_suite.py"))], capture_output=True, text=True)
    assert "ALL PASSED" in r.stdout, r.stdout + r.stderr
