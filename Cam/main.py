# OpenMV mode launcher
# Edit run_mode.txt to one of: detector, classifier

import uos

MODE_FILE = "run_mode.txt"
DEFAULT_MODE = "detector"


def read_mode():
    try:
        with open(MODE_FILE, "r") as f:
            mode = f.read().strip().lower()
            if mode in ("detector", "classifier"):
                return mode
    except Exception:
        pass
    return DEFAULT_MODE


mode = read_mode()
script = "main_detector.py" if mode == "detector" else "main_classifier.py"

print("Launcher mode:", mode)
print("Running script:", script)

try:
    uos.stat(script)
except Exception as e:
    raise Exception("Missing script '" + script + "': " + str(e))

with open(script, "r") as f:
    code = f.read()

exec(code, globals())
