import pathlib
from pymor.core.defaults import load_defaults_from_file
from pymor.core.logger import set_log_levels

MULTI = pathlib.Path(__file__).parent
pymor_defaults = MULTI / "pymor_defaults.py"

load_defaults_from_file(pymor_defaults.as_posix())
set_log_levels({"multi": "INFO"})

__version__ = "0.1"
