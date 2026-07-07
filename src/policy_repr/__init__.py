"""Policy representation learning for two-player zero-sum imperfect-information games."""

import sys as _sys

from policy_repr import utils as _utils

# Compatibility shim: the external `iig_rl_benchmark` PSRO code does a bare
# `from utils import UniformRandomAgent`, expecting the host project to provide a
# top-level `utils` module (its own `utils.py` does not define UniformRandomAgent).
# The pre-restructure flat layout satisfied this because `utils.py` sat at the repo
# root and was importable as top-level `utils`. Expose ours under that name so the
# benchmark resolves it regardless of the current working directory.
_sys.modules.setdefault("utils", _utils)
