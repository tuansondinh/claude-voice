# lazy_claude/__init__.py
#
# IMPORTANT: stdout_guard MUST be the very first import.
# It redirects fd 1 before any other code can accidentally write to it.
from lazy_claude import stdout_guard  # noqa: F401  — side-effect import

__version__ = "0.1.0"
