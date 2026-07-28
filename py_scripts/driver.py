from __future__ import annotations

import os
import sys

from fv3_init_driver import init_driver
from fv3_restart_driver import restart_driver
from fv3_runtime import handle_errors
from fv3_utils import exit_code


def main() -> None:
    if int(os.environ.get("CASE_RESUBMIT_INDEX", "0")) > 0:
        restart_driver()
    else:
        init_driver()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        exit_code(code)
        raise
    except BaseException:
        handle_errors(*sys.exc_info())
        exit_code(1)
        sys.exit(1)
    else:
        exit_code(0)
