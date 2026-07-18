from __future__ import annotations

import os

from fv3_init_driver import init_driver
from fv3_restart_driver import restart_driver
from fv3_utils import exit_code


def main() -> None:
    if int(os.environ.get("CASE_RESUBMIT_INDEX", 0)) > 0:
        restart_driver()
    else:
        init_driver()


if __name__ == "__main__":
    try:
        main()
        exit_code(0)
    except Exception:
        exit_code(-1)
