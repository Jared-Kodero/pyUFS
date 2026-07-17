from fv3_init_driver import init_driver
from fv3_restart_driver import restart_driver
from fv3_runtime import exit_code
from fv3_setup import preprocess_input
from fv3_state import state


def main():
    preprocess_input()  # Preprocess input and update state with any necessary derived values

    if state.warm_start:
        restart_driver()
    else:
        init_driver()


if __name__ == "__main__":
    # try:
    main()
    exit_code(0)
# except Exception:
#     exit_code(-1)
#     raise
