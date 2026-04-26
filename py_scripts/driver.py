from fv3gfs_init_driver import init_driver
from fv3gfs_restart_driver import restart_driver
from fv3gfs_runtime import exit_code
from fv3gfs_setup import preprocess_input
from fv3gfs_state import state


def main():
    preprocess_input()  # Preprocess input and update state with any necessary derived values

    if state.update_nml_only:
        restart_driver()
    else:
        init_driver()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        exit_code(-1)
        raise e
