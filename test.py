import os
import shutil
import argparse
from leafmachine2.machine.machine import machine
from leafmachine2.machine.general_utils import load_config_file_testing


def test_LM2(redownload=False):
    # Set LeafMachine2 dir
    dir_home = os.path.dirname(__file__)

    # If there's an error in the code that causes the ML model placement, the easiest way to fix it is to
    # delete everything in /bin and start over
    if redownload:
        bin_dir = os.path.join(dir_home, "bin")
        if os.path.exists(bin_dir):
            shutil.rmtree(bin_dir)
            print(f"Deleted contents of {bin_dir}")
        else:
            print(f"{bin_dir} does not exist. No contents to delete.")

    cfg_file_path = os.path.join(dir_home, "demo", "demo.yaml")

    cfg_testing = load_config_file_testing(dir_home, cfg_file_path)
    cfg_testing["leafmachine"]["project"]["dir_images_local"] = os.path.join(
        dir_home,
        cfg_testing["leafmachine"]["project"]["dir_images_local"][0],
        cfg_testing["leafmachine"]["project"]["dir_images_local"][1],
    )
    cfg_testing["leafmachine"]["project"]["dir_output"] = os.path.join(
        dir_home,
        cfg_testing["leafmachine"]["project"]["dir_output"][0],
        cfg_testing["leafmachine"]["project"]["dir_output"][1],
    )

    machine(cfg_file_path, dir_home, cfg_testing)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LeafMachine2 with optional redownload."
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Delete the contents of the /bin folder before running.",
    )

    args = parser.parse_args()

    test_LM2(redownload=args.redownload)
