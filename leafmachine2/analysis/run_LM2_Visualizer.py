import streamlit.web.cli as stcli
import os, sys


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    dir_home = os.path.dirname(__file__)

    # pip install protobuf==3.20.0

    sys.argv = [
        "streamlit",
        "run",
        resolve_path(os.path.join(dir_home,"LM2_Visualizer.py")),
        "--global.developmentMode=false",
        "--server.port=8599",

    ]
    sys.exit(stcli.main())