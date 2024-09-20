import streamlit.web.cli as stcli
import os, sys


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    dir_home = os.path.dirname(__file__)
    sys.argv = [
        "streamlit",
        "run",
        resolve_path(os.path.join(dir_home,"tool_paint_bucket.py")),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())