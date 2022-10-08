import os, yaml, datetime, argparse

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def validate_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def load_cfg(pathToCfg):
    try:
        with open(os.path.abspath(os.path.join(pathToCfg,"LeafMachine2.yaml")), "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    except:
        with open(os.path.join(os.path.dirname(os.path.dirname(pathToCfg)),"LeafMachine2.yaml"), "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    return cfg

def parse_cfg():
    parser = argparse.ArgumentParser(
            description='Parse inputs to read  config file',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    optional_args = parser._action_groups.pop()
    required_args = parser.add_argument_group('MANDATORY arguments')
    required_args.add_argument('--path-to-cfg',
                                type=str,
                                required=True,
                                help='Path to config file - LeafMachine.yaml. Do not include the file name, just the parent dir.')

    parser._action_groups.append(optional_args)
    args = parser.parse_args()
    return args

def get_datetime():
    day = "_".join([str(datetime.datetime.now().strftime("%Y")),str(datetime.datetime.now().strftime("%m")),str(datetime.datetime.now().strftime("%d"))])
    time = "-".join([str(datetime.datetime.now().strftime("%H")),str(datetime.datetime.now().strftime("%M")),str(datetime.datetime.now().strftime("%S"))])
    new_time = "__".join([day,time])
    return new_time

def print_error_to_console(cfg,indent_level,message,error):
    white_space = " " * 5 * indent_level
    if cfg['leafmachine']['print']['optional_warnings']:
        print(f"{bcolors.FAIL}{white_space}{message} ERROR: {error}{bcolors.ENDC}")

def print_warning_to_console(cfg,indent_level,message,error):
    white_space = " " * 5 * indent_level
    if cfg['leafmachine']['print']['optional_warnings']:
        print(f"{bcolors.WARNING}{white_space}{message} ERROR: {error}{bcolors.ENDC}")

def print_bold_to_console(cfg,indent_level,message):
    white_space = " " * 5 * indent_level
    if cfg['leafmachine']['print']['verbose']:
        print(f"{bcolors.BOLD}{white_space}{message}{bcolors.ENDC}")

def print_green_to_console(cfg,indent_level,message):
    white_space = " " * 5 * indent_level
    if cfg['leafmachine']['print']['verbose']:
        print(f"{bcolors.OKGREEN}{white_space}{message}{bcolors.ENDC}")

def print_cyan_to_console(cfg,indent_level,message):
    white_space = " " * 5 * indent_level
    if cfg['leafmachine']['print']['verbose']:
        print(f"{bcolors.OKCYAN}{white_space}{message}{bcolors.ENDC}")

def print_blue_to_console(cfg,indent_level,message):
    white_space = " " * 5 * indent_level
    if cfg['leafmachine']['print']['verbose']:
        print(f"{bcolors.OKBLUE}{white_space}{message}{bcolors.ENDC}")

def print_plain_to_console(cfg,indent_level,message):
    white_space = " " * 5 * indent_level
    if cfg['leafmachine']['print']['verbose']:
        print(f"{white_space}{message}")

# def set_yaml(path_to_yaml, value):
#     with open('file_to_edit.yaml') as f:
#         doc = yaml.load(f)

#     doc['state'] = state

#     with open('file_to_edit.yaml', 'w') as f:
#         yaml.dump(doc, f)