import os
from dataclasses import dataclass, field
from general_utils import validate_dir

@dataclass
class Specimen_Image():
    run_name: str = ''