import re
from pathlib import Path


def master_dir(join_with='', escape_sequence='/', project_name='string_arithmetic_ml'):
    return Path(str_master_dir(join_with, escape_sequence, project_name))


def str_master_dir(join_with='', escape_sequence='/', project_name='string_arithmetic_ml'):
    for parent in Path(__file__).parents.__reversed__():
        if parent.name == project_name:
            if join_with:
                if not re.match('^/|^\\+', join_with):
                    join_with = '\\' + join_with
                final = str(Path(str(parent) + join_with))
            else:
                final = str(parent)
            return re.sub('\\\\', escape_sequence, final)