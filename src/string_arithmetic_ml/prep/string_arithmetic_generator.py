from collections import namedtuple
import random
import re
import numpy as np
from pathlib import Path
import json
from string_arithmetic_ml.prep.utility import master_dir


# <editor-fold desc="Parameters">
# PARAM:
max_length: int = 1
int_size: int = 9

# CONSTS
simple_max_unit_length = (
                          ((len(str(int_size)) + 1) * 2) +  # 2 numbers (each number being 2 digits and a negative sign so 3 here)
                          3 +  # whitespace op whitespace
                          3  # second operator
                  ) * max_length - 3  # last doesnt have 2nd op
# In practice max_unit_length = 117 with all other parameters at default values, but this is more dynamic

number_encoded_max_unit_length = (  # ie: 2 + 2 + would result in 4 tokens
    2  # 2 Numbers
    + 2  # 2 operator
) * max_length - 1  # last number has no trailing operator

operators = ['+', '-', '*', '/']
function_times = {}


# </editor-fold>


# <editor-fold desc="Utility Functions">
def coinflip(): return random.choice((True, False))


def random_operator(): return random.choice(operators)


# </editor-fold>


# <editor-fold desc="Sample Generation">
Sample = namedtuple('Sample', ['problem', 'solution'])


def generate() -> str:
    """
    Generates a random string of integers up to and including the int_size param paired with random operators from
    the operator's list. The integers will randomly be either positive or negative. Each integer and each operator
    are spaced from each other with a space.

    Example (with int_size of 10): generate() => 1 + -7 // 2 * 10 * -3
    :return:
    """
    result = ''
    iterations = random.randint(1, max_length)
    for cycle in range(iterations):
        second_operator = f' {random_operator()} ' if cycle != iterations - 1 else ''
        neg1 = '-' if coinflip() else ''
        neg2 = '-' if coinflip() else ''
        result += f'{neg1}{random.randint(1, int_size)}' \
                  f' {random_operator()} ' \
                  f'{neg2}{random.randint(1, int_size)}' \
                  f'{second_operator}'
    return result


def solve(problem: str) -> int:
    if re.search('[a-zA-Z]', problem):
        raise Exception('Words in mathematical eval, danger of hack.')
    return eval(problem.replace('/', '//'))


def get_a_data() -> Sample:
    problem = generate()
    return Sample(problem, solve(problem))


# </editor-fold>


# <editor-fold desc="Encoder">
encoder_map = {
    '0': 0b0000,
    '1': 0b0001,
    '2': 0b0010,
    '3': 0b0011,
    '4': 0b0100,
    '5': 0b0101,
    '6': 0b0110,
    '7': 0b0111,
    '8': 0b1000,
    '9': 0b1001,
    '+': 0b1010,
    '-': 0b1011,
    '/': 0b1100,
    '*': 0b1101,
    ' ': 0b1110,
}

def fill_vector(vector: list, to: int, fill_char):
    from_ = len(vector)
    space_thats_left = to - from_
    for num in range(space_thats_left):
        vector.append(fill_char if not hasattr(fill_char, '__call__') else fill_char(num, from_))
    return vector

def simple_encoder(equation_expression: str) -> list[tuple[int, int]]:
    """
    Returns a hot encoded list (vector) with each character in the string being hot encoded by `encoder_map` as well as
    the character's position (encoded_char, position) and then pads the rest of the vector with these tuples but
    for the empty space character.
    :param equation_expression: String equation to encode
    :return: [(encoded_char, position), ...]
    """
    result = [(encoder_map[char], index) for index, char in enumerate(equation_expression)]
    result = fill_vector(result, simple_max_unit_length, lambda from_, current_index: (0b1111, from_ + current_index))
    return result

def integer_encoder_base10(integer: str):
    is_negative = int(integer.startswith('-'))
    if is_negative:
        integer = integer[1:]
    magnitude = len(integer) - 1
    return is_negative, magnitude, int(integer) / 10**magnitude

# [position x, is op 1234 no op is 0, 01 is direction, x is magnitude, 0-9 is value] filler space [pos,5,0,0,0]
def number_based_encoder(equation_expression):
    # TODO abstract this way too much hard coded in here
    result = []
    operator_map = {operator: index+1 for index, operator in enumerate(operators)}
    equation_expression += ' '  # For the regex to work
    for index, find in enumerate(re.finditer(r'([+\-*/])\s|\s?([\-0-9]+)\s', equation_expression)):
        if find.group(1):
            result.append([index, operator_map[find.group(1)], 0, 0, 0])
        elif find.group(2):
            result.append([index, 0, *integer_encoder_base10(find.group(2))])
    return fill_vector(
        result,
        number_encoded_max_unit_length,
        lambda starting_size, current_index: [starting_size + current_index, 5, 0, 0, 0]
    )

# </editor-fold>


# <editor-fold desc="Sample Aggregation">
def generate_dataset(size: int) -> list[Sample]:
    return [get_a_data() for _ in range(size)]


def dataset_to_np_array(dataset):
    return np.array([sample[0] for sample in dataset]), [sample[1] for sample in dataset]


# </editor-fold>


# <editor-fold desc="Store Dataset as Json">
def fix_path_for_json(path, overwrite):
    path = Path(path)
    if not path.stem == '.json':
        path = Path(f'{path}.json')
    if not overwrite and path.exists():
        raise FileExistsError(f'{path} to save string arithmetic data already exists. If you wish to delete the file'
                              f'please set overwrite=True')

def save(dataset: list[Sample], path, overwrite=False):
    fix_path_for_json(path, overwrite)
    pickled = json.dumps([list(sample) for sample in dataset])
    with open(path, 'w') as f:
        f.write(pickled)


def load(path):
    with open(path, 'r') as f:
        loaded_data = json.loads(f.read())
    return [Sample(sample[0], sample[1]) for sample in loaded_data]


default_save_path = master_dir('cache/dataset.json')
# </editor-fold>


if __name__ == '__main__':
    save(generate_dataset(100), master_dir('cache/simple_dataset.json'))
