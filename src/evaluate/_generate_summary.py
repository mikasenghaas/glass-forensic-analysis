import sys
from .test_script import test
from config.definitions import ROOT_DIR
import os

SUMMARIES = {
    'test': test,
    'test2': test
}

def generate_summary():
    
    # Load summaries to generate
    to_eval_path = os.path.join(ROOT_DIR, 'data', 'results', 'TO_EVAL.txt')
    with open(to_eval_path, 'r') as f:
        sum_names = [name.strip() for name in f.readlines()]

    # Generate summaries
    for name in sum_names:
        out_path = os.path.join(ROOT_DIR, 'data', 'results', f'{name}_output.txt')
        sys.stdout = open(out_path, 'w')
        SUMMARIES[name]()
        sys.stdout.close()
