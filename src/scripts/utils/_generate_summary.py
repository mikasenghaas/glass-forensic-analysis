import sys
from config.definitions import ROOT_DIR
import os
from tqdm import tqdm
from datetime import datetime

def generate_summary(**kwargs):

    """
    Generate summary for the given model. The summary is written to the specified ``filepath``.

    Parameters
    ----------
    kwargs : dict-like object
        Dict of parameters neeeded in order for summary to be built.
    """

    for key, val in kwargs.items():
        if key == 'filepath': 
            filepath = val
        elif key == 'name': 
            name = val
            s = f"Model: {name.replace('_', ' ').title()}\n"
            s += f"Training Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            s += "-"*10 + '\n\n'
        else: s += f"{key.__str__().replace('_', ' ').title()}\n{'-'*len(key.__str__())}\n{val.__str__()}\n\n"
    
    with open(f'{filepath}/{name}_results.txt', 'w') as outfile:
        outfile.write(s)

    print(f'Saved model training results at {filepath}/{name}_results.txt')
