import sys
from config.definitions import ROOT_DIR
import os
from tqdm import tqdm
from datetime import datetime

"""
from models.custom_dt import main as main1
from models.custom_nn import main as main2
from models.sklearn_dt import main as main3
from models.sklearn_svm import main as main4
from models.sklearn_knn import main as main5
from models.sklearn_random_forest import main as main6
from models.keras_nn import main as main7

SUMMARIES = {'custom_dt': main1, 
             'custom_nn': main2,
             'sklearn_dt': main3,
             'sklearn_svm': main4,
             'sklearn_knn': main5,
             'sklearn_random_forest': main6,
             'keras_nn': main7,
             }

def generate_summary():
    # load summaries to generate
    to_eval_path = os.path.join(ROOT_DIR, 'data', 'results', 'TO_EVAL.txt')
    with open(to_eval_path, 'r') as f:
        sum_names = [name.strip() for name in f.readlines()]

    # generate summaries
    for name in tqdm(sum_names):
        out_path = os.path.join(ROOT_DIR, 'data', 'results', f'{name}_output.txt')
        sys.stdout = open(out_path, 'w')
        SUMMARIES[name]()
        sys.stdout.close()
"""

def generate_summary(**kwargs):
    for key, val in kwargs.items():
        if key == 'filepath': 
            filepath = val;
        elif key == 'name': 
            name = val
            s = f"Model: {name.replace('_', ' ').title()}\n"
            s += f"Training Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            s += "-"*10 + '\n\n'
        else: s += f"{key.__str__().replace('_', ' ').title()}\n{'-'*len(key.__str__())}\n{val.__str__()}\n\n"
    
    with open(f'{filepath}/{name}_results.txt', 'w') as outfile:
        outfile.write(s)

    print(f'Saved model training results at {filepath}/{name}_results.txt')
