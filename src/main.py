print('\nLoading project with all dependencies and setting PYTHONPATH...\n')

# path setup
import sys
import os
import warnings 

sys.path.insert(0, os.path.abspath(''))
warnings.filterwarnings("ignore")

# own scripts
from scripts.utils import generate_summary
from preprocess import run_preprocessing, run_check_data
from eda import run_eda
from evaluate import run_evaluation
from models import run_custom_dt, run_custom_nn, run_sklearn_dt, run_sklearn_random_forest, run_keras_nn

# helper
def guided_run(f, section_desc):
    while True:
        ans = input(f'Do you wish to {section_desc}? (y/n) ')
        if ans == 'y':
            print()
            f() 
            print("\nSuccessfully run.\n")
            break
        elif ans == 'n':
            print("Didn't run.\n")
            break
        else:
            print("Invalid input. Try again.")

def main():
    # -- check data quality -----------------------------------------------
    guided_run(run_check_data, 'check the quality of the data')
    
    # -- data preprocessing -----------------------------------------------
    guided_run(run_preprocessing, 'preprocess the data')

    # -- eda -----------------------------------------------
    guided_run(run_eda, 'run the EDA')

    # -- assess correctness -----------------------------------------------
    guided_run(run_evaluation, 'assess the correctness of the custom implementations.')

    # -- build and evaluate models -----------------------------------------------
    guided_run(run_custom_dt, 'train the custom decision tree classifier on the data')
    guided_run(run_sklearn_dt, 'train the sklearn decision tree classifier on the data')
    guided_run(run_custom_nn, 'train the custom neural net on the data')
    guided_run(run_keras_nn, 'train the keras neural net on the data')
    guided_run(run_sklearn_random_forest, 'train the sklearn random forest on the data')

    print('Whole Project Pipeline done.')

if __name__ == '__main__':
    main()
