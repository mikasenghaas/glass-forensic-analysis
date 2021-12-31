print()
print('-- Load modules and set path ---------------------------------------------------')

# Path setup
import sys
import os
sys.path.insert(0, os.path.abspath(''))

# Own scripts
from scripts.utils import generate_summary
from preprocess import run_preprocessing
from eda import run_eda

# Flags
RUN_PREPROCESSING = True
RUN_EDA = True
RUN_EVALUATE = False
print('Success!', end='\n\n')

def main():
    
    print('-- Inspect, clean, transform -----------------------------------------------')
    run_preprocessing() if RUN_PREPROCESSING else None
    print('Success!', end='\n\n')

    print('-- EDA ---------------------------------------------------------------------')
    run_eda() if RUN_EDA else None
    print('Success!', end='\n\n')

    print('-- Evaluate ----------------------------------------------------------------')
    generate_summary() if RUN_EVALUATE else None
    print('Success!', end='\n\n')


if __name__ == '__main__':
    main()
