import sys
import os
from scripts.utils import generate_summary

def main():
    
    # -- Path setup --------------------------------------------------------------
    sys.path.insert(0, os.path.abspath(''))

    # -- Evaluate ----------------------------------------------------------------
    generate_summary()


if __name__ == '__main__':
    main()
