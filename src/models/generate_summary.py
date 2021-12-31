import os
import sys

sys.path.insert(0, os.path.abspath(''))

from scripts.utils._generate_summary import generate_summary

if __name__ == '__main__':
    generate_summary()
