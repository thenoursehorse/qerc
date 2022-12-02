from pathlib import Path
from distutils.util import strtobool
import argparse

import numpy as np

from encoder.qubit_mnist import PCAQubits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, required=True)
    parser.add_argument('-filename_root', required=True)
    parser.add_argument('-save', type=lambda x: bool(strtobool(x)), default='True')

    args = parser.parse_args()
    print(args)

    Path(args.filename_root).mkdir(parents=True, exist_ok=True)

    # Get pca data of digits
    input_data = PCAQubits(N=args.N, filename=args.filename_root)

    print("Finished.")
