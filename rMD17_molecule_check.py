import os
import numpy as np

molecule = input("Molecule name (e.g. aspirin): ").strip()
path = os.path.join("data", "rmd17_"+molecule+".npz")

if not os.path.exists(path):
    print("File does not exist:", path)
else:
    print(f"{os.path.getsize(path) / 1e6:.1f} MB")
    with np.load(path) as data:
        for k in data.files:
            print(k, data[k].shape)