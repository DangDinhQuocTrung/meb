import numpy as np
from pathlib import Path


def main():
    root_dir = Path("/home/common/general/affective/project/data/MEB")
    paths = root_dir.glob("*.npy")

    for path in paths:
        data = np.load(path)
        print(path.stem, "-", data.shape)
        print()
    return


if __name__ == "__main__":
    main()
