import numpy as np


if __name__ == '__main__':
    a = np.array([1, 0, 3])
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1

    print()