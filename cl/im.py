from scipy import stats
import numpy as np


def imjug() -> tuple[np.ndarray]:
    """_summary_

    Returns:
        tuple[np.ndarray]: _description_
    """
    big = 273.1, 269.7, 269.7, 259.0, 259.0, 255.0
    reg = 439.8, 399.6, 331.0, 315.1, 255.0, 255.0
    bell =  (1092.27,) * 6
    clown = (1092.27,) * 6
    grape = 6.02, 6.02, 6.02, 6.02, 6.02, 5.78
    cherry =  (32.28,) * 6
    replay = (7.298,) * 6
    role = 1/np.array([big, reg, bell, clown, grape, cherry, replay])
    lose = 1 - np.sum(role, axis=0).reshape(1, -1)  # ハズレ

    p = np.concatenate([role, lose]).T
    bet = 0.75
    out = np.array([3+42+bet, 3+16+bet, 3., 3., 3., 3., 3., 3.])
    saf = np.array([294., 112., 14., 10., 8., 2., 3., 0.])

    return p, out, saf

p, out, saf = imjug()

rates = (p * saf).sum(axis=1) / (p * out).sum(axis=1)
print(rates.reshape(-1, 1) * 100)

if __name__ == '__main__':
    ...
