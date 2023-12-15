from scipy import stats
import numpy as np


class ImJuggler:

    def __init__(self):
        bet = 3
        self.out = np.array([3+42+bet, 3+42+bet, 3+42+bet, 3+16+bet, 3+16+bet, 3, 3, 3, 3, 3, 3])
        saf = np.array([294, 294+2, 294+1, 112, 112+2, 14, 10, 8, 2, 3, 0])
        self.saf = saf + np.array([0, -2+(2*14/21), -1+(14/21), -2+(2*14/21), 0, -2+(2*14/21), 0, 0, 0, 0, 0])

    def imjug(self):
        big = 431.16, 422.81, 422.81, 417.43, 417.43, 407.06         
        cbig = 1129.93, 1129.93, 1129.93, 1092.27, 1092.27, 1092.27
        rbig = 2184.53, 2184.53, 2184.53, 1820.44, 1820.44, 1820.44
        reg = 642.51, 590.41, 474.90, 448.88, 364.09, 364.09
        creg = 1394.38, 1236.53, 1092.27, 1057.03, 851.12, 851.12
        bell =  (1092.27,) * 6
        clown = (1092.27,) * 6
        grape = 6.02, 6.02, 6.02, 6.02, 6.02, 5.78
        cherry =  (32.28,) * 6
        replay = (7.298,) * 6
        role = 1/np.array([big, cbig, rbig, reg, creg, bell, clown, grape, cherry, replay])
        # ベル  1/5734, ピエロ 1/14167, チェリー 1/50 へ変換
        role = role * np.array([1, 1, 1, 1, 1, 1092.27/5734, 1092.27/14167, 1, 32.28/50, 1]).reshape(-1, 1)
        loss = 1 - np.sum(role, axis=0) 
        p = np.concatenate([role, [loss]]).T

        return p

    def simulate(self, setting: int, out: int, random_state=42):

        size = 9000
        p = self.imjug()
        xk = np.arange(len(p.T))
        pk = p[setting-1]
        im = stats.rv_discrete(name='im', values=(xk, pk))
        sample = im.rvs(size=size, random_state=random_state)
        list_out = [self.out[x] for x in sample]
        cum = np.cumsum(list_out)
        games = (np.abs(cum - out)).argmin()
        result = sample[:games]
        saf = sum([self.saf[x] for x in result])
        bb = (result < 3).sum()
        rb = ((result > 2) & (result < 5)).sum()

        return bb, rb, games, out, saf

if __name__ == '__main__':
    im = ImJuggler()
    print(im.simulate(6, 12000, random_state=0))
