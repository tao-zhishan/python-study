import numpy as np
import math as math

T = [347.56, 158.68, 213.93, 319.17, 1620.92, 100.01, 201.73,
     1315.86, 594.43, 1114.84, 1756.27, 713.67, 1177.87,
     1407.19, 714.16, 37.09, 123.54, 23.25, 108.51, 102.55]

'''*****methode de Newton******'''
print("***methode de Newton***")


class EMV_weibull(object):
    def __init__(self, data):
        self._data = data

    '''@property'''
    def f1(self, beta=1):
        n = len(self._data)
        f = (sum(np.power(self._data, beta) * np.log(self._data))) / (sum(np.power(self._data, beta))) - (
                (1 / n) * sum(np.log(self._data))) - (1 / beta)
        return f

    def f2(self, beta=1):
        f_order_1 = sum(np.power(self._data, beta) * np.power(np.log(self._data), 2)) / sum(
            np.power(self._data, beta)) - (sum(np.power(self._data, beta) * np.log(self._data)) / sum(
            np.power(self._data, beta))) ** 2 + 1 / beta ** 2
        return f_order_1

    def iteration(self):
        max_iteration = 50
        tol = 0.0000001
        count = 1
        beta = 1
        while count <= max_iteration:
            count += 1
            beta1 = beta
            beta = beta - self.f1(beta) / self.f2(beta)
            if abs(beta - beta1) <= tol:
                # print("beta chapeau = ", beta)
                break

        print('nbr de iteration = ', count)
        if count == 51 and beta - beta1 > tol:
            print("pas de solution")
        return beta




if __name__ == '__main__':
    beta_chapeau = EMV_weibull(T).iteration()
    #print("beta_cahpeau = ",beta_chapeau)

    #beta_chapeau = iteration(T, beta=1)
    alpha_chapeau = np.power((1 / len(T) * sum(np.power(T, beta_chapeau))), 1 / beta_chapeau)
    print("beta chapeau = ", beta_chapeau)
    print("alpha chapeau = ", alpha_chapeau)
    z = 1.96
    alpha = [alpha_chapeau - 1.053 * z * alpha_chapeau / (math.sqrt(len(T)) * beta_chapeau),
             alpha_chapeau + 1.053 * z * alpha_chapeau / (math.sqrt(len(T)) * beta_chapeau)]
    self = [beta_chapeau - 0.78 * z * beta_chapeau / math.sqrt(len(T)),
            beta_chapeau + 0.78 * z * beta_chapeau / math.sqrt(len(T))]
    print('IC(alpha) = ', alpha)
    print('IC(beta) = ', self)
    print('')


    '''*****iteration*****'''
    self = 1
    beta_mem = 0
    tol = 0.0000001
    count = 0
    while (count <= 50):
        count += 1
        beta_mem = self
        self = 1 / (sum(np.power(T, self) * np.log(T)) / sum(np.power(T, self)) - (1 / len(T)) * sum(np.log(T)))
        if abs(self - beta_mem) <= tol:
            # print("beta chapeau = ", beta)
            break

    beta_chapeau = self
    alpha_chapeau = np.power(1 / len(T) * sum(np.power(T, beta_chapeau)), 1 / beta_chapeau)
    print("***iteration***")
    print("nbr de iteration = ", count)
    print("beta chapeau = ", beta_chapeau)
    print("alpha chapeau = ", alpha_chapeau)
