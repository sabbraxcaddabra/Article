import numpy as np
from dataclasses import dataclass
from benchmark import benchmark

# TODO: Прописать логику расчета баллистики по считанным из файла данным об арт.системе, снаряде и порохах
@dataclass
class ArtSystem:
    # Датакласс для данных об артиллерийской системе
    name: str  # Наименование артиллерийской системы
    S: float  # Приведенная площадь канала ствола
    d: float  # Калибр орудия
    W0: float  # Объем зарядной каморы
    l_d: float  # Полный путь снаряда
    l_k: float  # Длина зарядной каморы
    l0: float  # Приведенная длина зарядной каморы
    Kf: float  # Коэффициент слухоцкого


@dataclass
class Projectile:
    # Датакласс для данных о снаряде
    name: str  # Индекс снаряда
    q: float  # Масса снаряда
    i43: float  # Коэф формы по закону сопротивления 1943 года


@dataclass
class Powder:
    # Датакласс для данных о порохе
    name: str  # Марка пороха
    omega: float  # Масса метательного заряда
    rho: float  # Плотность пороха
    f_powd: float  # Сила пороха
    Ti: float  # Температура горения пороха
    Jk: float  # Конечный импульс пороховых газов
    alpha: float  # Коволюм
    teta: float  # Параметр расширения
    Zk: float  # Относительная толщина горящего свода, соответствующая концу горения
    kappa1: float  # 1-я, 2-я и 3-я хар-ки формы пороховых элементов до распада
    lambd1: float
    mu1: float
    kappa2: float  # 1-я, 2-я и 3-я характеристики формы пороховых элементов после распада
    lambd2: float
    mu2: float


class IntBalParams:
    # Класс начальных условий
    # Система "Орудие-заряд-снаряд"
    def __init__(self, syst, proj, P0, PV):
        self.syst = syst  # Арт.система для задачи
        self.proj = proj  # Снаряд для задачи
        self.charge = []  # Метательный заряд
        self.P0 = P0  # Давление форсирования
        self.PV = PV # Давление воспламенителя

    def add_powder(self, powder: Powder) -> None:
        self.charge.append(powder)

    def create_params_tuple(self) -> tuple:
        # Метод для создания исходных данных
        params = [
            self.P0,
            self.PV,
            50e6**0.25,
            self.syst.S,
            self.syst.W0,
            self.syst.l_k,
            self.syst.l0,
            sum(powd.omega for powd in self.charge),
            self.syst.Kf*self.proj.q,
            self.syst.l_d
        ]
        powders = []
        for powder in self.charge:
            tmp = (
                powder.omega,
                powder.rho,
                powder.f_powd,
                powder.Ti,
                powder.Jk,
                powder.alpha,
                powder.teta,
                powder.Zk,
                powder.kappa1,
                powder.lambd1,
                powder.mu1,
                powder.kappa2,
                powder.lambd2,
                powder.mu2,
            )
            powders.append(tmp)
        params.append(tuple(powders))
        return tuple(params)

def P(y, Pv, lambda_khi, S, W0, qfi, omega_sum, psis, powders):
    """

    :param y:
    :param Pv:
    :param lambda_khi:
    :param S:
    :param W0:
    :param qfi:
    :param omega_sum:
    :param psis:
    :param powders:
    :return:
    """
    thet = theta(psis, powders)
    chisl = 0
    znam = 0
    for i in range(len(powders)):
        f = powders[i][2]
        om = powders[i][0]
        delta = powders[i][1]
        alpha = powders[i][5]
        chisl += f*psis[i]*om
        znam += om*(((1-psis[i])/delta) + alpha*psis[i])
    p_mean = Pv + (chisl - thet*y[0]**2 * (qfi/2 + lambda_khi * omega_sum/6))/(W0 - znam + S*y[1])
    p_sn = (p_mean/(1 + (1/3) * (lambda_khi*omega_sum)/qfi))*(y[0] > 0.) + (y[0] == 0.)*p_mean
    p_kn = (p_sn*(1 + 0.5*lambda_khi*omega_sum/qfi))*(y[0] > 0.) + (y[0] == 0.)*p_mean

    return p_mean, p_sn, p_kn

def theta(psis, powders):
    """

    :param psis:
    :param powders:
    :return:
    """
    sum1 = 0
    sum2 = 0
    for i in range(len(powders)):
        sum1 += powders[i][2] * powders[i][0] * psis[i] / powders[i][3]
        sum2 += powders[i][2] * powders[i][0] * psis[i] / (powders[i][3] * powders[i][6])
    if sum2 != 0:
        return sum1 / sum2
    else:
        return 0.4

def psi(z, zk, kappa1, lambd1, mu1, kappa2, lambd2, mu2):
    """

    :param z:
    :param zk:
    :param kappa1:
    :param lambd1:
    :param mu1:
    :param kappa2:
    :param lambd2:
    :param mu2:
    :return:
    """
    if z < 1:
        return kappa1*z*(1 + lambd1*z + mu1*z**2)
    elif 1 <= z <= zk:
        z1 = z - 1
        psiS = kappa1 + kappa1*lambd1 + kappa1*mu1
        return psiS + kappa2*z1*(1 + lambd2*z1 +mu2*z1**2)
    else:
        return 1

def int_bal_rs(y, P0, PV, k50, S, W0, l_k, l_ps, omega_sum, qfi, powders):
    """

    :param y:
    :param P0:
    :param PV:
    :param k50:
    :param S:
    :param W0:
    :param l_k:
    :param l_ps:
    :param omega_sum:
    :param qfi:
    :param powders:
    :return:
    """
    f = np.zeros(2 + len(powders))
    psis = np.zeros(len(powders))
    for i in range(len(powders)):
        psis[i] = psi(y[2 + i], *powders[i][7:])
    lambda_khi = (y[1] + l_k)/(y[1] + l_ps)

    p_mean, p_sn, p_kn = P(y, PV, lambda_khi, S, W0, qfi, omega_sum, psis, powders)
    if y[0] == 0. and p_mean < P0:
        f[0] = 0
        f[1] = 0
    else:
        f[0] = (p_sn*S)/(qfi)
        f[1] = y[0]
    for k in range(len(powders)):
        if p_mean <= 50e6:
            f[2+k] = ((k50*p_mean**0.75)/powders[k][4]) * (y[2+k] < powders[k][7])
        else:
            f[2+k] = (p_mean/powders[k][4]) * (y[2+k] < powders[k][7])
    return f, p_mean, p_sn, p_kn

#@benchmark(iters=1000)#, file_to_write="TimeMeasurments/1000_iters_jitted_intbal_calc_jitted.txt", make_graphics = False)
def solve_ib(P0, PV, k50, S, W0, l_k, l_ps, omega_sum, qfi, l_d, powders, tmax = 1. , tstep = 1e-5):
    """

    :param P0:
    :param PV:
    :param k50:
    :param S:
    :param W0:
    :param l_k:
    :param l_ps:
    :param omega_sum:
    :param qfi:
    :param l_d:
    :param powders:
    :param tmax:
    :param tstep:
    :return:
    """
    y = np.zeros(2+len(powders))
    zk_list = np.array([powders[i][7] for i in range(len(powders))])
    lk = 0. # Координата по стволу, соответсвующая полному сгоранию порохового заряда
    t0 = 0.
    p_mean_max = PV
    p_sn_max = PV
    p_kn_max = PV
    while y[1] <= l_d:
        k1, p_mean1, p_sn1, p_kn1 = int_bal_rs(y, P0, PV, k50, S, W0, l_k, l_ps, omega_sum, qfi, powders)
        k2, p_mean2, p_sn2, p_kn2 = int_bal_rs(y+tstep*k1/2, P0, PV, k50, S, W0, l_k, l_ps, omega_sum, qfi, powders)
        k3, p_mean3, p_sn3, p_kn3 = int_bal_rs(y+tstep*k2/2, P0, PV, k50, S, W0, l_k, l_ps, omega_sum, qfi, powders)
        k4, p_mean4, p_sn4, p_kn4 = int_bal_rs(y+tstep*k3, P0, PV, k50, S, W0, l_k, l_ps, omega_sum, qfi, powders)
        y += tstep*(k1 + 2*k2 + 2*k3 + k4)/6
        t0 += tstep
        p_mean_max = max(p_mean1, p_mean2, p_mean3, p_mean4, p_mean_max)
        p_sn_max = max(p_sn1, p_sn2, p_sn3, p_sn4, p_sn_max)
        p_kn_max = max(p_kn1, p_kn2, p_kn3, p_kn4, p_kn_max)
        if np.all(zk_list <= y[2:]) and lk == 0.:
            lk = y[1]
        if t0 > tmax:
            raise Exception("Превышено максимальное время выстрела\nОшибка расчета!")
    return y, p_mean_max, p_sn_max, p_kn_max, lk


if __name__ == "__main__":
    artsys = ArtSystem('2А42', 0.000735299, .03, 0.125E-3, 2.263, 0.12, 0.125E-3 / 0.000735299, 1.136)

    proj = Projectile('30-мм', 0.389, 1.)

    int_bal_cond = IntBalParams(artsys, proj, 50e6, 4e6)
    int_bal_cond.add_powder(
        Powder('6/7', 0.07, 1.6e3, 988e3, 2800., 343.8e3, 1.038e-3, 0.236, 1.53, 0.239, 2.26, 0., 0.835, -0.943, 0.))
    int_bal_cond.add_powder(
        Powder('6/7', 0.03, 1.6e3, 988e3, 2800., 343.8e3, 1.038e-3, 0.236, 1.53, 0.239, 2.26, 0., 0.835, -0.943, 0.))
    int_bal_cond.add_powder(
        Powder('6/7', 0.02, 1.6e3, 988e3, 2800., 343.8e3, 1.038e-3, 0.236, 1.53, 0.239, 2.26, 0., 0.835, -0.943, 0.))

    y, p_mean_max, p_sn_max, p_kn_max, lk = solve_ib(*int_bal_cond.create_params_tuple())

    print("Печать результатов рачета\n")
    print(f"Дульная скорость: {round(y[0], 1)} м/с")
    print(f"Максимальное среднебаллистическое давление: {round(p_mean_max * 1e-6, 2)} МПа")
    print(f"Максимальное давление на дно снаряда: {round(p_sn_max * 1e-6, 2)} МПа")
    print(f"Максимальное давление на дно канала ствола: {round(p_kn_max * 1e-6, 2)} МПа")
    print(f"Координата полного сгорания порохового заряда {round(lk, 4)} м")