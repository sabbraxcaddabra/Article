from abc import ABC, abstractmethod
import numpy as np
from numpy.random import uniform
from InteriorBallistics import ArtSystem, Powder, Projectile, IntBalParams, solve_ib
from benchmark import benchmark

# TODO: Оформить определение целевой функции, функции вывода и тд в отдельный файл
# TODO: Добавить методы для удаления ограничений 1-го и 2-го рода по ключу
# TODO: Проверить качество и правильность реализации метода случайного сканирования
# TODO: Добавить еще несколько реализацций алгоритмов оптимизации
# TODO: Написать нормальную документацию для каждого класса

def max_speed_t_func(x_vec, params):
    y, p_mean_max, p_sn_max, p_kn_max, lk = solve_ib(*params.create_params_tuple())
    return -y[0], (p_mean_max, p_sn_max, p_kn_max, lk)

def out_bal_func(x_vec, f, sol, params):
    print(f"Масса снаряда: {params.proj.q = } кг")
    for powd in params.charge:
        print(f"Масса пороха {powd.name}: {powd.omega}")
    print(f"Дульная скорость: {-round(f, 1)} м/с")
    print(f"Максимальное среднебаллистическое давление: {round(sol[0] * 1e-6, 2)} МПа")
    print(f"Максимальное давление на дно снаряда: {round(sol[1] * 1e-6, 2)} МПа")
    print(f"Максимальное давление на дно канала ствола: {round(sol[2] * 1e-6, 2)} МПа")
    print(f"Координата полного сгорания порохового заряда {round(sol[3], 4)} м")
    print("*"*30+'\n')

def adapt_proj_mass(x, params):
    params.proj.q = x[0]

def adapt_powders_mass(x, params):
    for i, powd in enumerate(params.charge):
        powd.omega = x[i+1]

class Optimizer(ABC):
    """
    Абстрактный класс-оптимайзер.
    Конкретные реализации алгоритмов оптимизации должны наследоваться от данного класса
    """

    def __init__(self,
                 x_vec,
                 params = None,
                 adapters = [],
                 first_ground_boundary = dict(),
                 second_ground_boundary = dict(),
                 x_lims = None,
                 t_func = None,
                 out_func = None):
        self.x_vec = x_vec
        self.params = params
        self.adapters = adapters
        self.first_ground_boundary = first_ground_boundary
        self.second_ground_boundary = second_ground_boundary
        self.x_lims = x_lims
        self.t_func = t_func
        self.out_func = out_func

    def add_new_adapter(self, adapt_func) -> None:
        """
        Метод для добавления в задачу нового адаптера
        :param adapt_func: Функция, лямбда-функция, классовый метод и т.д(callable объект)
        :return: None
        """
        self.adapters.append(adapt_func)

    def _adapt(self, x_vec_new: list) -> None:
        """
        Метод адаптирует параметры задачи(подставляет значения x_vec_new в необходимые поля params для решения целевой функции
        :param x_vec_new: Новый вектор варьируемых параметров X
        :return: None
        """
        if self.adapters:
            for func in self.adapters:
                func(x_vec_new, self.params)

    def add_first_ground_boundary(self, name: str, func_dict: dict) -> None:
        """
        Метод для добавления функций - ограничений первого рода
        :param name: Название ограничения первого рода(оптимизатором не используется, необходимо для удобства пользователя
        и возможности проще удалить при необходимости)
        :param func_dict: Словарь с ключами func и lims, где func соответсвтует функция, лямбда и тд, которая принимат в себе параметры
        задачи и сравнивает необходимые поля параметров или их преобразования с lims
        :return: None
        """
        self.first_ground_boundary[name] = func_dict

    def _check_first_ground_boundary(self, x_vec_cur):
        """
        Проверка ограничений 1-го рода. Если словарь ограничений пуст, проверяется только вхождение каждой компоненты
        x_vec_cur в ограничения заданные x_lims
        :param x_vec_cur: Текущая реализация вектора варьируемых параметров
        :return: bool
        """
        if self.first_ground_boundary:
            check_list = [func_dict['func'](x_vec_cur, self.params, func_dict['lims']) for func_dict in self.first_ground_boundary]
        else:
            if len(self.x_lims) != len(x_vec_cur):
                raise Exception("Длина вектора варьируемых параметров не совпадает с длиной вектора ограничений")
            else:
                check_list = [lim[0] <= x <= lim[1] for lim, x in zip(self.x_lims, x_vec_cur)]
        return all(check_list)

    def add_second_ground_boundary(self, name: str, func_dict: dict) -> None:
        """
        Добавление функций-ограничений второго рода
        :param name: Название ограничения первого рода(оптимизатором не используется, необходимо для удобства пользователя
        и возможности проще удалить при необходимости)
        :param func_dict: Словарь с ключами func и lims, где func соответсвтует функция, лямбда и тд, которая принимат в себея параметры
        задачи и решение целевой функции и сравнивает необходимые поля решения с lims
        :return: None
        """
        self.second_ground_boundary[name] = func_dict

    def _check_second_ground_boundary(self, solution):
        """
        Проверка ограничений 2-го рода
        :param solution: Текущий результат решения целевой функции
        :return: bool
        """
        check_list = []
        if self.second_ground_boundary:
            check_list = [func_dict['func'](solution, self.params, func_dict['lim']) for func_dict in self.second_ground_boundary.values()]
            return all(check_list)
        else:
            return True

    def set_target_func(self, t_func) -> None:
        """
        Установка целевой функции
        :param t_func: Целевая функция(callable)
        :return:
        """
        self.t_func = t_func
    def set_out_func(self, o_func):
        self.out_func = o_func

    @abstractmethod
    def optimize(self):
        pass

class RandomScanOptimizer(Optimizer):
    """
    Класс-наследник оптимизатора
    Реализация алгоритма случайного сканирования
    """
    def _jump(self, x, i):
        """
        Расчет следующего приближения x
        :param x: Вектор варьируемых параметров
        :param i: Модификатор шага
        :return: Новое приближение x
        """
        ai = np.array([(lim[0] + lim[1])/2 for lim in self.x_lims])
        bi = np.array([abs(lim[0] - lim[1])/2 for lim in self.x_lims])
        x = ai + bi*uniform(-1./i, 1./i, len(x))
        return x

    #@benchmark(iters=200, file_to_write="opt_benc_200_jited.txt", make_graphics=True)
    def optimize(self, N = 50, max_modifier = 8, min_delta_f = 0.):
        """
        Реализация алгоритма оптимизации
        :param N: Максимальное число неудачных шагов
        :param max_modifier: Максимальный модификатор щага
        :param min_delta_f: Минимальное уменьшение целевой функции
        :return: last_x Оптимальное значение вектора x_vec
        """
        last_x = self.x_vec[:]
        self._adapt(last_x)
        last_f, last_second_ground_boundary = self.t_func(last_x, self.params)
        bad_steps_cur = 0 # Счетчик неудачных шагов
        cur_step_modifier = 1

        while bad_steps_cur < N and cur_step_modifier <= max_modifier:
            xx = self._jump(last_x, cur_step_modifier)
            self._adapt(xx)
            if self._check_first_ground_boundary(xx):
                try:
                    cur_f, cur_solution = self.t_func(xx, self.params)
                    if self._check_second_ground_boundary(cur_solution):
                        if cur_f <= last_f and abs(cur_f - last_f) > min_delta_f:
                            last_f, last_x = cur_f, xx[:]
                            if self.out_func:
                                self.out_func(xx, cur_f, cur_solution, self.params)
                            bad_steps_cur = 0
                        else:
                            bad_steps_cur += 1
                    else:
                        bad_steps_cur += 1
                except:
                    bad_steps_cur += 1
            else:
                bad_steps_cur += 1

            if bad_steps_cur == N and cur_step_modifier < max_modifier:
                bad_steps_cur = 0
                cur_step_modifier += 1
        return last_x


if __name__ == "__main__":
    artsys = ArtSystem('2А42', 0.000735299, .03, 0.125E-3, 2.263, 0.12, 0.125E-3 / 0.000735299, 1.136)

    proj = Projectile('30-мм', 0.389, 1.)

    int_bal_cond = IntBalParams(artsys, proj, 50e6, 4e6)
    int_bal_cond.add_powder(
        Powder('6/7', 0.07, 1.6e3, 988e3, 2800., 343.8e3, 1.038e-3, 0.236, 1.53, 0.239, 2.26, 0., 0.835, -0.943, 0.))
    int_bal_cond.add_powder(
        Powder('6/7', 0.05, 1.6e3, 988e3, 2800., 343.8e3, 1.038e-3, 0.236, 1.53, 0.239, 2.26, 0., 0.835, -0.943, 0.))

    p_max_dict_func = {'func':lambda sol, params, lim: sol[0] < lim, 'lim': 600e6}

    x_vec = np.array([0.389, 0.07, 0.05])
    opt = RandomScanOptimizer(x_vec,
                              params=int_bal_cond,
                              x_lims=[[0.389, 0.389], [0.07-0.07*0.1, 0.07+0.07*0.1], [0.05-0.05*0.1, 0.07+0.05*0.1]])
    opt.add_new_adapter(adapt_proj_mass)
    opt.add_new_adapter(adapt_powders_mass)
    opt.add_second_ground_boundary("Pmax", p_max_dict_func)
    opt.set_target_func(max_speed_t_func)
    opt.set_out_func(out_bal_func)
    #print(opt.second_ground_boundary.values())
    opt.optimize(N = 100*len(x_vec), min_delta_f=5, max_modifier=10)