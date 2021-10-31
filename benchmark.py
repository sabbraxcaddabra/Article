import time
import matplotlib.pyplot as plt
from statistics import mean, pstdev, median
from tqdm import tqdm

#TODO: Написать бенчмарк с возможностью сравнения скорости выполнения нескольких функций(для сравнения jit и non-jit версии расчета)

def benchmark(iters = 1000, file_to_write = None, show_graphics = False):
    """
    Функция возвращающая декоратор с параметрами, указанными выше
    :param iters: int Число итераций
    :param file_to_write: str Файл записи, по умолчанию None(печать в консоль)
    :param show_graphics: bool Выводить/Не выводить график распределения (по умолчанию False - не выводить
    :return: Декоратор
    """
    def actual_decorator(func):
        """
        Декоратор для замера измерения времени выполнения функции
        :param func: Функция для которой необходимо измерить время
        :return: Задекорированнную функцию
        """
        def wrapper(*args, **kwargs):
            time_list = []
            for _ in tqdm(range(iters)):
                start = time.perf_counter()
                ret_val = func(*args, **kwargs)
                end = time.perf_counter()
                time_list.append(end - start)
            text = [
                f"Первый запуск: {time_list[0]} сек",
                "*"*30,
                f"Среднее время выполнения без учета первого запуска: {mean(time_list[1:])} сек",
                f"Среднее время выполнения с учетом первого запуска: {mean(time_list)} сек",
                "*"*30,
                f"Стандартное отклонение без учета первого запуска: {pstdev(time_list[1:])} сек",
                f"Стандартное отклонение с учетом первого запуска: {pstdev(time_list)} сек",
                "*"*30,
                f"Медиана без учета первого запуска: {median(time_list[1:])} сек",
                f"Медиана с учетом первого запуска: {median(time_list)} сек",
                "*"*30,
                f"Общее время без первого запуска: {round(sum(time_list[1:]), 3)} сек",
                f"Общее время с первым запуском: {round(sum(time_list), 3)} сек"
            ]
            text = "\n".join(text)
            if file_to_write:
                with open(file_to_write, "w", encoding="utf8") as f:
                    print(text, file=f)
            else:
                print(text)
            if show_graphics:
                fig, ax = plt.subplots(figsize=(15, 10))
                ax.hist(time_list[1:], bins=25)
                ax.set(
                    title="Распределение времени выполнения функции без учета первого запуска")
                ax.set_xlabel('Время, сек')
                plt.show()
            return ret_val
        return wrapper
    return actual_decorator
