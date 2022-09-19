from datetime import datetime
from tqdm import tqdm


class TestWindowDescriptor:

    def __init__(self):
        self.__test_window = 0

    def __set__(self, instance, value):
        if value > 30:
            print("Использование более 30 дней для `test_window` не рекомендуется!")
        self.__test_window = value

    def __get__(self, instance, owner):
        return self.__test_window


class StartTestPointDescriptor:

    def __init__(self):
        self.__start_test_point = 0

    def __set__(self, instance, value):
        if not isinstance(value, str):
            raise TypeError("Используйте тип `str` для переменной `test_start_point`")
        self.__start_test_point = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')

    def __get__(self, instance, owner):
        return self.__start_test_point


class WindowDaysGenerator:
    """
    Класс-генератор для временных рядов по дням
    """
    test_window = TestWindowDescriptor()
    start_test_point = StartTestPointDescriptor()

    def __init__(self, data, train_window, test_window, start_test_point, first_train_window):
        """
        Конструктор генератора окон из времянного ряда
        :param data: Датасет для анализа
        :param config: Конфиг со всей информацией о форварде
        """
        self.data = data
        self.first_train_window = first_train_window
        self.train_window = train_window
        self.test_window = test_window
        self.start_test_point = start_test_point

    def run(self):
        """
        Запускаем ленивую генерацию окон
        :return window: Внутренее окно, по которому сеть сформирует фичи
        """
        previous_train_date = self.start_test_point.strftime("%Y-%m-%d")
        new_train_counter = self.test_window
        train_window_offset = self.first_train_window \
            if self.first_train_window is not None else self.train_window

        previous_start_test_index = 0
        previous_end_train_index = 0

        for index in range(self.data.shape[0]):
            if self.data.index[index] < self.start_test_point:
                previous_start_test_index = index + 1
                previous_end_train_index = index + 1
                continue

            is_now_clearing = WindowDaysGenerator.is_now_clearing(self.data.index[index])
            current_date = self.data.index[index].strftime("%Y-%m-%d")

            if current_date > previous_train_date:
                previous_train_date = current_date
                new_train_counter -= 1
                # Прошло необходимое число дней, обновляем счетчик
                if new_train_counter == 0:
                    new_train_counter = self.test_window

            # Если сейчас клиринг и время подошло, запускаем два окна
            # Или же, если конец датасета
            if (is_now_clearing and new_train_counter == self.test_window) \
                    or index == self.data.shape[0] - 1:
                train_window = previous_end_train_index - train_window_offset, previous_end_train_index
                test_window = previous_start_test_index, index + 1
                previous_end_train_index = test_window[1]
                previous_start_test_index = test_window[1]
                train_window_offset = self.train_window
                # print(self.data.index[train_window[0]], self.data.index[train_window[1]],
                #       self.data.index[test_window[0]], self.data.index[test_window[1]])
                yield train_window, test_window

    @staticmethod
    def is_now_clearing(bar_datetime: datetime):
        """
        Метод для получения клирингового времени
        и текущей дельты изменения в зависимости от времени года
        return: Кортеж с началом и окончанием клиринга, дельта в 0/60 минут
        """
        clearing_dates = {
            "2019": ("03-10", "11-03"),
            "2020": ("03-08", "11-01"),
            "2021": ("03-14", "11-07"),
            "2022": ("03-13", "11-06")
        }
        start_new_time, end_new_time = clearing_dates[str(bar_datetime.year)]
        current_date = bar_datetime.strftime("%m-%d")
        current_time = bar_datetime.strftime("%H:%M")

        # Проверяем дату, чтобы поменять зимнее на летнее время и наоборот
        if start_new_time < current_date < end_new_time:
            # летнее время
            start_clearing, end_clearing = "20:30", "22:30"
        else:
            # зимнее время
            start_clearing, end_clearing = "21:30", "23:30"
        return start_clearing == current_time
