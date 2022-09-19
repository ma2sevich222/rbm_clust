from forward.generators import WindowDaysGenerator
from tqdm import tqdm


class ForwardAnalysis:
    """
    Класс для форвардного переобучения
    """

    def __init__(self, dataset, train_window, test_window, start_test_point, first_train_window=None):
        """
        Конструктор форвардного анализа
        :param dataset: Датасет для анализа
        :param config: Файл конфигурации с настройкаим
        """
        self.window_generator = WindowDaysGenerator(
            dataset, train_window, test_window,
            start_test_point, first_train_window
        )

    def run(self):
        """
        Конструктор форвардного анализа времянного ряда
        :return: Ничего
        """
        for train_window, test_window in self.window_generator.run():#, desc="Forward Analysis"):
            yield train_window, test_window
