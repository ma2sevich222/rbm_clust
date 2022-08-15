##################################################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Sergey Chekhovskikh / Andrey Tsyrkunov
# Contacts: <chekh@doconsult.ru>
##################################################################################
from backtesting import Strategy
from loguru import logger


class LazyStrategy(Strategy):
    """
      Простая стратегия для тестирования сети.
      'Длинные' и 'короткие' сделки.
    """

    fix_sum = 0
    deal_amount = "capital"
    signal_signature = {"long_signal": 1, "short_signal": -1, "exit_signal": 0}
    preview_signal = 1

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.signal = None

    def __is_long_signal(self, signal: int) -> bool:
        """
        Относиться ли сигнал ко входу в лонг

        :param signal: Входной сигнал
        :return: Ответ
        """
        return self.signal_signature["long_signal"] == signal

    def __is_short_signal(self, signal: int) -> bool:
        """
        Относится ли сигнал ко входу в шорт

        :param signal: Входной сигнал
        :return: Ответ
        """
        return self.signal_signature["short_signal"] == signal

    def __is_exit_signal(self, signal: int) -> bool:
        """
        Относится ли сигнал к выходу из всех позиций

        :param signal: Входной сигнал
        :return: Ответ
        """
        return self.signal_signature["exit_signal"] == signal

    def __deal_size(self, price: float) -> float:
        """
        Метод для определения размера сделки
        
        :param price: Цена покупки 
        :return: Выходной размер сделки
        """ ""
        # Проверяем как выполняются сделки "На всю сумму" или "На фиксированную сумму".
        if self.deal_amount == "fix":
            # Проверяем хватает ли выделенной фиксированной суммы на покупку хотя бы одной акции
            if self.fix_sum // price:
                # Вычисляем размер сделки исходя из стоимости акций и имеющегося капитала
                deal_size = self.fix_sum // price if self.fix_sum <= self.equity else 0
            else:
                logger.warning(str(self.fix_sum) + " " + str(price))
                logger.warning("Деньги закончились :(")
                deal_size = 0
        else:
            deal_size = self.equity // price
        return deal_size

    def init(self):
        """
        Конструктор

        :return: Ничего
        """
        self.signal = self.I(
            lambda x: x, self.data.df.Signal, name="Signal", overlay=False
        )

    def next(self):
        """
        Итератор прохода по всем барам

        :return: Ничего
        """
        last_signal = self.signal[-1]

        if self.__is_long_signal(last_signal):
            if not self.position.is_long:
                self.buy(size=1)

        elif self.__is_short_signal(last_signal):
            if not self.position.is_short:
                self.sell(size=1)

        elif self.__is_exit_signal(last_signal):
            self.position.close()
