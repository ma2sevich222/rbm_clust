##################################################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Sergey Chekhovskikh / Andrey Tsyrkunov
# Contacts: <chekh@doconsult.ru>
##################################################################################
import pandas as pd

from backtesting import Strategy
from datetime import datetime, timedelta


class LazyStrategy(Strategy):
    """
      Простая стратегия для тестирования сети.
      'Длинные' и 'короткие' сделки.
    """
    fix_sum = 0
    deal_amount = 'capital'
    signal_signature = {'long_signal': 1, 'short_signal': -1, 'exit_signal': 2}
    preview_signal = 1
    clearing = False
    stop_loss = None
    take_profit = None
    time_frame = None
    patterns = {
        "1:-1": -1,
        "-1:1": 1,
        "1:0": 2,
        "-1:0": 2,
        "0:-1": -1,
        "0:1": 1,
    }
    wo_clearing_signals_path = None

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.signal = None
        self.previous_deal = None

        if self.wo_clearing_signals_path is not None:
            df = self.__replace_signals_to_clearing(data.df)
            self.__save_signals_to_file(df, filename=self.wo_clearing_signals_path)

    def __replace_signals_to_clearing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляем к сигналам нули по краям клиринга, чтобы выход происходил автоматически

        :param df: Датафрейм с сигналами
        :return: Новый датафрейм
        """
        for index in range(df.shape[0]):
            bar_datetime = df.index[index]
            start_clearing, end_clearing = self.get_clearing_time(bar_datetime)
            if start_clearing <= bar_datetime.strftime("%H:%M") <= end_clearing:
                df.iloc[index, -1] = 0
        return df

    @staticmethod
    def __save_signals_to_file(df: pd.DataFrame, filename: str) -> None:
        """
        Сохраняем новый датафрейм

        :param df: Датафрейм с сигналами
        :return: Ничего
        """
        df.to_csv(filename)

    def __is_long_signal(self, first_deal: bool, previous_signal: int, signal: int) -> bool:
        """
        Относиться ли сигнал ко входу в лонг

        :param signal: Входной сигнал
        :return: Ответ
        """
        pattern = signal if first_deal else self.patterns.get(f"{previous_signal}:{signal}", 0)
        return self.signal_signature['long_signal'] == pattern

    def __is_short_signal(self, first_deal: bool, previous_signal: int, signal: int) -> bool:
        """
        Относится ли сигнал ко входу в шорт

        :param signal: Входной сигнал
        :return: Ответ
        """
        pattern = signal if first_deal else self.patterns.get(f"{previous_signal}:{signal}", 0)
        return self.signal_signature['short_signal'] == pattern

    def __is_exit_signal(self, previous_signal: int, signal: int) -> bool:
        """
        Относится ли сигнал к выходу из всех позиций

        :param signal: Входной сигнал
        :return: Ответ
        """
        pattern = self.patterns.get(f"{previous_signal}:{signal}", 0)
        return self.signal_signature['exit_signal'] == pattern

    def __deal_size(self, price: float) -> float:
        """
        Метод для определения размера сделки
        
        :param price: Цена покупки 
        :return: Выходной размер сделки
        """""
        # Проверяем как выполняются сделки "На всю сумму" или "На фиксированную сумму".
        if self.deal_amount == 'fix':
            # Проверяем хватает ли выделенной фиксированной суммы на покупку хотя бы одной акции
            if self.fix_sum // price:
                # Вычисляем размер сделки исходя из стоимости акций и имеющегося капитала
                deal_size = self.fix_sum // price if self.fix_sum <= self.equity else 0
            else:
                print(str(self.fix_sum) + " " + str(price))
                print('Деньги закончились :(')
                deal_size = 0
        else:
            deal_size = self.equity // price
        return deal_size

    def init(self):
        """
        Конструктор

        :return: Ничего
        """
        self.signal = self.I(lambda x: x, self.data.df.Signal, name='Signal', overlay=False)

    def get_clearing_time(self, bar_time: datetime):
        clearing_dates = {
            "2019": ("03-10", "11-03"),
            "2020": ("03-08", "11-01"),
            "2021": ("03-14", "11-07"),
            "2022": ("03-13", "11-06")
        }
        current_date = bar_time.strftime("%m-%d")
        if self.time_frame == 60:
            start_clearing, end_clearing = datetime(year=1970, month=1, day=1, hour=19, minute=00), \
                                           datetime(year=1970, month=1, day=1, hour=22, minute=00)
        else:
            start_clearing, end_clearing = datetime(year=1970, month=1, day=1, hour=20, minute=30 - self.time_frame), \
                                           datetime(year=1970, month=1, day=1, hour=22, minute=30 - self.time_frame)

        start_new_time, end_new_time = clearing_dates[str(bar_time.year)]
        # Проверяем дату, чтобы поменять зимнее на летнее время
        if not (start_new_time < current_date < end_new_time):
            # зимнее время
            start_clearing += timedelta(hours=1)
            end_clearing += timedelta(hours=1)

        return start_clearing.strftime("%H:%M"), end_clearing.strftime("%H:%M")

    def next(self):
        """
        Итератор прохода по всем барам

        :return: Ничего
        """
        previous_signal = self.signal[-2]
        last_signal = int(self.signal[-1])
        price = self.data.Close[-1]
        first_deal = len(self.signal) == 2

        # Если у нас имеется уже сделка, которая может быть "открытой"
        # после клиринга, нужно ее обновить
        if self.previous_deal is not None:
            previous_signal = 0
            self.previous_deal = None

        start_clearing, end_clearing = self.get_clearing_time(self.data.index[-1])
        if self.clearing and (start_clearing <= self.data.index[-1].strftime("%H:%M") <= end_clearing):
            self.position.close()
            self.previous_deal = 0

        elif self.__is_long_signal(first_deal, previous_signal, last_signal):
            if not self.position.is_long:
                sl = None if self.stop_loss is None else price + self.stop_loss
                tp = None if self.take_profit is None else price + self.take_profit
                self.buy(size=1, sl=sl, tp=tp)

        elif self.__is_short_signal(first_deal, previous_signal, last_signal):
            if not self.position.is_short:
                sl = None if self.stop_loss is None else price - self.stop_loss
                tp = None if self.take_profit is None else price - self.take_profit
                self.sell(size=1, sl=sl, tp=tp)

        elif self.__is_exit_signal(previous_signal, last_signal):
            self.position.close()
