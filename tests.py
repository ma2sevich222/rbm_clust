from forward import ForwardAnalysis
import pandas as pd
from typing import Union
from os.path import exists


def load_dataset_from_file(path: str) -> Union[pd.DataFrame, None]:
    """
    Метод для загрузки данных торговли + исходные сигналы от стратегии
    :return: Данные OHLCV + сигналы
    """
    assert exists(path), \
        f"Датасет по пути {path} не был найден!"
    df = pd.read_csv(path, index_col="Datetime")
    df.index = pd.to_datetime(df.index)
    return df


def start_test() -> None:
    """
    Метод реализующий выполнение форвардного тестирования
    :return: Ничего
    """
    dataset = load_dataset_from_file("datasets/GC_15min.csv")
    forward = ForwardAnalysis(
        dataset,
        first_train_window=10000, train_window=1000,
        test_window=57, start_test_point="2021-11-01T00:00:00"
    )

    for train_window, test_window in forward.run():
        print(train_window, test_window)


if __name__ == "__main__":
    start_test()
