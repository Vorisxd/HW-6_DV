import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from collections.abc import Iterable


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Загрузит данные из CSV файла, преобразовывает метки в бинарные значения и токенизирует сообщения.

    Параметры:
    - filepath (str): Путь к CSV файлу с данными.

    Возвращает:
    - df (pd.DataFrame): DataFrame с предобработанными столбцами 'Category' и 'Message'.
    """
    df = pd.read_csv(filepath)
    df["Category"] = df["Category"].map({"spam": 1, "ham": 0})
    df["Message"] = df["Message"].apply(lambda x: word_tokenize(x.lower()))
    return df


def get_text_corpus(texts: Iterable[list]) -> list:
    """
    Создаёт корпус уникальных слов из токенизированных текстов.

    Параметры:
    - texts (Iterable[list]): Итерируемый объект токенизированных текстов.

    Возвращает:
    - corpus (list): Список уникальных слов в корпусе.
    """
    corpus = {word for text in texts for word in text}
    return list(corpus)


def build_embedding_dict(corpus: list, model) -> dict:
    """
    Построить словарь, сопоставляющий слова их эмбеддингам с помощью обученной модели.

    Параметры:
    - corpus (list): Список слов в корпусе.
    - model: Обученная модель FastText.

    Возвращает:
    - dict: Словарь, сопоставляющий слова их векторным эмбеддингам.
    """
    return {word: model.wv[word] for word in corpus}


def convert_texts_to_embeddings(texts: np.ndarray, embedding_dict: dict) -> list:
    """
    Преобразовать токенизированные тексты в эмбеддинги с использованием словаря эмбеддингов.

    Параметры:
    - texts (np.ndarray): Массив токенизированных текстов.
    - embedding_dict (dict): Словарь, сопоставляющий слова эмбеддингам.

    Возвращает:
    - list: Список numpy массивов, представляющих эмбеддинги текстов.
    """
    return [np.array([embedding_dict[word] for word in text]) for text in texts]


class FastTextDataset(Dataset):
    """
    Пользовательский Dataset для работы с эмбеддингами текстов и метками.

    Параметры:
    - X (list): Список numpy массивов с эмбеддингами.
    - y (np.ndarray): Массив меток.
    - max_seq_length (int, optional): Максимальная длина последовательности для паддинга. По умолчанию 400.

    Методы:
    - __len__: Возвращает общее количество образцов.
    - __getitem__: Получает образец и его метку по заданному индексу.
    """

    def __init__(self, X: list, y: np.ndarray, max_seq_length: int = 400):
        self.X = X
        self.y = y
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X_tensor = torch.tensor(self.X[index], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[index], dtype=torch.float32)
        seq_length = X_tensor.size(0)
        padding = self.max_seq_length - seq_length
        X_padded = nn.functional.pad(X_tensor, (0, 0, 0, padding))
        return X_padded, y_tensor, seq_length


def prepare_data_loaders(data_path: str, batch_size: int, emb_dim: int) -> tuple:
    """
    Подготавливает загрузчики данных для обучения и тестирования с использованием модели FastText.
    Функция выполняет следующие шаги:
    - Загружает и предобрабатывает данные из указанного пути.
    - Создает корпус текстов из сообщений.
    - Разделяет данные на обучающую и тестовую выборки.
    - Инициализирует и обучает модель FastText на обучающих данных.
    - Строит словарь эмбеддингов на основе обученной модели.
    - Преобразует тексты в эмбеддинги для обучающей и тестовой выборок.
    - Создает наборы данных FastTextDataset и соответствующие загрузчики DataLoader.
    Параметры:
        data_path (str): Путь к файлу с данными.
        batch_size (int): Размер пакета для загрузчиков данных.
        vector_size (int): Размерность векторов эмбеддингов.
    Возвращает:
        tuple: Кортеж из обучающего и тестового загрузчиков данных (train_dataloader, test_dataloader).
    """

    df = load_and_preprocess_data(data_path)
    corpus = get_text_corpus(df["Message"].values)
    X_train, X_test, y_train, y_test = train_test_split(
        df["Message"].values, df["Category"].values, test_size=0.2, random_state=42
    )

    ft_model = FastText(vector_size=emb_dim, window=3, min_count=1, seed=42)
    ft_model.build_vocab(corpus_iterable=df["Message"].values)
    ft_model.train(corpus_iterable=X_train, total_examples=len(X_train), epochs=10)

    emb_dict = build_embedding_dict(corpus, ft_model)
    X_train_emb = convert_texts_to_embeddings(X_train, emb_dict)
    X_test_emb = convert_texts_to_embeddings(X_test, emb_dict)

    train_dataset = FastTextDataset(X_train_emb, y_train)
    test_dataset = FastTextDataset(X_test_emb, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader
