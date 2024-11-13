import unittest
import pandas as pd
import numpy as np
import torch
from gensim.models import FastText
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/data"))
)

from make_dataset import (
    load_and_preprocess_data,
    get_text_corpus,
    build_embedding_dict,
    convert_texts_to_embeddings,
    FastTextDataset,
    prepare_data_loaders,
)


class TestMakeDataset(unittest.TestCase):
    @patch("make_dataset.pd.read_csv")
    @patch("make_dataset.word_tokenize")
    def test_load_and_preprocess_data(self, mock_word_tokenize, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame(
            {"Category": ["spam", "ham"], "Message": ["Hello world", "Test message"]}
        )
        mock_word_tokenize.side_effect = lambda x: x.lower().split()

        df = load_and_preprocess_data("dummy_path")
        self.assertEqual(df["Category"].tolist(), [1, 0])
        self.assertEqual(
            df["Message"].tolist(), [["hello", "world"], ["test", "message"]]
        )

    def test_get_text_corpus(self):
        texts = [["hello", "world"], ["test", "message"]]
        corpus = get_text_corpus(texts)

        self.assertEqual(set(corpus), {"hello", "world", "test", "message"})

    def test_build_embedding_dict(self):
        corpus = ["hello", "world"]
        model = MagicMock()
        model.wv = {"hello": np.array([1, 2]), "world": np.array([3, 4])}

        embedding_dict = build_embedding_dict(corpus, model)

        self.assertTrue(np.array_equal(embedding_dict["hello"], np.array([1, 2])))
        self.assertTrue(np.array_equal(embedding_dict["world"], np.array([3, 4])))

    def test_convert_texts_to_embeddings(self):
        texts = np.array([["hello", "world"], ["test", "message"]])
        embedding_dict = {
            "hello": np.array([1, 2]),
            "world": np.array([3, 4]),
            "test": np.array([5, 6]),
            "message": np.array([7, 8]),
        }

        embeddings = convert_texts_to_embeddings(texts, embedding_dict)

        self.assertTrue(np.array_equal(embeddings[0], np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.array_equal(embeddings[1], np.array([[5, 6], [7, 8]])))

    def test_fasttext_dataset(self):
        X = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        y = np.array([0, 1])
        dataset = FastTextDataset(X, y, max_seq_length=3)

        self.assertEqual(len(dataset), 2)

        X_padded, y_tensor, seq_length = dataset[0]
        self.assertTrue(
            torch.equal(
                X_padded, torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.float32)
            )
        )
        self.assertTrue(torch.equal(y_tensor, torch.tensor(0, dtype=torch.float32)))
        self.assertEqual(seq_length, 2)

    @patch("make_dataset.FastText")
    @patch("make_dataset.load_and_preprocess_data")
    def test_prepare_data_loaders(self, mock_load_and_preprocess_data, mock_fasttext):
        mock_load_and_preprocess_data.return_value = pd.DataFrame(
            {"Category": [1, 0], "Message": [["hello", "world"], ["test", "message"]]}
        )
        mock_fasttext.return_value = MagicMock()
        mock_fasttext.return_value.wv = {
            "hello": np.array([1, 2]),
            "world": np.array([3, 4]),
            "test": np.array([5, 6]),
            "message": np.array([7, 8]),
        }

        train_loader, test_loader = prepare_data_loaders(
            "dummy_path", batch_size=1, emb_dim=2
        )

        self.assertEqual(len(train_loader.dataset), 1)
        self.assertEqual(len(test_loader.dataset), 1)


if __name__ == "__main__":
    unittest.main()
