from typing import Optional
from pydantic import BaseModel
import os
import shutil

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt


class Item(BaseModel):
    prompt: str
    get_dataset:Optional[bool] = False


tf.get_logger().setLevel("ERROR")

def test_getting_dataset():
    print("Getting dataset")
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = tf.keras.utils.get_file(
        "aclImdb_v1.tar.gz", url, untar=True, cache_dir=".", cache_subdir=""
    )

    dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")

    train_dir = os.path.join(dataset_dir, "train")

    # remove unused folders to make it easier to load the data
    remove_dir = os.path.join(train_dir, "unsup")
    shutil.rmtree(remove_dir)


    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=seed,
    )

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.utils.text_dataset_from_directory(
        "aclImdb/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=seed,
    )

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.utils.text_dataset_from_directory(
        "aclImdb/test", batch_size=batch_size
    )

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    for text_batch, label_batch in train_ds.take(1):
        for i in range(3):
            print(f"Review: {text_batch.numpy()[i]}")
            label = label_batch.numpy()[i]
            print(f"Label : {label} ({class_names[label]})")


bert_model_name = "small_bert/bert_en_uncased_L-4_H-512_A-8"

tfhub_handle_encoder = (
    "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1"
)

tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

print(f"BERT model selected           : {tfhub_handle_encoder}")
print(f"Preprocess model auto-selected: {tfhub_handle_preprocess}")

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

bert_model = hub.KerasLayer(tfhub_handle_encoder)



def predict(item, run_id, logger):
    item = Item(**item)
    text_test = ['this is such an amazing movie!']
    text_preprocessed = bert_preprocess_model(text_test)
    print(f'Keys       : {list(text_preprocessed.keys())}')
    print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
    print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
    print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
    print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

    bert_results = bert_model(text_preprocessed)
    print(f'Loaded BERT: {tfhub_handle_encoder}')
    print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
    print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
    print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
    print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

    net = bert_results['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    print(f"Sigmoided sentiment output: {tf.sigmoid(net)}")

    return {"SentimentResult":tf.sigmoid(net).numpy().tolist()}
