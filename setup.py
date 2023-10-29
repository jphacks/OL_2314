# -*- coding: utf-8 -*-
"""EmoClass_FineTurning

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CjwKLlU-l6JrwjLJJ8n7nfkcm-Z29S_F
"""
import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed with error: {e.stderr}")
        return None

commands=['pip install pandas-bokeh', 
          'pip install japanize-matplotlib', 
          'pip install transformers datasets', 
          'pip install fugashi ipadic', 
          'pip install transformers[torch]',
          'pip install accelerate -U',
          'pip install -r requirements.txt', 
          'wget https://github.com/ids-cv/wrime/raw/master/wrime-ver1.tsv -P data/']
# コマンド実行
for command in commands:
    run_command(command=command)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from datasets import load_metric
import torch
import os
import numpy as np
import pandas as pd


# データセット読み込み
df_wrime = pd.read_table('data/wrime-ver1.tsv')
df_wrime.info()

# Plutchikの8つの基本感情
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']

# 客観感情の平均（"Avg. Readers_*"） の値をlist化し、新しい列として定義する
df_wrime['readers_emotion_intensities'] = df_wrime.apply(lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1)

# 感情強度が低いサンプルは除外する
# 感情強度は、無、弱、中、強で、ぞれぞれ0〜3で分類されている.
# (readers_emotion_intensities の max が２以上のサンプルのみを対象とする)
is_target = df_wrime['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
df_wrime_target = df_wrime[is_target]

# train / test に分割する
df_groups = df_wrime_target.groupby('Train/Dev/Test')
df_train = df_groups.get_group('train')
df_test = pd.concat([df_groups.get_group('dev'), df_groups.get_group('test')])
print('train :', len(df_train))  # train : 17104
print('test :', len(df_test))    # test : 1133


# 使用するモデルを指定して、トークナイザとモデルを読み込む
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=8)


# 1. Transformers用のデータセット形式に変換
# pandas.DataFrame -> datasets.Dataset
target_columns = ['Sentence', 'readers_emotion_intensities']
train_dataset = Dataset.from_pandas(df_train[target_columns])
test_dataset = Dataset.from_pandas(df_test[target_columns])

# 2. Tokenizerを適用（モデル入力のための前処理）
def tokenize_function(batch):
    """Tokenizerを適用 （感情強度の正規化も同時に実施する）."""
    tokenized_batch = tokenizer(batch['Sentence'], truncation=True, padding='max_length')
    tokenized_batch['labels'] = [x / np.sum(x) for x in batch['readers_emotion_intensities']]  # 総和=1に正規化
    return tokenized_batch

train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

del checkpoint, tokenizer
del df_wrime, df_groups, df_train, df_test
del train_dataset, test_dataset

# 評価指標を定義
# https://huggingface.co/docs/transformers/training
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    label_ids = np.argmax(labels, axis=-1)
    return metric.compute(predictions=predictions, references=label_ids)

# 訓練時の設定
# https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/trainer#transformers.TrainingArguments
training_args = TrainingArguments(
    output_dir='trainer',
    per_device_train_batch_size=8,
    num_train_epochs=1.0,
    evaluation_strategy="steps",  # 200ステップ毎にテストデータで評価する
    eval_steps=200)

# Trainerを生成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
    compute_metrics=compute_metrics,
)

# 訓練を実行
trainer.train()

# モデル保存
# モデルのパラメータを取得
parameters = list(model.parameters())

# モデルのパラメータをテンソルに変換
parameters_tensor = torch.cat([p.view(-1) for p in parameters])

# テンソルを分割
chunks = torch.chunk(parameters_tensor, chunks=8)
save_dir = '/static'

torch.save(model.to('cpu'), os.path.join(save_dir, 'model.pt'))