# ШАГ 3:
# токенизировать,
# найти 90-95 перцентиль длины текстов в датасете, чтобы все тексты добивались до одной max_length


import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
from colorama import Fore, Style

dataset = load_dataset("json", data_files="train.jsonl")
base_model_name = 'ai-forever/ruT5-base'
tokenizer = AutoTokenizer.from_pretrained(base_model_name)


# Функция для вычисления перцентилей
def calculate_percentile_length(dataset, percentile=95):
    all_lengths = []
    for sample in dataset["train"]:
        input_length = len(tokenizer(sample["input"])["input_ids"])
        target_length = len(tokenizer(sample["output"])["input_ids"])
        all_lengths.append(input_length)
        all_lengths.append(target_length)
    return int(np.percentile(all_lengths, percentile))


# Находим 90-95 перцентиль длины текстов
percentile_90 = calculate_percentile_length(dataset, 90)
percentile_95 = calculate_percentile_length(dataset, 95)

max_length = percentile_95
print(Fore.GREEN + f"90%% = {percentile_90}, 95%% = {percentile_95}")
print(f"Выбранный max_length: {max_length}")


# Функция препроцессинга с динамическим max_length
def preprocess_function(samples):
    inputs = [text for text in samples["input"]]
    targets = [text for text in samples["output"]]

    # Используем найденный max_length
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=max_length)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=max_length)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Токенизируем датасет
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets["train"].to_json("tokenized_train.jsonl")
print("Файл сохранен: tokenized_train.jsonl")

dataset = load_dataset("json", data_files="valid.jsonl")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets["train"].to_json("tokenized_valid.jsonl")
print("Файл сохранен: tokenized_valid.jsonl")


