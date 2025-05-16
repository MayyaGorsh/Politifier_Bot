# ШАГ 2: разделить тест и трейн
# токенизировать,
# найти 90-95 перцентиль длины текстов в датасете, чтобы все тексты добивались до одной max_length

import json
from sklearn.model_selection import train_test_split
from colorama import Fore, Style


input_file = "formatted_dataset.jsonl"
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]


# Разделяем на train (80%) и temp (20%) (валидация + тест)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
# Разделяем temp на validation (10%) и test (10%)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)


# Функция сохранения в JSONL
def save_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Сохраняем раздельные файлы
save_jsonl("train.jsonl", train_data)
save_jsonl("valid.jsonl", valid_data)
save_jsonl("test.jsonl", test_data)

print(Fore.GREEN + f"Разделение выполнено:")
print(f"Train: {len(train_data)} примеров")
print(f"Validation: {len(valid_data)} примеров")
print(f"Test: {len(test_data)} примеров")

