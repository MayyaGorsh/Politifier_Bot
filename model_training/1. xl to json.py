# ШАГ 1: преобразовать данные из екселя в словарь input output


import pandas as pd
import json
import os
from colorama import Fore, Style

file_path = os.path.join("..", "data_generation", "data_by_hand.xlsx")
xls = pd.ExcelFile(file_path)

# Считываем оба листа (первый - грубые, второй - вежливые)
rude_df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None)
polite_df = pd.read_excel(xls, sheet_name=xls.sheet_names[1], header=None)

# Проверяем, что в обоих листах одинаковое количество строк
assert len(rude_df) == len(polite_df), "Количество строк в двух листах не совпадает!"

# Создаём список словарей для Hugging Face Datasets
dataset = [{"input": rude, "output": polite} for rude, polite in zip(rude_df.iloc[:, 0], polite_df.iloc[:, 0])]

# Сохраняем в JSONL (формат, который можно загрузить в Hugging Face Datasets)
output_path = "formatted_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(Fore.GREEN + f"Файл сохранён в формате Hugging Face Datasets: {output_path}")