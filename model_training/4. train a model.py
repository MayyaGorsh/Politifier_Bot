# ШАГ 4: трейнить ruT5-base-detox


import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, AutoTokenizer, TrainingArguments, Trainer
from colorama import Fore
from custom_trainer import CustomTrainer


# Определяем параметры модели
base_model_name = "ai-forever/ruT5-base"
model_name = "s-nlp/ruT5-base-detox"
# save_model_name = "ruT5-base-detox-polite-NER"  # Имя сохраненной модели

# Загружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Загружаем датасет
train_dataset = load_dataset("json", data_files="tokenized_train.jsonl")["train"]
eval_dataset = load_dataset("json", data_files="tokenized_valid.jsonl")["train"]
print(Fore.YELLOW + 'датасет загружен')


# Параметры обучения
training_args = TrainingArguments(
    output_dir=save_model_name,
    evaluation_strategy="epoch",    # оценка и сохранение после каждой эпохи
    save_strategy="epoch",
    per_device_train_batch_size=8,   # Подбор в зависимости от GPU. Чем больше тем быстрее. Если вылетает по памяти, уменьши
    per_device_eval_batch_size=8,
    learning_rate=5e-5,     # скорость обучения. переобучается - уменьшь
    num_train_epochs=3,     # пройдет по датасету 3 раза. Если eval_loss начинает расти модель переобучается
    weight_decay=0.01,      # штраф на сложность модели
    logging_dir=f"{save_model_name}/logs",  # куда сохраняет логи
    save_total_limit=1,  # Хранить только 1 последнюю версию модели
    load_best_model_at_end=True
)

# Тренер
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Запуск обучения
trainer.train()

# Сохранение модели
model.save_pretrained(save_model_name)
tokenizer.save_pretrained(save_model_name)

print(Fore.GREEN + f"Fine-tuning завершен! Модель сохранена в: {save_model_name}")
