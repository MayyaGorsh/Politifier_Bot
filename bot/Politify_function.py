from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

# Загружаем модель один раз при импорте
MODEL_PATH = "../model_training/ruT5-base-detox-polite"
BASE_MODEL_NAME = "ai-forever/ruT5-base"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)


def Politify(initial_message, max_length=200):
    input_ids = tokenizer.encode(initial_message, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,  # один ответ
            do_sample=False  # модель всегда выдаёт наиболее вероятный вариант
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
