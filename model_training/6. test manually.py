from transformers import T5ForConditionalGeneration, AutoTokenizer
from colorama import Fore, Style
import json
import torch

base_model_name = 'ai-forever/ruT5-base'
new_model_path_1 = "ruT5-base-detox-polite"
new_model_path_2 = "ruT5-base-detox-polite-NER"


tokenizer = AutoTokenizer.from_pretrained(base_model_name)
new_model_1 = T5ForConditionalGeneration.from_pretrained(new_model_path_1)
new_model_2 = T5ForConditionalGeneration.from_pretrained(new_model_path_2)

sentences = []
with open("test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        sentences.append(json.loads(line))


def detoxify_text(model, tokenizer, text, max_length=50):
    input_ids = tokenizer.encode(text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


with open('test_results.csv', 'w') as res_file:
    print('Токсичный текст', 'Ожидаемый вывод', 'Новая модель', 'Новая модель c Natasha', file=res_file)
k = 0

for sentence in sentences:
    new_output_1 = detoxify_text(new_model_1, tokenizer, sentence['input'])
    new_output_2 = detoxify_text(new_model_2, tokenizer, sentence['input'])
    # print(Fore.GREEN + 'Токсичный текст:'.ljust(30) + sentence['input'])
    # print("Ожидаемый вывод:".ljust(30) + sentence['output'])
    # print(Fore.YELLOW + "Новая модель:".ljust(30) + new_output_1)
    # print(Fore.LIGHTYELLOW_EX + "Новая модель c Natasha:".ljust(30) + new_output_2)
    # print('')
    k += 1
    print(k)
    with open('test_results.csv', 'a') as res_file:
        res = [sentence['input'], sentence['output'], new_output_1, new_output_2]
        print(Style.RESET_ALL + ';;'.join(res), file=res_file)
