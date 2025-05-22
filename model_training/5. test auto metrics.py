import json
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer, MT5ForConditionalGeneration, MBartForConditionalGeneration
from bert_score import score as bert_score
from sacrebleu.metrics import CHRF
from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
import torch
from transformers import pipeline

flag = False
device = torch.device("cuda" if torch.cuda.is_available() and flag else "cpu")

# Инициализация NER
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

test_phrases = [
    "что за фигню ты 8го на совещании ляпнул? А как мне то быть если Ваня ниче не делает. Я завтра не приду, "
    "сам решай наши задачи.",
    "08.03 не приду",
    "ну что за жесть почему опять у вас ничего не получается",
    "Нет, я не собираюсь ехать. Пусть кто-то другой этим занимается.",
    "Всё, вопрос закрыт: испытательный срок три месяца, и никто не сделает исключений.",
    "сделайте пожалуста как можно быстрее",
    "Рабочий день же закончился, какого хрена я всё ещё тут?",
    "Гениальная инициатива, как и все предыдущие – бесполезная и ненужная.",
    "Ты опять забыл про сроки для семинара? Это уже начинает раздражать. Все материалы должны быть готовы за два дня, "
    "а у нас до сих пор нет даже списка участников. Как ты вообще планируешь всё успеть?",
    "Отвали"
]


def add_para(line):
    return 'paraphrase politely: ' + line


def extract_named_entities(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    return {span.text for span in doc.spans}


def evaluate_model(model, tokenizer, dataset_path, max_tokens=128, batch_size=8, mbart=False):
    # Чтение датасета с прогрессбаром
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading dataset"):
            data.append(json.loads(line))

    inputs = [d["input"] for d in data]
    references = [d["output"] for d in data]

    # Генерация с батчингом и прогрессбаром
    predictions = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="Generating outputs (8 sent * 125 iter)"):
        batch_inputs = inputs[i:i+batch_size]
        inputs_tokenized = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
        inputs_tokenized = {k: v.to(device) for k, v in inputs_tokenized.items()}
        if mbart:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs_tokenized,
                    forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang],
                    max_length=max_tokens,
                    num_beams=4,
                    do_sample=True,
                    temperature=1.0,
                )
            predictions += [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
        else:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs_tokenized["input_ids"],
                    attention_mask=inputs_tokenized["attention_mask"],
                    max_new_tokens=max_tokens
                )
            for g in generated_ids:
                predictions.append(tokenizer.decode(g, skip_special_tokens=True))
    print(f"Всего сгенерировано: {len(predictions)} / {len(inputs)}")

    # BERTScore
    print("Calculating BERTScore...")
    _, _, F1 = bert_score(predictions, references, lang="ru", model_type="xlm-roberta-large")
    bert_f1 = float(F1.mean())

    # ChrF
    print("Calculating ChrF...")
    chrf = CHRF()
    chrf_score = chrf.corpus_score(predictions, [references]).score

    # NER retention
    print("Calculating NER Retention...")
    total_ents, preserved_ents = 0, 0
    for src, pred in tqdm(zip(inputs, predictions), total=len(predictions), desc="NER Matching"):
        ents_src = extract_named_entities(src)
        ents_pred = extract_named_entities(pred)
        total_ents += len(ents_src)
        preserved_ents += len(ents_src & ents_pred)
    ner_retention = preserved_ents / total_ents if total_ents > 0 else 1.0

    # Toxicity
    print("Evaluating Toxicity...")
    toxicity_pipe = pipeline(
        "text-classification",
        model="s-nlp/russian_toxicity_classifier",
        tokenizer="s-nlp/russian_toxicity_classifier",
        top_k=None
    )

    scores = []
    for pred in tqdm(predictions, desc="⚠️ Evaluating toxicity"):
        result = toxicity_pipe(pred)[0]
        toxic_score = next((x['score'] for x in result if x['label'] == 'toxic'), 0.0)
        scores.append(toxic_score)

    avg_toxicity = sum(scores) / len(scores)

    # Результаты
    return {
        "bert_f1": round(bert_f1, 4),
        "chrf": round(chrf_score, 2),
        "ner_retention": round(ner_retention, 4),
        "avg_toxicity": round(avg_toxicity, 4)
    }


def show_examples(model, tokenizer, prompts, max_tokens=128):
    print("-" * 60)
    print("Примеры детоксификации:")
    for i, text in enumerate(prompts, 1):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens
            )
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"{i}. TOXIC: {text}\n   DETOX:   {output}\n")


candidates = {
    "ruT5-base": 'ai-forever/ruT5-base',
    "detox": 's-nlp/ruT5-base-detox',
    "paraphraser": 'cointegrated/rut5-base-paraphraser',
    "mt5-base": 'google/mt5-base',
    "mbart-detox": "s-nlp/mbart-detox-en-ru",
    "rut5-detox-v2": "orzhan/rut5-base-detox-v2",
    "mbart-multilingual-detox": "textdetox/mbart-detox-baseline",
    "ruT5-base-detox-polite (новая модель)": "ruT5-base-detox-polite",
    "ruT5-base-detox-polite-NER (новая модель с Natasha)": "ruT5-base-detox-polite-NER",
}

for name, path in candidates.items():
    if "mbart" in name:

        model = MBartForConditionalGeneration.from_pretrained(path).to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
        tokenizer.src_lang = "ru_RU"
        tokenizer.tgt_lang = "ru_RU"

        print(f"🔎 Модель: {name}")
        metrics = evaluate_model(model, tokenizer, "test.jsonl", mbart=True)
        print(metrics)

        def show_examples_mbart(model, tokenizer, prompts, max_tokens=128):
            print("-" * 60)
            print(f"Примеры детоксификации ({name}):")
            for i, text in enumerate(prompts, 1):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang],
                        max_length=max_tokens,
                        num_beams=4,
                        do_sample=True,
                        temperature=1.0,
                    )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"{i}. TOXIC: {text}\n   DETOX:   {decoded}\n")


        show_examples_mbart(model, tokenizer, test_phrases)

    elif name == 'mt5-base':
        model = MT5ForConditionalGeneration.from_pretrained(path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"🔎 Модель: {name}")
        metrics = evaluate_model(model, tokenizer, "test.jsonl")
        print(metrics)
        show_examples(model, tokenizer, list(map(add_para, test_phrases)))

    else:
        model = T5ForConditionalGeneration.from_pretrained(path).to(device)
        if 'новая' in name:
            tokenizer = AutoTokenizer.from_pretrained('ai-forever/ruT5-base')
        else:
            tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"🔎 Модель: {name}")
        metrics = evaluate_model(model, tokenizer, "test.jsonl")
        print(metrics)
        show_examples(model, tokenizer, test_phrases)
    print()




