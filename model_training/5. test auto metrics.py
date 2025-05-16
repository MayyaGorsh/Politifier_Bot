import json
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from bert_score import score as bert_score
from sacrebleu.metrics import CHRF
from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
import torch
from transformers import pipeline

# Инициализация NER
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)


def extract_named_entities(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    return {span.text for span in doc.spans}


def evaluate_model(model, tokenizer, dataset_path, max_tokens=128, batch_size=8):
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


base_model_name = 'ai-forever/ruT5-base'
old_model_name = 's-nlp/ruT5-base-detox'
new_model_path_1 = "ruT5-base-detox-polite"
new_model_path_2 = "ruT5-base-detox-polite-NER"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
old_model = T5ForConditionalGeneration.from_pretrained(old_model_name)
new_model_1 = T5ForConditionalGeneration.from_pretrained(new_model_path_1)
new_model_2 = T5ForConditionalGeneration.from_pretrained(new_model_path_2)

print(f'Метрики для ruT5-base-detox (старая модель):')
metrics = evaluate_model(old_model, tokenizer, "test.jsonl")
print(metrics)
print()

print(f'Метрики для ruT5-base-detox-polite (новая модель):')
metrics = evaluate_model(new_model_1, tokenizer, "test.jsonl")
print(metrics)
print()

print(f'Метрики для ruT5-base-detox-polite-NER (новая модель с Natasha):')
metrics = evaluate_model(new_model_2, tokenizer, "test.jsonl")
print(metrics)
print()

