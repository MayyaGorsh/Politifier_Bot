from transformers import Trainer
from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
import warnings
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is deprecated.*")

# Natasha: для извлечения NER
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

def extract_named_entities(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    return {span.text for span in doc.spans}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Отделяем метки
        labels = inputs.pop("labels")

        # Получаем стандартный loss от модели
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # Генерация ответов моделью
        if hasattr(self, 'tokenizer'):
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            source_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)
            generated_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

            # Подсчёт штрафа за пропущенные NER
            ner_penalty = 0.0
            for src, gen in zip(source_texts, generated_texts):
                src_ents = extract_named_entities(src)
                gen_ents = extract_named_entities(gen)
                missing = src_ents - gen_ents
                ner_penalty += len(missing) * 0.1  # коэффициент штрафа за каждую пропущенную сущность

            loss += ner_penalty

        return (loss, outputs) if return_outputs else loss
