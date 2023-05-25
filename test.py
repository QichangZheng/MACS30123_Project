print(1)
import transformers
print(2)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
print(3)
import torch
from mpi4py import MPI
print(4)


Models = [
    'uer/roberta-base-finetuned-jd-binary-chinese',
    'uer/roberta-base-finetuned-dianping-chinese',
    'philschmid/distilbert-base-multilingual-cased-sentiment',
    'philschmid/distilbert-base-multilingual-cased-sentiment-2',
]
# tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in Models]
# models = [AutoModelForSequenceClassification.from_pretrained(model_name) for model_name in Models]
nlps = [pipeline('sentiment-analysis', model=model_name) for model_name in Models]
print(5)
def get_score(text):
    scores = []
    results = [nlp(text) for nlp in nlps]
    for result in results:
        if result[0]['label'] == 'positive (stars 4 and 5)' or result[0]['label'] == 'positive':
            scores.append(result[0]['score'])
        else:
            scores.append(1 - result[0]['score'])
    return scores

print(6)





# def get_score(text):
#     sentiment_scores = []
#     for i in range(len(Models)):
#         tokenizer = tokenizers[i]
#         model = models[i]
#         encoded_input = tokenizer(text, return_tensors='pt')
#         with torch.no_grad():
#             logits = model(**encoded_input).logits
#             probabilities = torch.softmax(logits, dim=1).squeeze()
#             sentiment_scores.extend(probabilities.tolist())
#     return [round(num, 3) for num in sentiment_scores]

# print(get_score('sdf'))
#
# print(111)