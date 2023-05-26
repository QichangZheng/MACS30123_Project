import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

df = pd.read_json('dataset_zh_train.json', lines=True)
num = len(df) / size
df = df.iloc[round(rank * num): round((rank + 1) * num)]

Models = [
    'uer/roberta-base-finetuned-jd-binary-chinese',
    'uer/roberta-base-finetuned-dianping-chinese',
    'philschmid/distilbert-base-multilingual-cased-sentiment',
    'philschmid/distilbert-base-multilingual-cased-sentiment-2',
]
tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in Models]
models = [AutoModelForSequenceClassification.from_pretrained(model_name) for model_name in Models]
# nlps = [pipeline('sentiment-analysis', model=model_name) for model_name in Models]
# def get_score(text):
#     scores = []
#     results = [nlp(text) for nlp in nlps]
#     for result in results:
#         if result[0]['label'] == 'positive (stars 4 and 5)' or result[0]['label'] == 'positive':
#             scores.append(result[0]['score'])
#         else:
#             scores.append(1 - result[0]['score'])
#     return scores

# print(get_score('sdf'))

def get_score(text):
    sentiment_scores = []
    for i in range(len(Models)):
        tokenizer = tokenizers[i]
        model = models[i]
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            logits = model(**encoded_input).logits
            probabilities = torch.softmax(logits, dim=1).squeeze()
            sentiment_scores.extend(probabilities.tolist())
    scores = [round(num, 3) for num in sentiment_scores]
    return [scores[0], scores[2], scores[4], scores[5], scores[7], scores[8]]

# print(get_score('sdf'))
#
# print(111)