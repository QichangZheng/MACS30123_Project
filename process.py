from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

filename = 'dataset_zh_train'
df = pd.read_json(filename + '.json', lines=True)[:100]
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

def get_score(text):
    try:
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
    except:
        return [1, 1, 1, 0, 1, 0]

df['scores'] = df['review_body'].apply(get_score)

all_data = comm.gather(df['scores'].tolist(), root=0)

if rank == 0:
    # 在主节点上操作
    all_scores = [score for sublist in all_data for score in sublist]
    df = pd.read_json(filename + '.json', lines=True)[:100]
    result_df = pd.DataFrame(all_scores, columns=['score1', 'score2', 'score3', 'score4', 'score5', 'score6'])
    df = pd.concat([df, result_df], axis=1)
    df.to_csv('processed_' + filename + '.csv', index=False)
    print('Done!')