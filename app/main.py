from fastapi import FastAPI, Query, HTTPException
import uvicorn
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import normalize
import requests
import time  

app = FastAPI()

df = pd.read_csv('../data/cleaned_dataset.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

input_dim = 768
encoder = Encoder(input_dim)

def load_encoder():
    global encoder
    state_dict = torch.load('../data/encoder.pth', map_location=torch.device('cpu'))
    if not any(k.startswith('encoder.') for k in state_dict.keys()):
        state_dict = {'encoder.' + k: v for k, v in state_dict.items()}
    encoder.load_state_dict(state_dict)
    encoder.eval()

load_encoder()

def load_embeddings():
    global dataset_embeddings
    dataset_embeddings = np.load('../data/tuned_embeddings.npy')
    dataset_embeddings = normalize(dataset_embeddings)

load_embeddings()

def get_query_embedding(query):
    with torch.no_grad():
        inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = encoder(embeddings).numpy()
        embeddings = normalize(embeddings)
    return embeddings

@app.get("/")
def main_route():
    return {"message": "Welcome! Please make your search requests at the route /query"}

@app.get("/query")
def query_route(query: str = Query(..., description="Search query")):
    start_time = time.time()  # Record the start time
    try:
        query_embedding = get_query_embedding(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating query embedding: {e}")
    
    try:
        cosine_sim = np.dot(dataset_embeddings, query_embedding.T).flatten()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing similarity: {e}")
    
    similarity_scores = list(enumerate(cosine_sim))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in similarity_scores:
        if score > 0.8:
            result = {
                "city": df.iloc[idx]['City'],
                "attractions": df.iloc[idx]['Attractions'],
                "relevance": float(score)
            }
            results.append(result)
        if len(results) == 10:  
            break

    end_time = time.time()  
    response_time = end_time - start_time 
    response_time_ms = round(response_time * 1000, 2)  

    return {
        "results": results,
        "message": f"OK - Response time: {response_time_ms} ms"
    }

@app.get("/update_model")
def update_model(
    encoder_url: str = Query(..., description="URL to download encoder.pth"),
    embeddings_url: str = Query(..., description="URL to download tuned_embeddings.npy")
):
    encoder_path = '../data/encoder.pth'
    embeddings_path = '../data/tuned_embeddings.npy'
    
    try:
        encoder_response = requests.get(encoder_url, timeout=10)
        encoder_response.raise_for_status()
        with open(encoder_path, 'wb') as f:
            f.write(encoder_response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download encoder: {e}")
    
    try:
        embeddings_response = requests.get(embeddings_url, timeout=10)
        embeddings_response.raise_for_status()
        with open(embeddings_path, 'wb') as f:
            f.write(embeddings_response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download embeddings: {e}")
    
    try:
        load_encoder()
        load_embeddings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model and embeddings: {e}")
    
    return {"message": "Model and embeddings updated successfully"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
