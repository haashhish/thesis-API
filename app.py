from flask import Flask, request
import subprocess

from flask import jsonify

from openai import OpenAI
import openai

import string
import pandas as pd
import time


import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
key = os.environ["API_KEY"]


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

  # Replace with your OpenAI API key
OpenAI.api_key = key # Replace with your OpenAI API key


# Define the model engine
model_engine = "gpt-3.5-turbo"

# Set up parameters
bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
batch_size = 4
num_epochs = 8
learning_rate = 2e-5

def load_imdb_data(data_file):
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    labels = df['label'].tolist()  # Assuming 'label' column contains the labels directly
    return texts, labels

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def predict_text_source(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prob_ai_generated = probabilities[0][1].item()  # Probability of being AI-generated
        _, preds = torch.max(outputs, dim=1)
    if preds.item() == 1:
        return "AI-generated", prob_ai_generated
    else:
        return "Human-written", 1 - prob_ai_generated





def identify_ai_generated_sentences(essay, model, tokenizer, device, max_length=128):
  ai_sentences = []
  for sentence in essay.split('.'):  # Split on periods, handling potential irregularities
    sentence = sentence.strip()  # Remove leading/trailing whitespace

    if not sentence:
      continue  # Skip empty sentences

    encoding = tokenizer(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      probabilities = torch.softmax(outputs, dim=1)
      prob_ai_generated = probabilities[0][1].item()  # Probability of being AI-generated
      _, preds = torch.max(outputs, dim=1)

    if preds.item() == 1:
      ai_sentences.append((sentence, prob_ai_generated))

    print(len(ai_sentences),"/",len(sentence))

  return ai_sentences



class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):  # Correct indentation here
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

data_file = "finaldataset.csv"
texts, labels = load_imdb_data(data_file)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)
model.load_state_dict(torch.load("bert_classifier.pth"))

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)



client = OpenAI(
    # This is the default and can be omitted
    api_key="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"
)


app = Flask(__name__)

def generate_essays(prom):
    ai_generated_list = []
    counter = 0
    while counter < 5:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that can you write me an essay on the following prompt about 500 words. Please do not use any label; just write it in one paragraph without headers:"},
                    {"role": "user", "content": prom}
                ],
                model="gpt-3.5-turbo",
            )
            generated_essay = chat_completion.choices[0].message.content
            print(generated_essay)
            ai_generated_list.append(generated_essay)
            counter += 1
        except Exception as e:
            print(f"Error occurred in one of the iterations: {e}")
            # If an error occurs, increment the counter to ensure we don't loop indefinitely
            counter += 1
        time.sleep(10)
    return ai_generated_list



def generate_prompt(ess):
    chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can tell me the prompt of this essay:"},
                {"role": "user", "content": ess}
            ]
            ,
            model="gpt-3.5-turbo",
        )
    prompt = chat_completion.choices[0].message.content
    return prompt


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

def get_highest_similarity(essays, new_essay):
    preprocessed_essays = [preprocess_text(essay) for essay in essays]
    preprocessed_new_essay = preprocess_text(new_essay)
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([preprocessed_new_essay] + preprocessed_essays)
    
    similarities = cosine_similarity(vectors[0], vectors[1:])
    
    highest_similarity = max(similarities[0])
    
    
    return highest_similarity


@app.route('/', methods=['GET'])
def hello_world():
    essays = []
    essay = request.args.get('essay')
    prompt = request.args.get('prompt')
    
    if len(essay) == 0:
        return 'Please provide an essay. Cannot proceed without an essay.'
    
    if prompt:
        essays = generate_essays(prompt)
    else:
        generated_prompt = generate_prompt(essay)
        essays = generate_essays(generated_prompt)
        
    
    print("The Essays Length : ",{len(essays)})
    prediction,percentage = predict_text_source(essay, model, tokenizer, device)
    ai_sentences = identify_ai_generated_sentences(essay, model, tokenizer, device)

    CosineSimilarity = get_highest_similarity(essays, essay)

    
    if CosineSimilarity > 0.5 and prediction == "AI-generated":
        response = {
        'result': f'Strongly AI generated, by percentage {percentage}',
        'ai_sentences': ai_sentences  # Directly include ai_sentences in the response

    }
        return jsonify(response)
    
    elif CosineSimilarity < 0.5 and prediction == "AI-generated":
        response = {
        'result': f'Weakly AI generated, by percentage {percentage}',
        'ai_sentences': ai_sentences  # Directly include ai_sentences in the response

    }
        return jsonify(response)
    elif CosineSimilarity > 0.7 and prediction == "Human-written":
        response = {
        'result': f'Weakly Human Written , by percentage {percentage}',
        'ai_sentences': ai_sentences  # Directly include ai_sentences in the response
    }
        return jsonify(response)
    else:
        response = {
        'result': f'Strongly Human Written , by percentage {percentage}',
        'ai_sentences': ai_sentences  # Directly include ai_sentences in the response
    }
        return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
