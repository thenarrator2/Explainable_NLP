
#!pip install datasets transformers

from datasets import load_dataset

dataset=load_dataset('imdb')
data= dataset['train'].shuffle(seed=42).select(range(1000))

train_text=data['text']
train_labels=data['label']

data

from transformers import BertTokenizer, BertForSequenceClassification

import matplotlib.pyplot as plt



#!pip install torch captum
from captum.attr import IntegratedGradients

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
def encode(text):
   return tokenizer(text, padding=True, truncation=True, return_tensors='pt')
train_tokens=encode(train_text)
import torch
label_tensor=torch.tensor(train_labels)

model=BertForSequenceClassification.from_pretrained('bert-base-uncased')

from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

torch.cuda.empty_cache()

train_dataset=TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], label_tensor)
batch_data=DataLoader(train_dataset, batch_size=3, shuffle=True)

optimizer=AdamW(model.parameters(), lr=1e-5)

model.train()
for epoch in range(2):
    for batches in batch_data:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batches
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

def visualize_attention(input_text):
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

    attention = attentions[-1].detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention = attention.squeeze(0)
    plt.figure(figsize=(12, 6))
    plt.imshow(attention, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.show()

def interpret_model(input_text):
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(inputs=input_ids, additional_forward_args=(attention_mask,), return_convergence_delta=True)

    attributions = attributions.sum(dim=2).squeeze(0).detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    plt.figure(figsize=(12, 6))
    plt.bar(tokens, attributions, color='skyblue')
    plt.xticks(rotation=90)
    plt.show()

def predict_and_explain(input_text):
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=-1).item()
    label = 'positive' if prediction == 1 else 'negative'

    visualize_attention(input_text)
    interpret_model(input_text)

    return label

result = predict_and_explain("The movie was far below our expectations.")
print(f'Prediction: {result}')
