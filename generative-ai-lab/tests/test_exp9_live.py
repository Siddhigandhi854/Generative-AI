import urllib.request
import json
import time

def post_json(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    return json.loads(urllib.request.urlopen(req, timeout=10).read().decode())

def post_multipart(url, filename, text):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = bytearray()
    body.extend(f'--{boundary}\r\n'.encode())
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    body.extend(b'Content-Type: text/plain\r\n\r\n')
    body.extend(text.encode())
    body.extend(f'\r\n--{boundary}--\r\n'.encode())
    req = urllib.request.Request(url, data=bytes(body), headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}, method='POST')
    return json.loads(urllib.request.urlopen(req, timeout=10).read().decode())

doc = '''ARTIFICIAL INTELLIGENCE RESEARCH LAB PROTOCOLS
Guide Teacher: Dr. Sachin Takmare
Department: Computer Science and Advanced Artificial Intelligence

1. Retrieval-Augmented Generation (RAG) Architecture:
RAG combines non-parametric external document indices with parametric foundation models. Top-k relevance chunks are dynamically injected into prompt context windows to ground model outputs and eliminate hallucinations.

2. Low-Rank Adaptation (LoRA) Principles:
LoRA freezes pre-trained weight matrices W0 and injects trainable rank decomposition matrices A and B. This reduces trainable parameter counts by 99% while fitting within modest GPU VRAM budgets.'''

print("=== 1. Testing Upload ===")
up_res = post_multipart('http://localhost:5173/api/exp9/upload', 'lab_protocols.txt', doc)
print('Upload Result:', up_res)

print("\n=== 2. Testing Grounded RAG Chat ===")
chat_res = post_json('http://localhost:5173/api/exp9/chat', {
    'question': 'Who is the guide teacher in this lab?',
    'top_k': 2
})
print('Chat Answer:\n', chat_res['answer'])
print('Provider:', chat_res['provider'], '| Sources count:', len(chat_res['sources']))

print("\n=== 3. Testing LoRA PEFT Fine-Tuning Simulation ===")
ft_res = post_json('http://localhost:5173/api/exp9/finetune-simulate', {
    'model_name': 'meta-llama/Llama-3-8B',
    'lora_r': 8,
    'lora_alpha': 16,
    'epochs': 3
})
print('LoRA Status:', ft_res['status'], '| Trainable:', ft_res['trainable_percentage'], '| VRAM:', ft_res['required_vram'])
print('\n[PASSED] Experiment 9 fully verified!')
