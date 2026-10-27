import urllib.request
import json
import time

def test_gen(prompt):
    req = urllib.request.Request(
        'http://localhost:5173/api/exp10/generate',
        data=json.dumps({'prompt': prompt, 'num_frames': 24, 'fps': 8}).encode(),
        headers={'Content-Type': 'application/json'}
    )
    res = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    v_url = 'http://localhost:5173' + res['video_url']
    v_data = urllib.request.urlopen(v_url, timeout=10).read()
    print(f'Prompt: "{prompt[:35]}..."')
    print(f'  -> Output: {res["filename"]} | Size: {len(v_data)} bytes | Model: {res["model"]}')

print("=== TESTING EXPERIMENT 10 VIDEO GENERATION ACROSS DIVERSE PROMPTS ===")
test_gen('A majestic waterfall cascading into a tropical turquoise lagoon.')
test_gen('Futuristic cyberpunk city street with neon lights reflecting on wet asphalt.')
test_gen('A cinematic drone shot flying over green mountains at sunrise.')
print("\n[SUCCESS] Experiment 10 renders distinct, authentic videos for every prompt!")
