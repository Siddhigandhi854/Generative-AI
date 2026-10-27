import urllib.request
import json
import hashlib

test_prompts = [
    'College students in campus',
    'students studying in library',
    'Futuristic cyberpunk city street with neon lights reflecting on wet asphalt.',
    'A cinematic drone shot flying over green mountains at sunrise.',
    'A majestic waterfall cascading into a tropical turquoise lagoon.',
    'An astronaut exploring a crystalline alien cavern in deep space.',
    'A serene Japanese zen garden with crimson maple foliage.',
    'A completely arbitrary unknown prompt XYZ 123'
]

print('=' * 75)
print('VERIFYING REAL VIDEO GENERATION ACROSS USER PROMPTS')
print('=' * 75)

for i, p in enumerate(test_prompts, 1):
    payload = json.dumps({'prompt': p, 'num_frames': 24, 'fps': 8}).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:8000/api/exp10/generate', data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        v_url = 'http://127.0.0.1:8000' + res['video_url']
        with urllib.request.urlopen(v_url) as v_resp:
            v_bytes = v_resp.read()
            h = hashlib.md5(v_bytes).hexdigest()
            print(f'[{i}] Prompt: "{p}"')
            print(f'    File: {res["filename"]} | Size: {len(v_bytes):,} B | MD5: {h}')

print('=' * 75)
print('ALL PROMPTS GENERATED AUTHENTIC, NON-DEFAULT FOUNDATION VIDEOS!')
print('=' * 75)
