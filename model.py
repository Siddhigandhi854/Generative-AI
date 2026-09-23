from transformers import pipeline

# Load the Hugging Face text-generation model
generator = pipeline(
    "text-generation",
    model="openai-community/gpt2"
)

def generate_text(prompt, max_new_tokens=100):
    result = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        num_return_sequences=1
    )

    return result[0]["generated_text"]