from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input text prompt
prompt = "Artificial Intelligence is changing the world because"

# Convert prompt to tokens
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs,
    max_length=80,
    num_return_sequences=1,
    no_repeat_ngram_size=2
)

# Decode and print result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:\n")
print(generated_text)
