import random
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings


warnings.filterwarnings("ignore", category=UserWarning)




# Load the pre-trained model and tokenizer
model_name = 'gpt2'  # Using GPT-2 for simplicity
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


# Function to generate a review
def generate_review(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Create attention mask
    attention_mask = torch.ones(inputs.shape, device=device)
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id
        attention_mask=attention_mask  # Pass the attention mask
    )
    generated_review = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_review

dataset_path = 'synthetic_amazon_reviews.csv'  # Replace with your dataset path
df = pd.read_csv(dataset_path)

# Example prompts for generating synthetic reviews
prompts = [
    "Write a positive review about a B Complex vitamin that improves absorption.",
    "Write a positive review about a vitamin that has all-natural sources.",
    "Write a positive review about a liver support supplement that has good results.",
    "Write a positive review about a vitamin that helped someone after chemotherapy.",
    "Write a positive review about a vitamin product with fast delivery and great quality."
]

# Generate synthetic reviews
synthetic_reviews = []
for prompt in prompts:
    review = generate_review(prompt)
    synthetic_reviews.append(review)

# Create a DataFrame and save to CSV
df_synthetic = pd.DataFrame(synthetic_reviews, columns=['text'])
df_synthetic['title'] = ['Synthetic Review'] * len(synthetic_reviews)  # Placeholder title
df_synthetic['rating'] = [5] * len(synthetic_reviews)  # Assuming synthetic reviews are positive

# Rearrange columns to match the original dataset
df_synthetic = df_synthetic[['rating', 'title', 'text']]

# Combine original and synthetic datasets
combined_df = pd.concat([df, df_synthetic], ignore_index=True)

# Save combined dataset to CSV
combined_df.to_csv('combined_amazon_reviews.csv', index=False)

print("Combined dataset saved to 'combined_amazon_reviews.csv'")

