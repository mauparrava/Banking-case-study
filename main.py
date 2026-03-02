# Load the dataset
import random
import pandas as pd
import numpy as np
np.float_ = np.float64
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Import the 'load_dataset' function from the 'datasets' library for loading datasets.
from datasets import load_dataset
dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)

# Inspect the dataset
print(dataset)
print(dataset['train'][0])

# Data set overview

df = pd.DataFrame(dataset['train'])
df.head()

# -------Data cleaning--------

# Load label names
label_names = dataset['train'].features['label'].names
print(label_names)

# Map integer labels to their names
df['label_name'] = df['label'].apply(lambda x: label_names[x])
df.sample(5)

#------------------

#Cleaning the data of unimportant characters


def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"[^a-zA-Z0-9\s\?]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespaces
    return text

df['clean_text'] = df['text'].apply(clean_text)
df[['text', 'clean_text']].head()
print(df[['text', 'clean_text']].head(10).to_string(index=False)) #printing the cleaned data

# Keep only the cleaned text and the readable label
clean_df = df[['clean_text', 'label_name']].copy()

# Preview the cleaned dataset
clean_df.sample(5)

#Exploratory Data Analysis (EDA)

#Top 15 most frequent intent labels, Query Length Distribution, Top 20 Frequent Words

top_labels = clean_df['label_name'].value_counts().nlargest(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_labels.values, y=top_labels.index, palette='magma')
plt.title('Top 15 Most Frequent Intent Labels')
plt.xlabel('Count')
plt.ylabel('Intent Label')
plt.tight_layout()
plt.show()# Card payment fee charged be the most asked query
#-- Adding word count-----

# Add a new column for word count and plotting query length
clean_df['word_count'] = clean_df['clean_text'].apply(lambda x: len(x.split()))

# Plot distribution
plt.figure(figsize=(10, 5))
sns.histplot(clean_df['word_count'], bins=30, kde=True, color='steelblue')
plt.title("Distribution of Query Lengths (Words)")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

#Getting most frequent words in in cleaned  queries

from collections import Counter
import itertools

# Tokenize all text
all_words = list(itertools.chain.from_iterable([x.split() for x in clean_df['clean_text']]))

# Get word frequencies
word_freq = Counter(all_words).most_common(20)

# Convert to DataFrame for plotting
word_df = pd.DataFrame(word_freq, columns=['word', 'count'])

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=word_df, x='count', y='word', palette='cubehelix')
plt.title("Top 20 Most Frequent Words in Cleaned Queries")
plt.xlabel("Count")
plt.ylabel("Word")
plt.show()

#LLM insight generation

from huggingface_hub import hf_hub_download
model_name_or_path = "TheBloke/Llama-2-7B-chat-GGUF"
model_basename = "llama-2-7b-chat.Q5_K_M.gguf" # the model is in gguf format
model_path = hf_hub_download(
    repo_id=model_name_or_path,
    filename=model_basename
)

from llama_cpp import Llama
import time

print(f"Starting to load model: {model_basename}")
start_time = time.time()

try:
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,  # -1 = try max GPU offload
        n_batch=512,  # larger is often better if VRAM allows
        n_ctx=4096,
        n_threads=8,
        verbose=True
    )
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.1f} seconds!")

    # Quick test
    response = llm("Say hello world in Spanish.", max_tokens=30, temperature=0.0)
    print("Test generation:", response['choices'][0]['text'].strip())

except Exception as e:
    print("Failed to load model!")
    print(e)
    raise