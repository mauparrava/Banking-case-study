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

#---LLM insights---
def generate_with_llm(prompt):
    response = llm(
        prompt,
        max_tokens=300,
        temperature=0.7
    )
    out = response["choices"][0]["text"].strip()
    return out if out else "No output generated."

# Interacting with LLM

def chunk_text(text, max_words=400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def generate_summary_llm(text, label_name):
    # Prepare prompt
    unified_prompt = f"""You are an expert summarizer.
Summarize the customer queries and issues for the label: **{label_name}** into 3-5 crystal-clear bullet points.
- Do not repeat points.
- Make sure each bullet is a complete sentence.
- Keep the language clear and specific.
- End each bullet with proper punctuation.
- Focus only on key themes for this label.
- Be concise and clear.
TEXT:
"""

    # The generate_summary_llm function uses the LLM to generate a concise, bullet-point summary of customer queries based on a specific label.
    #
    # Function Purpose:
    #
    # To extract 3–5 key insights from long text data (e.g., customer complaints) related to a particular label, using clear and actionable bullet points.
    #Its ideal for generating compact summaries from large datasets of customer feedback, categorized by themes or labels like "Account Access," "Billing Issues," etc.
    chunks = list(chunk_text(text, max_words=400))
    summaries = []

    for idx, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {idx}/{len(chunks)} for label '{label_name}'...")
        summaries.append(generate_with_llm(unified_prompt + chunk))

    # Final summary from partial summaries
    combined_text = " ".join(summaries)
    return generate_with_llm(unified_prompt + combined_text) if len(chunks) > 1 else summaries[0]

import random

label_summaries = {}
unique_labels = clean_df['label_name'].unique()
selected_labels = random.sample(list(unique_labels), 4)  # pick 4 random labels

for label_name in selected_labels:
    subset_df = clean_df[clean_df['label_name'] == label_name]
    sampled_texts = subset_df['clean_text'].sample(min(30, len(subset_df)), random_state=42)
    combined_text = " ".join(sampled_texts)

    if combined_text.strip():
        label_summaries[label_name] = generate_summary_llm(combined_text, label_name)