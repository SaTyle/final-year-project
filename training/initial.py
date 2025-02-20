
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the Llama-2 model and tokenizer
model_name = "D:\llama-2-7b-chat.ggmlv3.q8_0.bin"  # You can change this if you want a different variant
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Load your dataset (assuming it's a CSV with a 'text' column)
dataset = load_dataset('csv', data_files='path/to_Your/dataset.csv')

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Apply the preprocessing function to the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Number of epochs
    per_device_train_batch_size=4,   # Batch size per device during training
    gradient_accumulation_steps=16,  # Accumulate gradients to fit a larger batch
    evaluation_strategy="steps",     # Evaluate during training
    logging_dir='./logs',            # Directory for logs
    logging_steps=200,               # Log every 200 steps
    save_steps=500,                  # Save model checkpoint every 500 steps
    save_total_limit=2,              # Keep only the last 2 checkpoints
    fp16=True,                       # Enable mixed precision
    push_to_hub=False                # Don't push to Hugging Face Hub
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # The model to be trained
    args=training_args,                  # Training arguments
    train_dataset=tokenized_dataset["train"],  # Training dataset
    eval_dataset=tokenized_dataset["test"],    # Evaluation dataset (if available)
    tokenizer=tokenizer                  # Tokenizer for padding and truncation
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_llama2')
tokenizer.save_pretrained('./fine_tuned_llama2')
