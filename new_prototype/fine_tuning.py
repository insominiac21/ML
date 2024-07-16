from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch
import json

# Define paths and parameters
model_path = 'C:\\Users\\Raghu\\MedAssist\\model\\llama-2-7b-chat.ggmlv3.q8_0.bin'
fine_tune_data_path = 'C:\\Users\\Raghu\\MedAssist\\output\\output.json'
output_model_path = 'C:\\Users\\Raghu\\MedAssist\\fine_tuned_model'

# Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load and preprocess dataset (example dataset class)
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input': self.data[idx]['input']}  # Adjust based on your dataset format

# Tokenize dataset (if necessary)
tokenizer = AutoTokenizer.from_pretrained(model_path)  # Adjust based on your tokenizer

# Fine-tuning parameters
training_args = TrainingArguments(
    output_dir=output_model_path,
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=MyDataset(fine_tune_data_path),  # Replace with your dataset instance
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(output_model_path)
