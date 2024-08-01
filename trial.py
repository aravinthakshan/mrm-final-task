"""
# Run in a seperate cell, pip installs in google colab
!pip install torch transformers
!pip install datasets
!pip install scikit-learn
!pip install transformers datasets scikit-learn

"""


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import os

def base_model_trainer(model_name, model_path, dataset_name, dependency_dataset=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    if dependency_dataset:
        dataset = load_dataset("glue", dataset_name)
    else:
        dataset = load_dataset(dataset_name)

    def benchmark_model(model_name, model_path, dataset):
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        def tokenize_function(examples):
            return tokenizer(examples["text" if "text" in examples else "sentence"], padding="max_length", truncation=True)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"./results/{model_name}_{dataset_name}",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"./logs/{model_name}_{dataset_name}",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test" if "test" in tokenized_dataset else "validation"],
        )
        
        trainer.train()
        
        # Save the model as .pth file
        save_path = f"./models/{model_name}_{dataset_name}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        predictions = trainer.predict(tokenized_dataset["test" if "test" in tokenized_dataset else "validation"])
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids
        
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        
        return accuracy, f1, model, tokenizer

    print(f"Benchmarking {model_name} on {dataset_name}")
    accuracy, f1, trained_model, tokenizer = benchmark_model(model_name, model_path, dataset)
    
    results = {
        f"{model_name}_{dataset_name}": {"accuracy": accuracy, "f1": f1}
    }

    for key, value in results.items():
        print(f"{key}: Accuracy = {value['accuracy']:.4f}, F1 = {value['f1']:.4f}")

    return trained_model, tokenizer, results

# Usage example:
# trained_model, tokenizer, results = base_model_trainer("bert", "bert-base-uncased", "sst2", "sst2")