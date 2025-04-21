import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {
            key: val[idx] 
            for key, val in self.encodings.items()
        }
        item['labels'] = self.labels[idx]
        return item
        
    def __len__(self):
        return len(self.labels)

class TextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
        self.num_labels = num_labels
        
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
             batch_size=16, epochs=3, learning_rate=2e-5, warmup_steps=0,
             weight_decay=0.01, max_grad_norm=1.0, callback=None):
        """
        Fine-tune the model on a text classification dataset.
        
        Args:
            train_texts: List of training texts
            train_labels: List of corresponding labels
            val_texts: Optional list of validation texts
            val_labels: Optional list of validation labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm (for gradient clipping)
            callback: Optional callback function to execute during training
            
        Returns:
            Dictionary containing training and validation metrics
        """
        # Prepare dataset
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation set if provided
        if val_texts is not None and val_labels is not None:
            val_dataset = TextDataset(val_texts, val_labels, self.tokenizer)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
            do_eval = True
        else:
            do_eval = False
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training loop
        global_step = 0
        train_loss = 0.0
        self.model.train()
        train_history = {'loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in epoch_iterator:
                # Get the inputs
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                
                global_step += 1
                epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if callback is not None:
                    callback(epoch, global_step, loss.item())
                
            avg_train_loss = train_loss / len(train_dataloader)
            train_history['loss'].append(avg_train_loss)
            print(f"Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss:.4f}")
            
            if do_eval:
                val_results = self.evaluate(val_dataloader)
                train_history['val_loss'].append(val_results['loss'])
                train_history['val_accuracy'].append(val_results['accuracy'])
                print(f"Validation - Loss: {val_results['loss']:.4f}, Accuracy: {val_results['accuracy']:.4f}")
        
        return train_history
            
    def evaluate(self, dataloader):
        """Evaluate the model on a validation dataset"""
        self.model.eval()
        val_loss = 0.0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true = labels.cpu().numpy()
                
                predictions.extend(preds)
                true_labels.extend(true)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        return {
            'loss': val_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
            
    def predict(self, texts, batch_size=16):
        """
        Make predictions on new texts
        
        Args:
            texts: List of texts to classify
            batch_size: Batch size for inference
            
        Returns:
            Numpy array of probabilities for each class
        """
        self.model.eval()
        all_probs = []
        
        # Create DataLoader for batch processing
        encodings = self.tokenizer(
            texts, 
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'], 
            encodings['attention_mask']
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
                
        return np.concatenate(all_probs, axis=0)
    
    def predict_classes(self, texts, batch_size=16):
        """
        Predict the most likely class for each text
        
        Args:
            texts: List of texts to classify
            batch_size: Batch size for inference
            
        Returns:
            List of predicted class indices
        """
        probs = self.predict(texts, batch_size=batch_size)
        return np.argmax(probs, axis=1)
    
    def save(self, path):
        """
        Save the model, tokenizer, and configuration
        
        Args:
            path: Directory to save the model to
        """
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save the model
        self.model.save_pretrained(path)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save the configuration
        config = {
            'num_labels': self.num_labels
        }
        
        with open(os.path.join(path, 'classifier_config.json'), 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path, device=None):
        """
        Load a saved model
        
        Args:
            path: Directory containing the saved model
            device: Device to load the model on (default: auto-detect)
            
        Returns:
            Initialized TextClassifier with loaded model
        """
        # Load the configuration
        with open(os.path.join(path, 'classifier_config.json'), 'r') as f:
            config = json.load(f)
        
        # Initialize the classifier
        classifier = cls(
            model_name=path,
            num_labels=config['num_labels'],
            device=device
        )
        
        return classifier