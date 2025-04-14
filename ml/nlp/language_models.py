import os
import re
import json
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    pipeline,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class CryptoLanguageModels:
    """
    Custom language models for cryptocurrency text analysis and generation
    """
    
    def __init__(self, 
                model_name: str = "distilgpt2",
                tokenizer_name: Optional[str] = None,
                device: Optional[str] = None,
                cache_dir: Optional[str] = None):
        """
        Initialize language model
        
        Args:
            model_name (str): Name of the pre-trained model to use
            tokenizer_name (str, optional): Name of the tokenizer to use (defaults to model_name)
            device (str, optional): Device to use for computation ('cpu', 'cuda', 'cuda:0', etc.)
            cache_dir (str, optional): Directory to cache models
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else model_name
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize tokenizer and model to None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.cache_dir = cache_dir
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, 
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def load_text_generation_model(self):
        """
        Load a text generation model
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.startswith('cuda') else -1
            )
            
            logger.info(f"Loaded text generation model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading text generation model: {e}")
            raise
    
    def load_classification_model(self, num_labels: int = 2):
        """
        Load a text classification model
        
        Args:
            num_labels (int): Number of classification labels
        """
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            # Create text classification pipeline
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.startswith('cuda') else -1
            )
            
            logger.info(f"Loaded classification model: {self.model_name} with {num_labels} labels")
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            raise
    
    def generate_text(self, 
                    prompt: str, 
                    max_length: int = 100,
                    num_return_sequences: int = 1,
                    temperature: float = 0.7,
                    top_p: float = 0.9,
                    top_k: int = 50,
                    repetition_penalty: float = 1.2,
                    do_sample: bool = True) -> List[str]:
        """
        Generate text based on a prompt
        
        Args:
            prompt (str): Input prompt for text generation
            max_length (int): Maximum length of generated text
            num_return_sequences (int): Number of sequences to generate
            temperature (float): Temperature for sampling
            top_p (float): Nucleus sampling parameter
            top_k (int): Top-k sampling parameter
            repetition_penalty (float): Penalty for repetition
            do_sample (bool): Whether to use sampling
            
        Returns:
            List[str]: List of generated text sequences
        """
        if self.pipeline is None or not isinstance(self.pipeline, pipeline):
            self.load_text_generation_model()
        
        # Generate text
        outputs = self.pipeline(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        )
        
        # Extract generated text
        generated_texts = [output['generated_text'] for output in outputs]
        
        # Remove the prompt from the generated text
        result_texts = []
        for text in generated_texts:
            if text.startswith(prompt):
                result_texts.append(text[len(prompt):].strip())
            else:
                result_texts.append(text.strip())
        
        return result_texts
    
    def classify_text(self, texts: Union[str, List[str]]) -> List[Dict[str, Union[str, float]]]:
        """
        Classify text using the loaded classification model
        
        Args:
            texts (Union[str, List[str]]): Text or list of texts to classify
            
        Returns:
            List[Dict[str, Union[str, float]]]: Classification results
        """
        if self.pipeline is None or not isinstance(self.pipeline, TextClassificationPipeline):
            raise ValueError("Classification model not loaded. Call load_classification_model() first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Classify texts
        results = self.pipeline(texts)
        
        return results
    
    def fine_tune_model(self,
                       train_texts: List[str],
                       train_labels: Optional[List[Union[int, str]]] = None,
                       val_texts: Optional[List[str]] = None,
                       val_labels: Optional[List[Union[int, str]]] = None,
                       output_dir: str = "./model_output",
                       num_train_epochs: int = 3,
                       batch_size: int = 8,
                       learning_rate: float = 5e-5,
                       weight_decay: float = 0.01,
                       max_length: int = 128,
                       task_type: str = "classification"):
        """
        Fine-tune a language model on custom data
        
        Args:
            train_texts (List[str]): Training texts
            train_labels (List[Union[int, str]], optional): Training labels for classification
            val_texts (List[str], optional): Validation texts
            val_labels (List[Union[int, str]], optional): Validation labels for classification
            output_dir (str): Directory to save the model
            num_train_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay
            max_length (int): Maximum sequence length
            task_type (str): Task type ('classification' or 'generation')
        """
        # Validate input data
        if task_type == "classification" and train_labels is None:
            raise ValueError("train_labels must be provided for classification task")
        
        # Split data if validation set is not provided
        if val_texts is None:
            if task_type == "classification" and train_labels is not None:
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    train_texts, train_labels, test_size=0.1, random_state=42
                )
            else:
                train_texts, val_texts = train_test_split(
                    train_texts, test_size=0.1, random_state=42
                )
        
        # Load appropriate model and tokenizer for task
        if task_type == "classification":
            if self.model is None or not isinstance(self.model, AutoModelForSequenceClassification):
                # Determine number of unique labels
                unique_labels = len(set(train_labels))
                self.load_classification_model(num_labels=unique_labels)
                
            # Prepare dataset
            train_encodings = self.tokenizer(
                train_texts, 
                truncation=True, 
                padding=True, 
                max_length=max_length
            )
            val_encodings = self.tokenizer(
                val_texts, 
                truncation=True, 
                padding=True, 
                max_length=max_length
            )
            
            # Create datasets
            train_dataset = Dataset.from_dict({
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask'],
                'labels': train_labels
            })
            val_dataset = Dataset.from_dict({
                'input_ids': val_encodings['input_ids'],
                'attention_mask': val_encodings['attention_mask'],
                'labels': val_labels
            })
            
        elif task_type == "generation":
            if self.model is None or not isinstance(self.model, AutoModelForCausalLM):
                self.load_text_generation_model()
            
            # Prepare dataset for language modeling
            def tokenize_function(examples):
                return self.tokenizer(examples["text"])
            
            # Create datasets
            train_dataset = Dataset.from_dict({"text": train_texts})
            val_dataset = Dataset.from_dict({"text": val_texts})
            
            # Tokenize
            train_dataset = train_dataset.map(
                tokenize_function, 
                batched=True, 
                remove_columns=["text"]
            )
            val_dataset = val_dataset.map(
                tokenize_function, 
                batched=True, 
                remove_columns=["text"]
            )
            
            # Create data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=learning_rate,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator if task_type == "generation" else None
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Update model reference
        self.model_name = output_dir
        
        # Re-create the pipeline
        if task_type == "classification":
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.startswith('cuda') else -1
            )
        else:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.startswith('cuda') else -1
            )
    
    def generate_crypto_news_headlines(self, 
                                     coin_name: str,
                                     num_headlines: int = 5,
                                     sentiment: Optional[str] = None) -> List[str]:
        """
        Generate crypto news headlines for a specific coin
        
        Args:
            coin_name (str): Name of the cryptocurrency
            num_headlines (int): Number of headlines to generate
            sentiment (str, optional): Desired sentiment ('bullish', 'bearish', or None)
            
        Returns:
            List[str]: List of generated headlines
        """
        if self.pipeline is None:
            self.load_text_generation_model()
        
        # Create appropriate prompt based on sentiment
        if sentiment == 'bullish':
            prompt = f"Breaking news: {coin_name} price surges as"
        elif sentiment == 'bearish':
            prompt = f"Breaking news: {coin_name} price drops as"
        else:
            prompt = f"Latest {coin_name} news:"
        
        # Generate headlines
        headlines = []
        for _ in range(num_headlines):
            generated = self.generate_text(
                prompt=prompt,
                max_length=50,  # Headlines are usually short
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2
            )[0]
            
            # Clean up the headline
            headline = generated.split('\n')[0].strip()
            if headline:
                headlines.append(headline)
        
        return headlines
    
    def analyze_sentiment_distribution(self, 
                                     texts: List[str],
                                     labels: Optional[List[str]] = None) -> Tuple[plt.Figure, Dict[str, int]]:
        """
        Analyze sentiment distribution in a corpus of texts
        
        Args:
            texts (List[str]): List of texts to analyze
            labels (List[str], optional): Pre-defined sentiment labels
            
        Returns:
            Tuple[plt.Figure, Dict[str, int]]: Matplotlib figure and sentiment counts
        """
        if self.pipeline is None or not isinstance(self.pipeline, TextClassificationPipeline):
            # Load a sentiment analysis model if not already loaded
            self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.load_classification_model(num_labels=2)
        
        # Classify texts
        if labels is None:
            results = self.classify_text(texts)
            sentiment_labels = [result['label'] for result in results]
        else:
            sentiment_labels = labels
        
        # Count sentiments
        sentiment_counts = {}
        for label in sentiment_labels:
            if label in sentiment_counts:
                sentiment_counts[label] += 1
            else:
                sentiment_counts[label] = 1
        
        # Sort counts
        sentiment_counts = dict(sorted(sentiment_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()), ax=ax)
        
        # Add count labels on top of bars
        for i, (label, count) in enumerate(sentiment_counts.items()):
            ax.text(i, count + 0.5, str(count), ha='center')
        
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        
        plt.tight_layout()
        
        return fig, sentiment_counts
    
    def compare_generation_parameters(self,
                                    prompt: str,
                                    parameter_sets: List[Dict[str, Any]],
                                    num_samples: int = 3) -> Dict[str, List[str]]:
        """
        Compare different parameter settings for text generation
        
        Args:
            prompt (str): Input prompt
            parameter_sets (List[Dict[str, Any]]): List of parameter dictionaries
            num_samples (int): Number of samples per parameter set
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping parameter set names to generated texts
        """
        if self.pipeline is None:
            self.load_text_generation_model()
        
        results = {}
        
        for i, params in enumerate(parameter_sets):
            # Create a name for this parameter set
            param_name = params.get('name', f"Parameter Set {i+1}")
            
            # Generate samples with these parameters
            texts = []
            for _ in range(num_samples):
                generation_params = params.copy()
                if 'name' in generation_params:
                    del generation_params['name']
                
                generated = self.generate_text(prompt=prompt, **generation_params)
                texts.extend(generated)
            
            results[param_name] = texts
        
        return results
    
    def batch_generate(self,
                     prompts: List[str],
                     max_length: int = 100,
                     batch_size: int = 10,
                     **kwargs) -> List[str]:
        """
        Generate text for multiple prompts efficiently
        
        Args:
            prompts (List[str]): List of input prompts
            max_length (int): Maximum length of generated texts
            batch_size (int): Batch size for processing
            **kwargs: Additional generation parameters
            
        Returns:
            List[str]: List of generated texts
        """
        if self.pipeline is None:
            self.load_text_generation_model()
        
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating texts"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Generate for this batch
            batch_results = []
            for prompt in batch_prompts:
                generated = self.generate_text(
                    prompt=prompt,
                    max_length=max_length,
                    **kwargs
                )
                batch_results.extend(generated)
            
            results.extend(batch_results)
        
        return results
    
    def save_model(self, output_dir: str):
        """
        Save the model and tokenizer
        
        Args:
            output_dir (str): Directory to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not initialized")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'tokenizer_name': self.tokenizer_name,
            'model_type': 'classification' if isinstance(self.model, AutoModelForSequenceClassification) else 'generation'
        }
        
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """
        Load a saved model and tokenizer
        
        Args:
            model_dir (str): Directory containing the saved model
        """
        # Load configuration
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.model_name = config.get('model_name', model_dir)
            self.tokenizer_name = config.get('tokenizer_name', model_dir)
            model_type = config.get('model_type', 'generation')
        else:
            # Default to a generation model if config doesn't exist
            self.model_name = model_dir
            self.tokenizer_name = model_dir
            model_type = 'generation'
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load appropriate model type
        if model_type == 'classification':
            # Get number of labels from config
            model_config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(model_config_path):
                with open(model_config_path, 'r') as f:
                    model_config = json.load(f)
                num_labels = model_config.get('num_labels', 2)
            else:
                num_labels = 2
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=num_labels
            ).to(self.device)
            
            self.pipeline = TextClassificationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.startswith('cuda') else -1
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir
            ).to(self.device)
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.startswith('cuda') else -1
            )
        
        logger.info(f"Model loaded from {model_dir}")
    
    def generate_market_analysis(self, 
                               coin_name: str,
                               price_change: float,
                               volume_change: float,
                               time_period: str = "24h") -> str:
        """
        Generate market analysis text based on coin performance
        
        Args:
            coin_name (str): Name of the cryptocurrency
            price_change (float): Price change percentage
            volume_change (float): Volume change percentage
            time_period (str): Time period for the analysis
            
        Returns:
            str: Generated market analysis
        """
        if self.pipeline is None:
            self.load_text_generation_model()
        
        # Determine sentiment based on price change
        if price_change > 5:
            sentiment = "extremely bullish"
        elif price_change > 2:
            sentiment = "bullish"
        elif price_change < -5:
            sentiment = "extremely bearish"
        elif price_change < -2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        # Create prompt
        prompt = f"{coin_name} {time_period} Analysis: Price change: {price_change:.2f}%, Volume change: {volume_change:.2f}%. Market sentiment is {sentiment}. "
        
        # Generate analysis
        analysis = self.generate_text(
            prompt=prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )[0]
        
        return analysis
    
    def generate_trading_scenario(self, 
                                coin_name: str,
                                scenario_type: str = "bullish",
                                current_price: Optional[float] = None) -> str:
        """
        Generate hypothetical trading scenario
        
        Args:
            coin_name (str): Name of the cryptocurrency
            scenario_type (str): Type of scenario ('bullish', 'bearish', or 'sideways')
            current_price (float, optional): Current price of the coin
            
        Returns:
            str: Generated trading scenario
        """
        if self.pipeline is None:
            self.load_text_generation_model()
        
        # Build prompt based on scenario type
        price_str = f" at ${current_price}" if current_price else ""
        
        if scenario_type == "bullish":
            prompt = f"Bullish trading scenario for {coin_name}{price_str}: The price is likely to increase because"
        elif scenario_type == "bearish":
            prompt = f"Bearish trading scenario for {coin_name}{price_str}: The price might decrease because"
        else:  # sideways
            prompt = f"Sideways trading scenario for {coin_name}{price_str}: The price will likely remain stable because"
        
        # Generate scenario
        scenario = self.generate_text(
            prompt=prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )[0]
        
        return scenario 