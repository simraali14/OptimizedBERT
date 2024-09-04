import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM

class ModelSentenceEvaluator:
    def __init__(self, models, ground_truth):
        self.models = models
        self.ground_truth_name, self.ground_truth_model = ground_truth
        self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.inference_results = {}
        self.sentence = None

    def model_inference(self, model, sentence):
      input_tensors = self.tokenizer(sentence, return_tensors='pt').to(self.device)
      input_tensors = {key: value.to(self.device) for key, value in input_tensors.items()}
      mask_token_index = input_tensors['input_ids'][0].tolist().index(self.tokenizer.mask_token_id)

      model.to(self.device)
      start_time = time.time()
      with torch.no_grad():
          outputs = model(**input_tensors)
      end_time = time.time()

      logits = outputs.logits[0]
      mask_logits = logits[mask_token_index]
      inference_time = end_time - start_time

      results = {
          "logits": logits,
          "mask_logits": mask_logits,
          "inference_time": inference_time }
      return results

    def all_model_inference(self, sentence, verbose=True):
        self.sentence = sentence
        self.inference_results[self.ground_truth_name] = self.model_inference(self.ground_truth_model, sentence)

        if verbose:
            iterator = tqdm(self.models.items(), desc="Inferencing models")
        else:
            iterator = self.models.items()

        for model_name, model in iterator:
            self.inference_results[model_name] = self.model_inference(model, sentence)

    def get_top_k_tokens(self, logits, k):
        top_k_tokens = torch.topk(logits, k, dim=-1).indices
        decoded_tokens = [self.tokenizer.decode(idx.item()) for idx in top_k_tokens]
        return decoded_tokens

    def get_inference_time(self):
        results = {model_name: res["inference_time"] for model_name, res in self.inference_results.items()}
        return results

    def evaluate_token_overlap(self, k):
        ground_truth_tokens = self.get_top_k_tokens(self.inference_results[self.ground_truth_name]["mask_logits"], k)
        performance_results = {self.ground_truth_name: 100.0}

        for model_name, results in self.inference_results.items():
            if model_name == self.ground_truth_name:
                continue
            model_tokens = self.get_top_k_tokens(results["mask_logits"], k)
            overlap = len(set(ground_truth_tokens) & set(model_tokens))
            performance_results[model_name] = overlap / k * 100

        return performance_results

    def get_tokens_needed_for_overlap(self, desired_percentage=100, k=1000, max_tokens=10000, token_step=1000):
        tokens_needed = {}

        for model_name, result in self.inference_results.items():
            mask_logits = result["mask_logits"]
            ground_truth_mask_logits = self.inference_results[self.ground_truth_name]["mask_logits"]

            ground_truth_sample = torch.topk(ground_truth_mask_logits, k).indices
            ground_truth_sample = [self.tokenizer.decode(idx.item()) for idx in ground_truth_sample]

            token_count = k
            achieved_overlap = False

            while token_count <= max_tokens:
                model_sample = torch.topk(mask_logits, token_count).indices
                model_sample = [self.tokenizer.decode(idx.item()) for idx in model_sample]

                overlap_count = len(set(ground_truth_sample) & set(model_sample))
                overlap_percentage = (overlap_count / k) * 100

                if overlap_percentage >= desired_percentage:
                    tokens_needed[model_name] = token_count
                    achieved_overlap = True
                    break
                else:
                    token_count += token_step

            if not achieved_overlap:
                tokens_needed[model_name] = None  # Indicate that desired percentage was not achieved

        return tokens_needed

    def evaluate_bleu(self):
        bleu_results = {}
        for model_name, result in self.inference_results.items():
            hypothesis_sentence = self.tokenizer.decode(result["logits"].argmax(dim=-1), skip_special_tokens=True)
            ground_truth_sentence = self.tokenizer.decode(self.inference_results[self.ground_truth_name]["logits"].argmax(dim=-1), skip_special_tokens=True)
            smoothing_function = SmoothingFunction().method1
            bleu_score = sentence_bleu([ground_truth_sentence.split()], hypothesis_sentence.split(), smoothing_function=smoothing_function)
            bleu_results[model_name] = bleu_score
        return bleu_results

    def plot_results(self, results, title="", ylabel=""):
        plt.figure(figsize=(10, 6))
        bars = []
        for model_name, value in results.items():
            if value is not None:
                bars.append((model_name, value))

        # Extract model names and values for plotting
        model_names = [item[0] for item in bars]
        values = [item[1] for item in bars]

        # Plot bars
        plt.bar(model_names, values)

        # Add labels to bars
        for model_name, value in bars:
            plt.text(model_name, value, round(value, 2), va='bottom', ha='center')

        # Set labels and title
        plt.xlabel('Models')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_single_result(self, ax, results, title, ylabel):
        bars = []
        for model_name, value in results.items():
            if value is not None:
                bars.append((model_name, value))

        model_names = [item[0] for item in bars]
        values = [item[1] for item in bars]

        ax.bar(model_names, values)
        for model_name, value in bars:
            ax.text(model_name, value, round(value, 2), va='bottom', ha='center')

        ax.set_xlabel('Models')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=90)

    def plot_all_metrics(self, k=1000, desired_percentage=100, token_step=100, max_tokens=10000):
        inference_time_results = self.get_inference_time()
        token_overlap_results = self.evaluate_token_overlap(k=k)
        tokens_needed_results = self.get_tokens_needed_for_overlap(desired_percentage=desired_percentage, k=k, max_tokens=max_tokens, token_step=token_step)
        bleu_results = self.evaluate_bleu()

        fig, axs = plt.subplots(2, 2, figsize=(16, 14))

        self.plot_single_result(axs[0, 0], inference_time_results, title="Inference Time", ylabel="Time (s)")
        self.plot_single_result(axs[0, 1], token_overlap_results, title=f"Token Overlap ({k} Tokens)", ylabel="Overlap (%)")
        self.plot_single_result(axs[1, 0], tokens_needed_results, title=f"Tokens Needed to Include {desired_percentage}% of {k} Tokens", ylabel="Tokens Needed")
        self.plot_single_result(axs[1, 1], bleu_results, title="BLEU Score", ylabel="BLEU Score")

        plt.tight_layout()
        plt.show()

