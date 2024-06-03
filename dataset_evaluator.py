import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sentence_evaluator import ModelSentenceEvaluator

class DatasetEvaluator:
    def __init__(self, models, ground_truth, sentences):
        self.models = models
        self.ground_truth = ground_truth
        self.sentences = sentences
        self.sentence_evaluator = ModelSentenceEvaluator(models, ground_truth)
        self.inference_results = []
        self.token_overlap_results = []
        self.tokens_needed_results = []
        self.inference_time_results = []
        self.bleu_results = []
        self.k = 1000
        self.desired_percentage = 100

    def evaluate(self, k=1000, desired_percentage=100, token_step=100, max_tokens=10000):
        self.k = k
        self.desired_percentage = desired_percentage
        for sentence in tqdm.tqdm(self.sentences, desc="Evaluating sentences", mininterval=1, ncols=100, leave=False):
            self.sentence_evaluator.all_model_inference(sentence, verbose=False)
            self.inference_results.append(self.sentence_evaluator.get_inference_time())
            self.token_overlap_results.append(self.sentence_evaluator.evaluate_token_overlap(k=k))
            self.tokens_needed_results.append(self.sentence_evaluator.get_tokens_needed_for_overlap(desired_percentage=desired_percentage, k=k, max_tokens=max_tokens, token_step=token_step))
            self.bleu_results.append(self.sentence_evaluator.evaluate_bleu())

    def average_metrics(self):
        avg_inference_time = pd.DataFrame(self.inference_results).mean().to_dict()
        avg_token_overlap = pd.DataFrame(self.token_overlap_results).mean().to_dict()
        avg_tokens_needed = pd.DataFrame(self.tokens_needed_results).mean().to_dict()
        avg_bleu = pd.DataFrame(self.bleu_results).mean().to_dict()

        return {
            "average_inference_time": avg_inference_time,
            "average_token_overlap": avg_token_overlap,
            "average_tokens_needed": avg_tokens_needed,
            "average_bleu": avg_bleu
        }

    def plot_metrics(self):
        metrics = self.average_metrics()

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        self.plot_single_result(axs[0, 0], metrics["average_inference_time"], title="Average Inference Time", ylabel="Time (s)")
        self.plot_single_result(axs[0, 1], metrics["average_token_overlap"], title=f"Average Token Overlap ({self.k} Tokens)", ylabel="Overlap (%)")
        self.plot_single_result(axs[1, 0], metrics["average_tokens_needed"], title=f"Average Tokens Needed to Include {self.desired_percentage}% of {self.k} Tokens", ylabel="Tokens Needed")
        self.plot_single_result(axs[1, 1], metrics["average_bleu"], title="Average BLEU Score", ylabel="BLEU Score")

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
        ax.tick_params(axis='x', rotation=45)

