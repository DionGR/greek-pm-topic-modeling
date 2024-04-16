from typing import List
from octis.optimization.optimizer import Optimizer
from octis.models.model import AbstractModel
from octis.evaluation_metrics.metrics import AbstractMetric
from octis.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import time
import json


class OCTISModelOptimizer:
    
    def __init__(self, model: AbstractModel, dataset: Dataset, 
                 search_space, 
                 validation_metric: AbstractMetric, other_metrics: List[AbstractMetric], 
                 topk: int = 7,
                 optimization_runs: int = 20, model_runs: int = 5, save_path: str = 'data/hyperparameter_opt'):
        
        self.model = model
        self.dataset = dataset
        self.search_space = search_space
        self.validation_metric = validation_metric
        self.other_metrics = other_metrics
        self.optimization_runs = optimization_runs
        self.model_runs = model_runs
        self.save_path = save_path
        self.topk = topk
        
        self.optimization_result = None
        
    def optimize(self):
        optimizer = Optimizer()
        
        start = time.time()
        
        self.optimization_result = optimizer.optimize(
            self.model, self.dataset, 
            self.validation_metric, 
            self.search_space, 
            topk=self.topk,
            number_of_call=self.optimization_runs, 
            model_runs=self.model_runs, 
            save_models=False, 
            extra_metrics=self.other_metrics,
            save_path=self.save_path)
        
        end = time.time()
        duration = end - start
        print('Optimizing model took: ' + str(round(duration)) + ' seconds.')
    
        self.optimization_result.save(f"{self.save_path}/opt_results.json")
        
        self.plot_optimization()
        
        return self.get_best_parameters()
    
    def plot_optimization(self):
        results = json.load(open(f"{self.save_path}/opt_results.json", "r"))
        
        plt.axhline(y=max(results["f_val"]), color='r', linestyle='--')
        plt.axvline(x=results["f_val"].index(max(results["f_val"])), color='r', linestyle='--')    
        
        plt.xlabel("Iteration")
        plt.ylabel("Validation Metric Value")
        plt.title("Median Validation Metric Value")
        
        plt.plot(results["f_val"])
        plt.savefig(f"{self.save_path}/opt_plot.png")
        
        plt.show()
        
    def get_best_parameters(self):
        if self.optimization_result is None:
            raise Exception("Model has not been optimized yet.")
        
        results = json.load(open(f"{self.save_path}/opt_results.json", "r"))
        max_idx = results["f_val"].index(max(results["f_val"]))
        
        best_parameters = {parameter: results["x_iters"][parameter][max_idx] for _, parameter in enumerate(self.search_space.keys())}
        
        with open(f"{self.save_path}/best_parameters.json", "w") as f:
            json.dump(best_parameters, f)
        
        return best_parameters
        
        
