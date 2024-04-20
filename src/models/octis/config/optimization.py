from skopt.space.space import Real, Categorical, Integer

OPTIMIZATION_RESULT_PATH = "data/hyperparameter_opt/"

TOP_K = 7
NUM_TOPICS = 30
CHUNKSIZE = 4000

NUM_PROCESSES = 6
MODEL_RUNS = 5

search_space = {
    "lsi": {
        "num_topics": Categorical({NUM_TOPICS}),
        "chunksize": Categorical({CHUNKSIZE}),
        "power_iters": Integer(1, 10),
        "extra_samples": Categorical({100, 200, 300}),
    },
    "lda": {
        "num_topics": Categorical({NUM_TOPICS}),
        "chunksize": Categorical({CHUNKSIZE}),
        "passes": Categorical({1, 5, 10}),
        "alpha": Real(0.0, 1.0),
        "eta": Categorical({None, 0.1, 0.5, 1.0})
    },
    "hdp": {
        "chunksize": Categorical({CHUNKSIZE}),
        "alpha": Real(0.0, 1.0),
        "eta": Categorical({0, 0.1, 0.5, 1.0}),
        "gamma": Categorical({0.1, 0.3, 0.5, 0.7, 1.0}),
        "tau": Categorical({32, 64, 128}),
        "kappa": Categorical({0.5, 1.0})
    },
    "nmf": {
        "num_topics": Categorical({NUM_TOPICS}),
        "chunksize": Categorical({CHUNKSIZE}),
        "kappa": Categorical({0.5, 1.0}),
        "minimum_probability": Real(0.0, 0.2),
    },
    "neural_lda": {
        "num_topics": Categorical({NUM_TOPICS}),
        "batch_size": Categorical({64}),
        "lr": Real(0.0001, 0.005),
        "dropout": Real(0.0, 0.5),
        "num_epochs": Categorical({50, 100, 200}),
        "momentum": Real(0.5, 0.99),
        "num_layers": Integer(1, 5),
        "num_neurons": Integer(50, 500),
        "activation": Categorical({"relu", "softplus"}),
        "solver": Categorical({"adam", "sgd"})
    },
    "prod_lda": {
        "num_topics": Categorical({NUM_TOPICS}),
        "batch_size": Categorical({64}),
        "lr": Real(0.0001, 0.005),
        "dropout": Real(0.0, 0.5),
        "num_epochs": Categorical({50, 100, 200}),
        "momentum": Real(0.5, 0.99),
        "num_layers": Integer(1, 5),
        "num_neurons": Integer(50, 500),
        "activation": Categorical({"relu", "softplus"}),
        "solver": Categorical({"adam", "sgd"})
    }
}