from models.octis.config.optimization import NUM_TOPICS, CHUNKSIZE

lsi_params = {"num_topics": NUM_TOPICS,
              "chunksize": CHUNKSIZE, 
              "power_iters": 4,
              "extra_samples": 300
              }

nmf_params = {"num_topics": NUM_TOPICS, 
              "chunksize": CHUNKSIZE, 
              "kappa": 0.5, 
              "minimum_probability": 0.01892522044842238,
              "normalize": True,
              }

lda_params = {"num_topics": NUM_TOPICS, 
              "chunksize": CHUNKSIZE, 
              "passes": 10, 
              "alpha": 0.23805907018308653,
              "eta": None
              }

hdp_params = {"chunksize": CHUNKSIZE, 
              "alpha": 0.06441597102460885, 
              "eta": 1.0, 
              "gamma": 0.3,
              "tau": 32, 
              "kappa": 0.5
              }

neural_lda_params = {"num_topics": NUM_TOPICS,
                     "batch_size": 128, 
                     "lr": 0.0008473675700221679, 
                     "dropout": 0.014222347892753232,
                     "num_epochs": 200, 
                     "momentum": 0.9264900935847131,
                     "num_layers": 1, 
                     "num_neurons": 461, 
                     "activation": "softplus", 
                     "solver": "adam"
                     }

prod_lda_params = {"num_topics": NUM_TOPICS, 
                   "batch_size": 128, 
                   "lr": 0.003772716556963602,
                   "dropout": 0.4387154564371324, 
                   "num_epochs": 50, 
                   "momentum": 0.6117497103810823,
                   "num_layers": 1, 
                   "num_neurons": 236,
                   "activation": "softplus",
                   "solver": "adam"
                   }