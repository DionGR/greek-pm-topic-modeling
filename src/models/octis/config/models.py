from models.octis.config.optimization import NUM_TOPICS, CHUNKSIZE

lsi_params = {"num_topics": NUM_TOPICS, 
              "chunksize": CHUNKSIZE, 
              "power_iters": 9, 
              "extra_samples": 200
              }

nmf_params = {"num_topics": NUM_TOPICS, 
              "chunksize": CHUNKSIZE, 
              "kappa": 1.0,
              "minimum_probability": 0.05431083873516954
              }

lda_params = {"num_topics": NUM_TOPICS, 
              "chunksize": CHUNKSIZE, 
              "passes": 10, 
              "alpha": 0.2454138992346559, 
              "eta": None
              }

hdp_params = {"chunksize": CHUNKSIZE, 
              "alpha": 0.16115137541071503, 
              "eta": 0.1, 
              "gamma": 1.0, 
              "tau": 32, 
              "kappa": 0.5
              }

neural_lda_params = {"num_topics": NUM_TOPICS, 
                     "batch_size": 64, 
                     "lr": 0.00403528982962455, 
                     "dropout": 0.014569283790608959,
                     "num_epochs": 100, 
                     "momentum": 0.8883848975940785,
                     "num_layers": 1, 
                     "num_neurons": 136, 
                     "activation": "softplus",
                     "solver": "adam"
                     }

prod_lda_params = {"num_topics": NUM_TOPICS, 
                   "batch_size": 64, 
                   "lr": 0.003772716556963602,
                   "dropout": 0.3387154564371324, 
                   "num_epochs": 100, 
                   "momentum": 0.6117497103810823,
                   "num_layers": 1, 
                   "num_neurons": 232,
                   "activation": "softplus",
                   "solver": "adam"
                   }