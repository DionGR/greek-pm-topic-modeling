from models.octis.config.optimization import NUM_TOPICS, CHUNKSIZE

lsi_params = {"num_topics": NUM_TOPICS, 
              "chunksize": CHUNKSIZE,
              "power_iters": 6,
              "extra_samples": 100
              }

nmf_params = {"num_topics": NUM_TOPICS, 
              "chunksize": CHUNKSIZE, 
              "kappa": 1.0,
              "minimum_probability": 0.1902737479106116
              }

lda_params = {"num_topics": NUM_TOPICS,
              "chunksize": CHUNKSIZE, 
              "passes": 10, 
              "alpha": 0.11795881168296588, 
              "eta": None
              }

hdp_params = {"chunksize": CHUNKSIZE, 
              "alpha": 0.13419411242193027, 
              "eta": 0.5,
              "gamma": 0.5,
              "tau": 32, 
              "kappa": 0.5
              }

neural_lda_params = {"num_topics": NUM_TOPICS, 
                     "batch_size": 64, 
                     "lr": 0.004868879360882034,
                     "dropout": 0.007311675749379744,
                     "num_epochs": 200,
                     "momentum": 0.8929994385857404,
                     "num_layers": 1, 
                     "num_neurons": 285,
                     "activation": "softplus",
                     "solver": "adam",
                     "use_partitions": False
                     }

prod_lda_params = {"num_topics": NUM_TOPICS,
                   "batch_size": 64,
                   "lr": 0.003772716556963602, 
                   "dropout": 0.04387154564371324, 
                   "num_epochs": 175, 
                   "momentum": 0.6117497103810823,
                   "num_layers": 1, 
                   "num_neurons": 236, 
                   "activation": "softplus",
                   "solver": "adam",
                    "use_partitions": False
                   }