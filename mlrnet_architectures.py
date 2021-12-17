import architectures
mlrnetfast = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":2}           
                }
mlrnetstandard = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":512,"depth":2}           
                }
mlrnetHPO = mlrnetstandard
mlrnetresblock = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,
                "hidden_params" :  {"width":512,"depth":2,"block_depth":2}           
                }
#ablation
mlrnetnopermut = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":2},
                 "n_permut":False          
                }
#regular nets
regularnetfast = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":2,"dropout":0.2,"batch_norm":True}           
                }
regularnetstandard = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":512,"depth":2,"dropout":0.2,"batch_norm":True}           
                }
regularnetresblock = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,
                "hidden_params" :  {"width":512,"depth":2,"block_depth":2,"dropout":0.2,"batch_norm":True}           
                }
#equivalent with no closeform on last layer
mlrnetfastnocf = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":2},
                 "closeform_parameter_init":False
                }
mlrnetstandardnocf = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":512,"depth":2},
                 "closeform_parameter_init":False           
                }
mlrnetresblocknocf = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,
                "hidden_params" :  {"width":512,"depth":2,"block_depth":2},
                "closeform_parameter_init":False           
                }