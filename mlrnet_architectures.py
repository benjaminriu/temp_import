import architectures
mlrnetfastv3 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":2}           
                }
mlrnetHPO = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":256,"depth":2}           
                }
mlrnetslowv1 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":2048,"depth":2}           
                }
mlrnetslowv2 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":1024,"depth":2}           
                }
mlrnetshallowv1 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":4096,"depth":1}           
                }
mlrnetdeepv1 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-4, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":1024,"depth":4}           
                }