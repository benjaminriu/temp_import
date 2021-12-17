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
                "hidden_params" :  {"width":512,"depth":2}           
                }
mlrnetstandardv1 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":512,"depth":2}           
                }
mlrnetresblockv1 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-2, "total_steps" : 200},
                "max_iter":200,
                "learning_rate":1e-3, #useless with OneCycleLR
                "hidden_nn" : architectures.ResidualLayers,
                "hidden_params" :  {"width":512,"depth":2,"block_depth":2}           
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
mlrnetw2048 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-4, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":2048,"depth":2}           
                }
mlrnetw4096 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-4, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":4096,"depth":2}           
                }
mlrnetw8192 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-4, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":8192,"depth":2}           
                }
mlrnetw16384 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-4, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":16384,"depth":2}           
                }
mlrnetw32768 = {"lr_scheduler" : "OneCycleLR",
                "lr_scheduler_params" : {"max_lr":1e-3, "total_steps" : 500},
                "max_iter":500,
                "learning_rate":1e-4, #useless with OneCycleLR
                "hidden_nn" : architectures.DenseLayers,
                "hidden_params" :  {"width":32768,"depth":2}           
                }