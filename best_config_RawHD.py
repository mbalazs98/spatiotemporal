class Config:
    
    ################################################
    #            General configuration             #
    ################################################

    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    dataset = 'rawhd'                    
    datasets_path = '../SNN-delays/Datasets/RawHD'

    seed = 0

    # model type could be set to : 'snn_delays' |  'snn_delays_lr0' |  'snn'
    model_type = 'snn_delays'          
    

    epochs = 150
    batch_size = 256

    ################################################
    #               Model Achitecture              #
    ################################################
    init_tau = 10.05       


    n_inputs = 40
    n_hidden_layers = 1
    n_hidden_neurons = 256 
    n_outputs = 20
    used_sparsities = [0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875, 0.99609375, 0.998046875]
    sparsity_p = used_sparsities[1]
    l1_lambda = 0.01#1e-5
    dynamic = True
    rigl = True
    dalean = True

    dropout_p = 0.4
    use_batchnorm = True
    detach_reset = True

    loss = 'sum'
    loss_fn = 'CEloss'

    v_threshold = 1.0
    alpha = 5.0

    init_w_method = 'kaiming_uniform'

    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adam'
    optimizer_pos = 'adam'

    weight_decay = 0#1e-5

    lr_w = 1e-3
    lr_pos = 100*lr_w   if model_type =='snn_delays' else 0
    
    # 'one_cycle', 'cosine_a', 'none'
    scheduler_w = 'one_cycle'    
    scheduler_pos = 'cosine_a'   if model_type =='snn_delays' else 'none'


    # for one cycle
    max_lr_w = 5 * lr_w
    max_lr_pos = 5 * lr_pos


    # for cosine annealing
    t_max_w = epochs
    t_max_pos = epochs

    ################################################
    #                    Delays                    #
    ################################################
    DCLSversion = 'gauss' if model_type =='snn_delays' else 'max'
    decrease_sig_method = 'exp'
    kernel_count = 1

    max_delay = 25
    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number
    
    # For constant sigma without the decreasing policy, set model_type == 'snn_delays' and sigInit = 0.23 and final_epoch = 0
    sigInit = max_delay // 2        if model_type == 'snn_delays' else 0
    final_epoch = (1*epochs)//4     if model_type == 'snn_delays' else 0


    left_padding = max_delay-1
    right_padding = 0

    init_pos_method = 'uniform'
    init_pos_a = -max_delay//2
    init_pos_b = max_delay//2


    #############################################
    #                      Save                 #
    #############################################

    run_name = 'Run Name'


    run_info = f'||{model_type}||{dataset}||dynamic={dynamic}'

    run_name = run_name + f'||seed={seed}' + run_info
    
    # REPL is going to be replaced with best_acc or best_loss for best model according to validation accuracy or loss
    save_model_path = f'{run_name}_REPL.pt'
