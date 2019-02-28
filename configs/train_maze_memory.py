class config():
    # env config
    render_train = False
    render_test = False
    env_name = "Maze"
    overwrite_render = True
    record = False
    high = 255.
    maze_size = 8
    memory_unit_size = 32

    # output config
    output_path = "results/train_nature_memory_marker/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    record_path = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    grad_clip = True
    clip_val = 10
    saving_freq = 10000
    log_freq = 50
    eval_freq = 10000
    record_freq = 250000
    soft_epsilon = 0.05

    # nature paper hyper params
    nsteps_train = 5000000
    batch_size = 64
    buffer_size = 1000000
    target_update_freq = 10000
    memory_update_freq = 10000
    gamma = 0.8
    top_bottom_loss_tradeoff = 0.2
    learning_freq = 4
    state_history = 4
    lr_begin = 0.00025
    lr_end = 0.00005
    lr_nsteps = nsteps_train / 2
    eps_begin = 0.9
    eps_end = 0.1
    eps_nsteps = 1000000
    learning_start = 50000
    use_memory = True

    # architecture config
    cnn_filters = [16, 8]
    cnn_kernel = [5, 3]
    hidden_size = 32
