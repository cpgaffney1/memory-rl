class Config():
    # env config
    render_train = False
    render_test = False
    env_name = "Maze"
    overwrite_render = True
    record = False
    high = 255.
    maze_size = 10


    # output config
    def set_paths(self, output_path):
        self.output_path = output_path
        self.model_output = output_path + "model.weights/"
        self.log_path = output_path + "log.txt"
        self.plot_output = output_path + "scores.png"
        self.record_path = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    grad_clip = True
    clip_val = 10
    saving_freq = 250000
    log_freq = 50
    eval_freq = 10000
    record_freq = 250000
    soft_epsilon = 0.05
    extended_eval_threshold = 0.5

    # nature paper hyper params
    nsteps_train = 5000000
    batch_size = 32
    buffer_size = 1000000
    target_update_freq = 10000
    memory_update_freq = 10000
    gamma = 0.99
    learning_freq = 1
    state_history = 1
    lr_begin = 0.00025
    lr_end = 0.00005
    lr_nsteps = nsteps_train / 2
    top_bottom_loss_tradeoff = 0.5
    eps_begin = 1
    eps_end = 0.1
    eps_nsteps = 1000000
    learning_start = 50000
    use_memory = True
    use_rnn = True
    hard = True

    assert(use_rnn and use_memory)

    # architecture config
    rnn_layers = [32, 32]
    cnn_filters = [16, 8]
    cnn_kernel = [5, 3]
    hidden_size = 32
    memory_unit_size = 2 * hidden_size
