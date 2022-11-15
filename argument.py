def add_arguments(parser):
    parser.add_argument('--epsilon_start', type=float, default=1, help='Initial Epsilon Value')
    parser.add_argument('--epsilon_end', type=float, default=0.1, help='Final Epsilon Value')
    parser.add_argument('--decay_start', type=int, default=0, help='When to start decaying epsilon')
    parser.add_argument('--decay_end', type=int, default=100000, help='When to end decaying epsilon')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')

    parser.add_argument('--buffer_size', type=int, default=300000, help='Buffer Size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--n_episodes', type=int, default=500000, help='Episodes')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma')

    parser.add_argument('--optimize_interval', type=int, default=4, help='Optimization Interval')
    parser.add_argument('--target_update_interval', type=int, default=5000, help='How often to update targe network')
    parser.add_argument('--evaluate_interval', type=int, default=10000, help='How often to evaluate')

    parser.add_argument('--initialize_weights', type=bool, default=False, help='Initialize manually or let pytorch do it')
    parser.add_argument('--clip_grad', type=bool, default=True, help='To clip gradients or not')
    parser.add_argument('--load_checkpoint', type=int, default=False, help='The model to load')

    parser.add_argument('--test_n', type=int, default=False, help='The experiment n to save')
    return parser
