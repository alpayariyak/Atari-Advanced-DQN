def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--epsilon_start', type=float, default=1, help='Initial Epsilon Value')
    parser.add_argument('--epsilon_end', type=float, default=0.1, help='Final Epsilon Value')
    parser.add_argument('--decay_start', type=int, default=0, help='When to start decaying epsilon')
    parser.add_argument('--decay_end', type=int, default=100000, help='When to end decaying epsilon')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')

    parser.add_argument('--buffer_size', type=int, default=300000, help='Buffer Size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--n_episodes', type=int, default=500000, help='Episodes')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma')

    parser.add_argument('--optimize_interval', type=int, default=4, help='optimize_interval')
    parser.add_argument('--target_update_interval', type=int, default=5000, help='target_update_interval')
    parser.add_argument('--evaluate_interval', type=int, default=10000, help='evaluate_interval')

    parser.add_argument('--initialize_weights', type=bool, default=False, help='Initialize manually or pytorch')

    return parser
