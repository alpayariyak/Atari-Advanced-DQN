def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--epsilon_start', type=float, default=1, help='Initial Epsilon Value')
    parser.add_argument('--epsilon_end', type=float, default=0.025, help='Final Epsilon Value')
    parser.add_argument('--buffer_size', type=int, default=1000, help='Buffer Size')
    parser.add_argument('--decay_start', type=int, default=False, help='When to start decaying epsilon')
    parser.add_argument('--decay_end', type=int, default=False, help='When to start decaying epsilon')
    parser.add_argument('--n_episodes', type=int, default=50_000, help='Episodes')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma')
    
    parser.add_argument('--optimize_interval', type=int, default=4, help='optimize_interval')
    parser.add_argument('--target_update_interval', type=int, default=False, help='target_update_interval')
    parser.add_argument('--evaluate_interval', type=int, default=False, help='evaluate_interval')
    parser.add_argument('--grad_clip', type=int, default=1, help='grad_clip')

    parser.add_argument('--checkpoint_name', type=int, default=False, help='checkpoint_name')
    return parser
