import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run Model.")

    parser.add_argument('--validation', type=int, default=5)

    parser.add_argument('--epoch', type=int, default=650)

    parser.add_argument('--mi_num', type=int, default=962)

    parser.add_argument('--circ_num', type=int, default=2346)

    parser.add_argument('--alpha', type=int, default=0.11)

    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--nlayer', type=int, default=2)

    parser.add_argument('--n_hidden', type=int, default=20)

    parser.add_argument('--n_head', type=int, default=5)

    parser.add_argument('--nmodal', type=int, default=2)

    return parser.parse_args()