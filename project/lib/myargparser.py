import argparse as ap


class InputArguments:
    def __init__(self):
        self.parser = ap.ArgumentParser()
        self.parser.add_argument('--ngames', type=int, help='number of games to run')
        self.parser.add_argument('--lr', type=float,
                                 help='learning rate typ 0.00025..0.05')

    def prepare(self):
        return 0