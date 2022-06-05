import argparse

from parso import parse

def main(file:str):
    parser.add_argument('pos_arg', type=int,
    help='A required integer positional argument')
    assert file.endswith('.jpg')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, required=True, help="Required location of input image")
    arg = parser.parse_args()

    main(arg.f)