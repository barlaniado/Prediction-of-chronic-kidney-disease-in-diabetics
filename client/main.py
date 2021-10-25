import argparse
from client import ObservationToPredict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--path_json_file', nargs=1, type=str,
                        help='Please insert the path to the json file that contains x')

    args = parser.parse_args()
    if not args.path_json_file:
        raise AssertionError('Expect to get a path to the json file that contains x')
    else:
        try:
            return ObservationToPredict(args.path_json_file[0])
        except FileNotFoundError:
            print('Please check the file path again')
            return None


if __name__ == "__main__":
    print(main())