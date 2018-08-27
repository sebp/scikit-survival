import argparse
from os.path import dirname, join
import setuptools
from pkg_resources import parse_requirements as _parse_requirements

parser = argparse.ArgumentParser()
parser.add_argument('req_file')


def parse_line(line, basedir):
    sline = line.strip()
    if sline.startswith("-r"):
        _, nested_req = sline.split(" ", 1)
        parsed_req = parse_requirements(join(basedir, nested_req))
    else:
        parsed_req = list(_parse_requirements(line))
    return parsed_req


def parse_requirements(filename):
    basedir = dirname(filename)
    parsed_requirements = []
    with open(filename) as fin:
        for line in fin:
            parsed_requirements.extend(parse_line(line, basedir))
    return parsed_requirements


def main():
    args = parser.parse_args()
    for req in parse_requirements(args.req_file):
        print(str(req))


if __name__ == '__main__':
    main()

