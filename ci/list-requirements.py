import argparse
import os
from os.path import dirname, join

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


def get_pinned_packages():
    pinned = set()
    pkgs = {
        "NUMPY": "numpy",
        "PANDAS": "pandas",
        "SKLEARN": "scikit-learn",
    }
    for env_name, pkg_name in pkgs.items():
        ver = os.environ.get("{}_VERSION".format(env_name), None)
        if ver is not None:
            pinned.add(pkg_name)
    return pinned


def parse_requirements(filename):
    basedir = dirname(filename)
    parsed_requirements = []
    pinned = get_pinned_packages()
    with open(filename) as fin:
        for line in fin:
            reqs = [r for r in parse_line(line, basedir) if r.project_name not in pinned]
            parsed_requirements.extend(reqs)
    return parsed_requirements


def main():
    args = parser.parse_args()
    for req in parse_requirements(args.req_file):
        print(str(req))


if __name__ == '__main__':
    main()
