import argparse
from itertools import chain
import os

from packaging.requirements import Requirement
import tomli

parser = argparse.ArgumentParser()
parser.add_argument("toml_file")


def get_pinned_packages():
    pinned = set()
    pkgs = {
        "NUMPY": "numpy",
        "PANDAS": "pandas",
        "SKLEARN": "scikit-learn",
    }
    for env_name, pkg_name in pkgs.items():
        ver = os.environ.get(f"{env_name}_VERSION", None)
        if ver is not None:
            pinned.add(pkg_name)
    return pinned


def parse_requirements(filename):
    parsed_requirements = []
    pinned = get_pinned_packages()
    with open(filename, "rb") as fin:
        toml_dict = tomli.load(fin)
        proj_dict = toml_dict["project"]

    for line in chain(proj_dict["dependencies"], proj_dict["optional-dependencies"]["dev"]):
        r = Requirement(line)
        if r.name not in pinned:
            parsed_requirements.append(r)
    return parsed_requirements


def main():
    args = parser.parse_args()
    for req in parse_requirements(args.toml_file):
        print(str(req))


if __name__ == '__main__':
    main()
