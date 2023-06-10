import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("yaml_file")


def get_pinned_packages():
    pkgs = {
        "NUMPY",
        "PANDAS",
        "SKLEARN",
        "PYTHON",
    }
    pinned = {}
    for env_name in pkgs:
        key = f"CI_{env_name}_VERSION"
        ver = os.environ.get(key, "*")
        pinned[key] = ver
    return pinned


def render_requirements(filename):
    pinned = get_pinned_packages()
    with open(filename) as fin:
        contents = "".join(fin.readlines())

    return contents.format(**pinned)


def main():
    args = parser.parse_args()
    req = render_requirements(args.yaml_file)
    print(req)


if __name__ == "__main__":
    main()
