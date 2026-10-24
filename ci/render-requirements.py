# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jsonschema",
#     "packaging",
# ]
# ///
import argparse
import json
from pathlib import Path

from jsonschema import validate as validate_json
from packaging.specifiers import SpecifierSet

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "CI_PYTHON_VERSION": {"type": "string", "minLength": 1, "maxLength": 30},
        "CI_PANDAS_VERSION": {"type": "string", "minLength": 1, "maxLength": 30},
        "CI_NUMPY_VERSION": {"type": "string", "minLength": 1, "maxLength": 30},
        "CI_SKLEARN_VERSION": {"type": "string", "minLength": 1, "maxLength": 30},
        "CI_NO_SLOW": {"type": "boolean"},
    },
    "required": ["CI_PYTHON_VERSION", "CI_PANDAS_VERSION", "CI_NUMPY_VERSION", "CI_SKLEARN_VERSION", "CI_NO_SLOW"],
    "additionalProperties": False,
}
VersionsDict = dict[str, str | bool]


def validate_dependency_version(version: str) -> str:
    """Check that string is a valid package version specifier."""
    SpecifierSet(f"== {version}")
    return version


def validate_version_file(version_str: str) -> str:
    """Check that string refers to a dependency file in the deps/ directory."""
    if len(version_str) == 3 and version_str.isdigit() and version_str.startswith("3"):
        return f"py{version_str}.json"
    raise ValueError(f"{version_str} is an invalid dependency file")


def read_version_file(json_path: Path) -> VersionsDict:
    """Read JSON specifying versions of dependencies."""
    with json_path.open(encoding="utf-8") as fin:
        versions = json.load(fin)

    validate_json(instance=versions, schema=SCHEMA)
    for key, ver in versions.items():
        if key != "CI_NO_SLOW":
            validate_dependency_version(ver)

    return versions


def print_requirements(template_path: Path, versions: VersionsDict) -> None:
    """Render and print conda requirements YAML using specified dependency versions."""
    pkgs = (
        "NUMPY",
        "PANDAS",
        "SKLEARN",
        "PYTHON",
    )
    pinned = {}
    for env_name in pkgs:
        key = f"CI_{env_name}_VERSION"
        pinned[key] = versions[key]

    contents = template_path.read_text()

    req = contents.format(**pinned)
    print(req)


def print_pinned_versions(versions: VersionsDict) -> None:
    target_pins = {
        "CI_NUMPY_VERSION": "numpy",
        "CI_PANDAS_VERSION": "pandas",
        "CI_SKLEARN_VERSION": "scikit-learn",
    }
    for source, target in target_pins.items():
        print(f"{target} {versions[source]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["yaml", "pinned"], default="yaml")
    parser.add_argument("version", type=validate_version_file)

    args = parser.parse_args()

    deps_dir = Path(__file__).parent / "deps"
    json_path = deps_dir / args.version
    versions = read_version_file(json_path)

    if args.mode == "yaml":
        yaml_path = deps_dir / "requirements.yaml.tmpl"
        print_requirements(yaml_path, versions)
    elif args.mode == "pinned":
        print_pinned_versions(versions)


if __name__ == "__main__":
    main()
