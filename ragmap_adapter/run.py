"""ragmap-run: OpenMask3D behind RAGMAP's object-mapping container contract (stub)."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ragmap-run")
    parser.add_argument("--input", default="/input")
    parser.add_argument("--output", default="/output")
    parser.add_argument("overrides", nargs="*")
    parser.parse_args(argv)
    raise SystemExit("stub")


if __name__ == "__main__":
    main()
