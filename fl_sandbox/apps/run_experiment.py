"""Module entrypoint for one configured benchmark run."""

from fl_sandbox.run.run_experiment import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
