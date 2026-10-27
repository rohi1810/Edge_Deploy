"""Compare tensorrt with Ahmed's original encoder. Run manually."""
import sys

sys.dont_write_bytecode = True

if __name__ == '__main__':
    from _support.experiment import run
    run('tensorrt')
