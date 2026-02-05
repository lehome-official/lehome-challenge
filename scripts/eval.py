import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

from .utils import common
from .utils.parser import setup_eval_parser
from .utils.common import launch_app_from_args
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for evaluation script."""
    parser = setup_eval_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = launch_app_from_args(args)
    try:
        import lehome.tasks.bedroom
        from .utils.evaluation import eval

        if getattr(args, "headless", False):
            import os

            os.environ["LEHOME_DISABLE_KEYBOARD"] = "1"
        eval(args, simulation_app)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        common.close_app(simulation_app)


if __name__ == "__main__":
    main()
