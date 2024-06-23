import logging
import sys
import pathlib
from random import randint

FORMAT = "%(asctime)s\t -\t%(name)s - [%(levelname)s] : %(message)s"
LEVEL = logging.DEBUG


class Color:
    GREY = "\033[38;20m"
    YELLOW = "\033[33;20m"
    BOLD_RED = "\033[31;1m"
    RED = "\033[1;31m"
    BLUE = "\033[1;34m"
    CYAN = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD = "\033[;1m"
    REVERSE = "\033[;7m"


class CustomLogger:
    """Custom logger object used to record the ML activities"""

    format = logging.Formatter(fmt=FORMAT)

    def __init__(self, model_id: str) -> None:
        # define logger
        self.logger = logging.getLogger(name=model_id)

        # set minimum logger level
        self.set_logger_level()

        # define the logger output options

        # 1. file
        self.set_file_config(model_id=model_id)

        # 2. terminal
        self.set_terminal_config()

        # Introduce new session
        self.start_session(session_model_id=model_id)

    def set_logger_level(self) -> None:
        """Set the logger min level -> default : INFO"""
        self.logger.setLevel(level=LEVEL)

    def set_file_config(self, model_id: str) -> None:
        """Define the file output to have persistent records. File name depends on 'model_name' parameter used to create the Class instance"""
        # Create Log dir if it doesn't exist
        pathlib.Path().parent.joinpath("Logs").mkdir(exist_ok=True)

        # Define filename
        filename = pathlib.Path().parent.joinpath("Logs", f"{model_id}.log")
        # filename = Path.

        # Configure the file handler
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setFormatter(fmt=CustomLogger.format)

        # Add handler to model logger
        self.logger.addHandler(hdlr=file_handler)

    def set_terminal_config(self) -> None:
        """Define the terminal output for logs"""

        # Configure the terminale handler
        terminal_handler = logging.StreamHandler(stream=sys.stdout)
        terminal_handler.setFormatter(CustomLogger.format)

        # Add handler to model logger
        self.logger.addHandler(terminal_handler)

    def write_error(self, msg: str) -> None:
        """Write 'ERROR' level logs"""

        print(Color.RED, end="")
        self.logger.error(msg)
        print(Color.RESET, end="")

    def write_info(self, msg: str) -> None:
        """Write 'INFO' level logs"""

        self.logger.info(msg)
        print(Color.RESET, end="")

    def write_warning(self, msg: str) -> None:
        """Write 'WARNING' level logs"""

        print(Color.YELLOW, end="")
        self.logger.warning(msg)
        print(Color.RESET, end="")

    def start_session(self, session_model_id: str) -> None:
        """Log message used to initiate a new session: 'INFO' level message"""

        self.logger.info(f"NEW SESSION STARTING : {session_model_id}")

    def conclude_session(self, success=False, msg="") -> None:
        """Log message used to terminate a session: Check whether the sessions finished correctly depending on the success parameter."""
        if success == True:
            self.write_info(
                "END SESSION: SUCCESS \n---------------------------------------------------------------"
            )
        else:
            self.write_error(
                f"SESSION TERMINATED: {msg}\n---------------------------------------------------------------"
            )
