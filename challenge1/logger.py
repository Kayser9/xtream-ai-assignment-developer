import logging
import sys
from pathlib import Path
from random import randint


FORMAT = "%(asctime)s\t -\t%(name)s - [%(levelname)s] : %(message)s"
LEVEL = logging.INFO


class Color():
    GREY = "\033[38;20m"
    YELLOW = "\033[33;20m"
    BOLD_RED = "\033[31;1m"
    RED   = "\033[1;31m"  
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"
    
# add logger session_id to filename

class CustomLogger():
    """Custom logger object used to record the ML activities"""
    
    format = logging.Formatter(fmt=FORMAT,) #datefmt="%d-%m-%Y %H:%M:%S")
    
    def __init__(self, model_name : str) -> None:
                
        # define logger
        self.logger = logging.getLogger(name=model_name)
        
        # set minimum logger level
        self.set_logger_level()
        
        # define the logger output options
        
            # 1. file
        self.set_file_config(model_name=model_name)
            
            # 2. terminal 
        self.set_terminal_config()
        
        
    def set_logger_level(self) -> None:
        """Set the logger min level -> default : INFO"""
        self.logger.setLevel(level=LEVEL)
        
    
    def set_file_config(self, model_name : str) -> None:
        """Define the file output to have persistent records. File name depends on 'model_name' parameter used to create the Class instance"""
        # Define filename
        filename = f"Logs\{model_name}.log"
        
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
        
    def write_error(self, msg : str) -> None:
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
    
    def start_session(self,) -> None:
        """Log message used to initiate a new session: 'INFO' level message"""
        
        self.logger.info("NEW SESSION STARTING...")
        
    def conclude_session(self, success = False , msg = "") -> None:
        """Log message used to terminate a session: Check whether the sessions finished correctly depending on the success parameter.
        
        """
        
        if success:
            self.logger.info("END SESSION: SUCCESS")
        else:
            self.logger.error(f"SESSION TERMINATED WITH ERROR: {msg}")
            
        
# if __name__ == "__main__":
    
#     l = CustomLogger("REGRESSION")
    
#     for i in range(25):
#         k = randint(0,11) 
#         l.write_info(k)
#         if k % 3 == 0:
#             l.write_error(i)
#         elif k % 2 == 0:
#             l.write_warning(i)

            
    
    
    
    
    
    

    
    
    