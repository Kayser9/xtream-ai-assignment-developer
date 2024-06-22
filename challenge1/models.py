from logger import CustomLogger
import pandas as pd
import uuid
from datetime import datetime
from time import time
import json




class Model():
    """Model Abstract Class"""
    
    def __init__(self, database_path : str, model_type: str) -> None:
        
        self.model_logger = CustomLogger(model_type)        
        self.model_id = self.set_model_id()
        self.df = self.download_database(database_path)
        
            
    def set_model_id(self,) -> str:
        
        self.model_logger.start_session()
        
        time_id = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        id = uuid.uuid4()
        
        model_id = f"{time_id}_{id}"
        
        self.model_logger.write_info(f"Model ID : {model_id}")
        
        return model_id
        
    def download_database(self, database_path : str) -> pd.DataFrame:
        
        self.model_logger.write_info("Downloading database ...")
        
        try:
            
            df = pd.read_csv(database_path)

        except Exception as e:
            self.model_logger.conclude_session(msg=f"Error downloading the database: {e}")
            #raise Exception(f"Error downloading the database: {e}")
        
        return df
        
    def data_pre_processing(self,):
        pass
        
    def train(self,):
        self.model_logger.write_info("Starting training ...")
        start_time = time()
    
    def predict(self,):
        pass
        
    
    def export_model(self,):
        pass
        
    
    def evaluate_performance(self,):
        pass
        
    
    def visualize_results(self,):
        pass
    
    def model_report(self,):
        pass
        

class LinearRegression(Model):
    
    def __init__(self, database_path: str, model_type : str) -> None:
        super().__init__(database_path, model_type)
        
            
    def data_pre_processing(self,):
        pass
        
    def train(self,):
        self.model_logger.write_info("Starting training ...")
        start_time = time()
    
    def predict(self,):
        pass
        
    
    def export_model(self,):
        pass
        
    
    def evaluate_performance(self,):
        pass
        
    
    def visualize_results(self,):
        pass
    
    def model_report(self,):
        pass
    
    


if __name__ == "__main__":
    
    #l = LinearRegression("..\data\diamonds.csv","LinearRegression").train()
    
    
    
    
        
        
        
    
        
        