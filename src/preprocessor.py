import os
import torch
import h5py
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from load_qwen import load_qwen_model  # Ensure this file correctly loads the untrained model

class LLMTIMEPreprocessor:
    """Preprocesses Lotka-Volterra time-series data for Qwen2.5-Instruct."""
    
    FILE_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","lotka_volterra_data.h5"))

    def __init__(self,file_path=None,scale_factor=None,decimal_places=3):
        self.file_path=file_path or self.FILE_PATH
        self.decimal_places=decimal_places
        self.trajectories,self.time_points=self.load_dataset()
        self.tokenizer,_,_=load_qwen_model()
        self.scale_factor=scale_factor or self.auto_scale_factor()
        print(f"Using scale factor: {self.scale_factor:.3f}")

    def load_dataset(self):
        """Loads Lotka-Volterra dataset from an HDF5 file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        with h5py.File(self.file_path,"r") as f:
            trajectories=f["trajectories"][:]
            time_points=f["time"][:]
        print(f"Dataset loaded: {trajectories.shape[0]} samples, {trajectories.shape[1]} time steps")
        return trajectories,time_points

    def auto_scale_factor(self):
        """Computes an appropriate scale factor for stability."""
        return max(np.percentile(self.trajectories[:,:,0],95),np.percentile(self.trajectories[:,:,1],95))/10

    def scale_and_format(self,values):
        """Scales and rounds values to a fixed precision."""
        return [f"{x:.{self.decimal_places}f}" for x in np.round(values/self.scale_factor,self.decimal_places)]

    def format_input(self,sample_index,num_steps=50):
        """Formats time-series data into LLMTIME structured text."""
        prey,predator=self.trajectories[sample_index,:num_steps,0],self.trajectories[sample_index,:num_steps,1]
        return ";".join([f"{p},{q}" for p,q in zip(self.scale_and_format(prey),self.scale_and_format(predator))])

    def tokenize_input(self,text):
        """Tokenizes formatted text using Qwen2.5 tokenizer."""
        return self.tokenizer(text,return_tensors="pt",truncation=True,padding=True)["input_ids"]

    def preprocess_sample(self,sample_index,num_steps=50):
        """Formats and tokenizes a time-series sample."""
        text=self.format_input(sample_index,num_steps)
        return text,self.tokenize_input(text)


if __name__=="__main__":
    preprocessor=LLMTIMEPreprocessor()
    sample_index,num_steps=0,50
    print("\n🔄 Preprocessing Sample...")
    raw_text,tokenized_seq=preprocessor.preprocess_sample(sample_index,num_steps)
    print("\n📝 Formatted Input Text:\n",raw_text)
    print("\n🔢 Tokenized Sequence:\n",tokenized_seq.tolist())
