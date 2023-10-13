from typing import Optional,Callable
from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd
import os,sys
from medhalt.prompts.utils import get_samples

class PromptDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        prompt_template_fn: Optional[Callable[[],str]],
    ):
        super().__init__() 
        self.dataset = get_samples(dataset_name=dataset_name,shots=2,prompt_version='v0') 
        self.prompt_template_fn = prompt_template_fn

    @staticmethod
    def _collate_fn(tokenizer,batch):
        prompts = [batch_item["prompt"] for batch_item in batch]
        model_inputs = tokenizer.batch_encode_plus(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        #model_inputs = {keydel_inputs[key].to(device) for key in model_inputs}
        model_inputs["prompts"] = prompts
        return model_inputs
    
    @staticmethod
    def _restclient_collate_fn(batch):
        prompts = [batch_item["prompt"] for batch_item in batch]
        ids = [batch_item["id"] for batch_item in batch]
       
        #model_inputs = {key: model_inputs[key].to(device) for key in model_inputs}
        #model_inputs["prompts"] = prompts
        return prompts,ids
    
    def __getitem__(self, index):
        return self.prompt_template_fn(self.dataset[index])

    def __len__(self):
        return len(self.dataset)