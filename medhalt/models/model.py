import os,json,time
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
import torch
from medhalt.models.utils import PromptDataset
import asyncio
from transformers import AutoTokenizer,AutoModelForCausalLM
from text_generation import AsyncClient
import csv

class Model:
    
    def __init__(self,model_id_or_path,revision=None,load_in_8bit=False,load_in_4bit=False,rest_client=None) -> None:
        
        self.rest_client = rest_client
        self.model_path = model_id_or_path
        
        if rest_client:
            self.client = AsyncClient(rest_client)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                revision=revision,
                padding_side="left",
                truncation_side="left",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                revision=revision,
                torch_dtype=torch.float16,
                load_in_8bit=load_in_8bit,
                device_map="balanced_low_0",
                trust_remote_code=True,
            )
        
            
            if not load_in_8bit:
                self.model.half()
            
            self.model.eval()
            
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    async def rest_batch_generate(self,batch_inputs,**gen_kwargs):
        async_calls = [self.client.generate(prompt,**gen_kwargs) for prompt,_ in zip(*batch_inputs)] 
        ids = [_id for _,_id in zip(*batch_inputs)]
        results = await asyncio.gather(*async_calls)
        return results,ids
        
    def batch_generate(self,batch_input,**gen_kwargs):
        with torch.no_grad():
            for key in batch_input:
                if torch.is_tensor(batch_input[key]):
                    batch_input[key] = batch_input[key].to("cuda:0")
            generated_tokens =self.model.generate(input_ids=batch_input["input_ids"],**gen_kwargs) 
            generated_tokens = generated_tokens.cpu().numpy()
            generated_text = self.tokenizer.batch_decode(generated_tokens,
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)
        return generated_text
    
    def run_generation(self,dataset_name,prompt_template_fn,batch_size=16,output_folder=None,**gen_kwargs):
        outputs = []
        dataset = PromptDataset(dataset_name,prompt_template_fn)
        
        if self.rest_client:
            _collate_fn = dataset._restclient_collate_fn   
            _collate_fn = partial(_collate_fn)
        else:
            _collate_fn = dataset._collate_fn
            _collate_fn = partial(_collate_fn,
                              self.tokenizer)
            
        dataloader = DataLoader(dataset,batch_size,collate_fn=_collate_fn)
        pred_folder = os.path.join(output_folder,self.model_path.split("/")[1])
        os.makedirs(pred_folder,exist_ok=True)
        
        
        for batch in tqdm(dataloader):
            if self.rest_client:
                try:
                    generated_texts,ids = asyncio.run(self.rest_batch_generate(batch,**gen_kwargs))
                except Exception as e:
                    generated_texts,ids = [f"error:{str(e)}"]*len(batch[0]),["error"]*len(batch[0])
            else:
                generated_texts,ids = self.batch_generate(batch,**gen_kwargs)
            
            with open(os.path.join(pred_folder,f"{dataset_name}.csv"), 'a') as f:
                writer = csv.writer(f)
                for gtext,_id in  zip(generated_texts,ids):
                    writer.writerow([_id,gtext.generated_text])
                    
            outputs.append({"generated_text":[gtext.generated_text for gtext in generated_texts],"id":ids})
        
        with open(os.path.join(pred_folder,"gen_kwargs.json"),'w') as fp:
            json.dump(gen_kwargs,fp)
             
        return outputs

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path",type=str)
    parser.add_argument("--dataset_name",type=str)
    parser.add_argument("--greedy",action="store_false")
    parser.add_argument("--load_in_8bit",action="store_true")
    parser.add_argument("--load_in_4bit",action="store_true")
    parser.add_argument("--temperature",type=float,default=0.2)
    parser.add_argument("--max_new_tokens",type=int,default=64)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--top_p",type=float,default=0.95)
    #parser.add_argument("--top_k",type=float,default=0)
    parser.add_argument("--rest_client",type=str)
    parser.add_argument("--output_folder",type=str)

    
    
    args = parser.parse_args()
    
    model_cls = Model(model_id_or_path=args.model_path,
                      load_in_8bit=args.load_in_8bit,
                      load_in_4bit=args.load_in_4bit,
                      rest_client=args.rest_client)
    
    prompt_template_fn = lambda row: row
    
    for ds_name in ["Nota","fake", "FCT","abs2pub", "pmid2title", "url2title", "title2pub"]:
        try:
            
            print(f"Running predictions for - {ds_name}")

            generations = model_cls.run_generation(dataset_name=ds_name,
                                                    prompt_template_fn=prompt_template_fn,
                                                    batch_size=args.batch_size,
                                                    temperature=args.temperature,
                                                    do_sample= not args.greedy,
                                                    max_new_tokens=args.max_new_tokens,
                                                    top_p=args.top_p,
                                                    output_folder=args.output_folder,
                                                    stop_sequences=["Stop Here"],
                                                    seed=42) 
        except Exception as e:
            print(e)


