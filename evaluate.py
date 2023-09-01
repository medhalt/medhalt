import pandas as pd
from medhalt.eval.eval_full import FullDataEval
import glob,os,json
import ast,re,numpy as np

pred_prefix_dict = {
    'abs2pub':  "IR_abstract2pubmedlink",
    'pmid2title' : 'IR_pmid2title',
    'url2title' : 'IR_pubmedlink2title',
    'title2pub' : 'IR_title2pubmedlink',
    'fake'      : 'reasoning_fake',
    'FCT'       : 'reasoning_FCT',
    'Nota'      : 'reasoning_nota'
}

ds_name_dict = {v:k for k,v in pred_prefix_dict.items()}

def escaped_(data: str):
    if "'" in data:
        escaped_str = re.sub(r"(?<=\w)(')(?=\w)", r"\"", data)
    else:
        escaped_str = re.sub(r'(?<=\w)(")(?=\w)', r"\'", data)
    
#     print(escaped_str)
    return escaped_str

def parse_key_values(out_str):
    #regex = r"""['"]{key}['"]\s*:\s*['"]*(.*?)['"]*\s*[,}}]""".format(key=key)
    regex = r"""['"](.*?)['"]\s*:\s*['"]*(.*?)['"]*\s*[,}]"""
    regex = re.compile(regex)
    return regex.findall(out_str)

def recreate(out_str):
    kvs = parse_key_values(out_str)
    return {kv[0].replace("\\",""):kv[1] for kv in kvs}
    
def clean_output(id,out_str):
    try:
        if np.isnan(out_str):
            import pdb;pdb.set_trace()
        out_str = out_str.strip().split("\n")[0]
        out_str = out_str.replace("Stop Here","")
        out_str = out_str.strip()
        out_str = out_str.replace("'s","s")
        #out_str = re.sub(r":\s*'",':"""',out_str)
        #out_str = re.sub(r"'\s*}",'"""}',out_str)
        #out_str = re.sub(r"'\s*,",'""",',out_str)
        out_str = escaped_(out_str)
        return ast.literal_eval(out_str)
    except Exception as e:
        #{'cop'\s*:(.*),\s*['"]cop_index['"]:(.*),\s*['"]why_correct['"]:(.*),\s*['"]why_others_incorrect['"]:(.*)}
         b_str = out_str 
         out_str = recreate(out_str)
         if len(out_str.keys()) == 0:
             print(b_str)
         print("Exception during parsing data - recreated str",id,out_str,b_str)
         return out_str
     
def convert_to_json(prediction_folder,dataset_folder):    
    
    pred_files = glob.glob(os.path.join(prediction_folder,"*.csv")) 
    for pred_file in pred_files:
        if os.path.basename(pred_file)=='results.csv':
            continue;
        filename = os.path.basename(pred_file)
        prefix = filename.split(".")[0]
        dataset_name = pred_prefix_dict[prefix]
        dataset_df = pd.read_csv(os.path.join(dataset_folder,f"{dataset_name}.csv"))
        pred_df = pd.read_csv(pred_file,names=["id","output"])
        merge_df = pd.merge(left=dataset_df,right=pred_df,on=['id'])
        merge_df["output"] = merge_df[['id','output']].fillna("").apply(lambda params : clean_output(*params),axis='columns')
        merge_dict = merge_df.to_dict(orient='records')
        
        print(f"Merging and converting the prediction to Json files - {dataset_name}")
        with open(os.path.join(prediction_folder,f"{dataset_name}.json"),'w') as fp:
            json.dump(merge_dict,fp)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_folder",type=str)
    parser.add_argument("--dataset_folder",type=str)
    parser.add_argument("--do_json_conversion",action='store_true')
    parser.add_argument("--point_score",action='store_true')
    
    args = parser.parse_args()
    results_df = pd.DataFrame()
    
    if args.do_json_conversion:
        convert_to_json(args.prediction_folder,args.dataset_folder)
        
    for incorrect_score in [1,-0.25]:
        evaluator = FullDataEval(args.prediction_folder,1,incorrect_score)
        full_df = evaluator.run_all_evaluations()
        full_df["point_score"] = (incorrect_score==-0.25)
        results_df = pd.concat([full_df,results_df],ignore_index=True)
    
    results_df.to_csv(os.path.join(args.prediction_folder,"results.csv"),index=False)