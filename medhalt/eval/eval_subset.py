import pandas as pd
import os
import glob
from tqdm import tqdm
import json

class FullDataEvalSubset(object):
    
    def __init__(self, folder_name, correct_score=1, incorrect_score=-0.25):
        
        self.correct_score   = correct_score
        self.incorrect_score = incorrect_score
        
        self.all_sub_folders = [f"{folder_name}/{m}" for m in os.listdir(f"{folder_name}") if 'mcq' in m]
        
    def read_json(self, file):
        try:
            with open(file, 'rb') as json_file:
                file_data = json.loads(json_file.read().decode('utf-8'))
            return file_data
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
    
    

    def evaluate_answer(self, predicted, correct):
        return str(predicted.lower()) == str(correct.lower())
    
    def calculate_scores(self, df):
        df['score'] = (df['correct'] * self.correct_score + df['wrong'] * self.incorrect_score) / 100
        return df

    def create_dataframe(self, task_name, correct, wrong):
        total = correct + wrong
        df_dict = {'task_name': [task_name], 'total': [total], 'correct': [correct], 'wrong': [wrong]}
        return pd.DataFrame(df_dict)

    def handle_exceptions(self, task_name, sample, exception):
        print(task_name, sample['id'], exception)
        return 1

    def reasoning_functional_eval(self, sample, task_name):

        correct, wrong, exception_count = 0, 0, 0
        possible_keys = ['correct_answer', 'answer', 'correct answer', 'corrent_answer', 'Correct Answer', 
                         'Answer', 'Correct_answer', "Correct answer"]

        try:
            predicted_answer = None
            for key in possible_keys:
                if key in eval(str(sample['gpt_output'])):
                    predicted_answer = eval(str(sample['gpt_output']))[key]
                    break

            if predicted_answer is None:
                raise KeyError("No valid key found in 'gpt_output'")

            if self.evaluate_answer(str(predicted_answer), sample['testbed_data']['correct_answer']):
                correct += 1
            else:
                wrong += 1

        except Exception as e:
            exception_count += self.handle_exceptions(task_name, sample, e)

        result = {
            "correct": correct,
            "wrong": wrong,
            "exception_count": exception_count
        }

        return self.create_dataframe(task_name, correct, wrong)
    
    
    def IR_pmid2title_pubmedlink2title_eval(self, sample, task_name):
        
        correct = wrong = exception_count = 0
        
        
        try:
            predicted_title = eval(str(sample['gpt_output']))['paper_title']
            
            if self.evaluate_answer(predicted_title, sample['testbed_data']['Title']):
                correct += 1
            else:
                wrong += 1
        except Exception as e:
            exception_count += self.handle_exceptions(task_name, sample, e)
        return self.create_dataframe(task_name, correct, wrong)
    
    
    def IR_title2pubmedlink_abstract2pubmedlink_eval(self, sample, task_name):
        correct = wrong = exception_count = 0
        try:
            predicted_url = eval(str(sample['gpt_output']))['url']

            if self.evaluate_answer(predicted_url, sample['testbed_data']['url']):
                correct += 1
            else:
                wrong += 1
        except Exception as e:
            exception_count += self.handle_exceptions(task_name, sample, e)
        return self.create_dataframe(task_name, correct, wrong)

    def reasoning_nota_eval(self, sample, task_name):
        correct = wrong = exception_count = 0
        try:
            predicted_answer = eval(str(sample['gpt_output']))['cop']

            if self.evaluate_answer(predicted_answer, sample['testbed_data']['correct_answer']):
                correct += 1
            else:
                wrong += 1
        except Exception as e:
            exception_count += self.handle_exceptions(task_name, sample, e)
        return self.create_dataframe(task_name, correct, wrong)

    def reasoning_fake_eval(self, sample, task_name):
        correct = wrong = exception_count = 0
        try:
            predicted_answer = str(eval(str(sample['gpt_output']))['cop']).lower()

            if any(term in predicted_answer for term in ['i do not know', 'conceding defeat', 'admit', 'none of the above',
                                                          'acknowled', 'irrelevant', 'fiction', 'all of the above', 
                                                          'nonsensical', 'no correct', 'absurd', 'defy', 'i don"t know.', 
                                                          'defies']):
                correct += 1
            else:
                wrong += 1
        except Exception as e:
            exception_count += self.handle_exceptions(task_name, sample, e)
    
        return self.create_dataframe(task_name, correct, wrong)
    
    
    def correct_df(self, row):
        if 'vinci' in row['task_name']:
            row['task_name'] = row['task_name'].split('vinci_')[1]
            row['model_name'] = 'Davinci'
        elif 'gpt3' in row['task_name']:
            row['task_name'] = row['task_name'].split('gpt3_')[1]
            row['model_name'] = 'gpt-3.5-turbo'
        return row

    def finalise_dataframe(self, df):
        
        df['accuracy'] = (df['correct'] / df['total'] * 100).round(3)
        df['precision'] = df['correct'] / (df['correct'] + df['wrong'])
        df['recall'] = df['correct'] / df['total']
        df['f1_score'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
        df = df.apply(self.correct_df, axis=1)
        df = self.calculate_scores(df)
        df = self.sum_and_avg(df)
        return df
    
    
    def sum_and_avg(self, df):
        # calculate sum for 'total', 'correct', 'wrong' and avg for 'accuracy', 'precision', 'recall', 'f1_score'
        df_sum = df.sum(numeric_only=True)
        df_avg = df[['accuracy', 'precision', 'recall', 'f1_score', 'score']].mean()

        # append new row to df
        df_new_row = pd.DataFrame({
            'task_name': 'total/avg',
            'total': df_sum['total'],
            'correct': df_sum['correct'],
            'wrong': df_sum['wrong'],
            'accuracy': df_avg['accuracy'],
            'precision': df_avg['precision'],
            'recall': df_avg['recall'],
            'f1_score': df_avg['f1_score']
        }, index=[df.index[-1]+1])
        df = pd.concat([df, df_new_row])
        return df
    
    
    
    def run_all_evaluations_full(self):
        eval_dict = {
                  'IR_pmid2title': self.IR_pmid2title_pubmedlink2title_eval,
                 'IR_abstract2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                 'IR_pubmedlink2title': self.IR_pmid2title_pubmedlink2title_eval,
                 'reasoning_nota': self.reasoning_nota_eval,
                 'IR_title2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                 'reasoning_fake': self.reasoning_fake_eval,
                 'reasoning_FCT': self.reasoning_functional_eval
                }
        
        
        all_datas_df = []
        
        for each_dataset_ in tqdm(self.all_sub_folders):
            print("calcuation for", each_dataset_)
            self.full_data = self.read_json(each_dataset_)
            evaluations_temp = []
        
            for sample in self.full_data:
                evaluation_func = eval_dict[sample['testbed_data']['dataset_name']]
                evaluation_result = evaluation_func(sample, sample['testbed_data']['dataset_name'])
                evaluations_temp.append(evaluation_result)

            
            df = pd.concat(evaluations_temp)
            df = df.groupby('task_name').sum().reset_index()
            
            df = self.finalise_dataframe(df)
            df = df[df['task_name']=='total/avg']
            df['task_name'] = each_dataset_.split('/')[1].split('.json')[0]
            all_datas_df.append(df)
        
        df_fu = pd.concat(all_datas_df)
        return df_fu
    

# evaluator = FullDataEval('./Prompt_Test')
# dfr = evaluator.run_all_evaluations_full()
# dfr
