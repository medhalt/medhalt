import pandas as pd
import glob
from tqdm import tqdm
import json

class FullDataEval(object):
    
    def __init__(self, folder_name, correct_score=1, incorrect_score=-0.25):
        self.evaluations = []
        self.all_files = {k.split('.json')[-2].split('/')[2]: k for k in glob.glob(f'./{folder_name}/*json')}
        
        self.correct_score   = correct_score
        self.incorrect_score = incorrect_score

    def read_json(self, file):
        with open(file, 'r') as json_file:
            file_data = json.load(json_file)
        return file_data

    def evaluate_answer(self, predicted, correct):
        return str(predicted.lower()) == str(correct.lower())

    def create_dataframe(self, task_name, correct, wrong, score):
        total = correct + wrong
        df_dict = {'task_name': [task_name], 'total': [total], 'correct': [correct], 'wrong': [wrong], 'score': [score]}
        return pd.DataFrame(df_dict)

    def handle_exceptions(self, task_name, sample, exception):
        print(task_name, sample['id'], exception)
        return 1
    
    def calculate_score(self, correct, wrong):
        return (correct * self.correct_score + wrong * self.incorrect_score) / 100

    def reasoning_functional_eval(self, data, task_name):
        correct, wrong, exception_count = 0, 0, 0
        all_files_data = self.read_json(self.all_files[data])

        possible_keys = ['correct_answer', 'answer', 'correct answer', 'corrent_answer', 'Correct Answer', 
                         'Answer', 'Correct_answer', "Correct answer"]

        

        for sample in tqdm(all_files_data):
            try:
                predicted_answer = None
                for key in possible_keys:
                    if key in sample['gpt_output']:
                        predicted_answer = sample['gpt_output'][key]
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
        
        print(result)
        score = self.calculate_score(correct, wrong)  # calculate the score here
        return self.create_dataframe(task_name, correct, wrong, score)


    def IR_pmid2title_pubmedlink2title_eval(self, data, task_name):
        correct = wrong = exception_count = 0
        all_files_data = self.read_json(self.all_files[data])

        for sample in tqdm(all_files_data):
            try:
                predicted_title = sample['gpt_output']['paper_title']

                if self.evaluate_answer(predicted_title, sample['testbed_data']['Title']):
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                exception_count += self.handle_exceptions(task_name, sample, e)

        print(f"correct {correct} wrong -> {wrong} exception_count {exception_count}")
        score = self.calculate_score(correct, wrong)  # calculate the score here
        return self.create_dataframe(task_name, correct, wrong, score)

    def IR_title2pubmedlink_abstract2pubmedlink_eval(self, data, task_name):
        correct = wrong = exception_count = 0
        all_files_data = self.read_json(self.all_files[data])

        for sample in tqdm(all_files_data):
            try:
                predicted_url = sample['gpt_output']['url']

                if self.evaluate_answer(predicted_url, sample['testbed_data']['url']):
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                exception_count += self.handle_exceptions(task_name, sample, e)

        print(f"correct {correct} wrong -> {wrong} exception_count {exception_count}")
        score = self.calculate_score(correct, wrong)  # calculate the score here
        return self.create_dataframe(task_name, correct, wrong, score)

    def reasoning_nota_eval(self, data, task_name):
        correct = wrong = exception_count = 0
        all_files_data = self.read_json(self.all_files[data])

        for sample in tqdm(all_files_data):
            try:
                predicted_answer = str(sample['gpt_output']['cop'])

                if self.evaluate_answer(predicted_answer, sample['testbed_data']['correct_answer']):
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                exception_count += self.handle_exceptions(task_name, sample, e)

        print(f"correct {correct} wrong -> {wrong} exception_count {exception_count}")
        score = self.calculate_score(correct, wrong)  # calculate the score here
        return self.create_dataframe(task_name, correct, wrong, score)

    def reasoning_fake_eval(self, data, task_name):
        correct = wrong = exception_count = 0
        all_files_data = self.read_json(self.all_files[data])

        for sample in tqdm(all_files_data):
            try:
                predicted_answer = str(sample['gpt_output']['cop']).lower()

                if any(term in predicted_answer for term in ['i do not know', 'conceding defeat', 'admit', 'none of the above',
                                                              'acknowled', 'irrelevant', 'fiction', 'all of the above', 
                                                              'nonsensical', 'no correct', 'absurd', 'defy', 'i don"t know.', 
                                                              'defies']):
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                exception_count += self.handle_exceptions(task_name, sample, e)

        print(f"correct {correct} wrong -> {wrong} exception_count {exception_count}")
        score = self.calculate_score(correct, wrong)  # calculate the score here
        return self.create_dataframe(task_name, correct, wrong, score)
    
    
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
        return df
    
    
    def run_all_evaluations(self):
        
        eval_dict = {
                 'vinci_IR_pmid2title': self.IR_pmid2title_pubmedlink2title_eval,
                 'vinci_reasoning_nota': self.reasoning_nota_eval,
                 'vinci_IR_title2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                 'gpt3_reasoning_fake': self.reasoning_fake_eval,
                 'gpt3_IR_pubmedlink2title': self.IR_pmid2title_pubmedlink2title_eval,
                 'vinci_IR_abstract2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                 'vinci_IR_pubmedlink2title': self.IR_pmid2title_pubmedlink2title_eval,
                 'gpt3_IR_title2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                 'gpt3_reasoning_nota': self.reasoning_nota_eval,
                 'gpt3_IR_pmid2title': self.IR_pmid2title_pubmedlink2title_eval,
                 'gpt3_IR_abstract2pubmedlink': self.IR_title2pubmedlink_abstract2pubmedlink_eval,
                 'gpt3_reasoning_FCT': self.reasoning_functional_eval,
                 'vinci_reasoning_fake': self.reasoning_fake_eval,
                 'vinci_reasoning_FCT': self.reasoning_functional_eval
                }

        for key in self.all_files:
            evaluation_func = eval_dict[key]
            evaluation_result = evaluation_func(key, key)
            self.evaluations.append(evaluation_result)

        df = pd.concat(self.evaluations)
        return self.finalise_dataframe(df)



#evaluator = FullDataEval('full_data_eval/')
#results_df = evaluator.run_all_evaluations()
