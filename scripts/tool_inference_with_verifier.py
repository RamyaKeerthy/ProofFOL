import json
import os
from fol_solver.prover9_solver import FOL_Prover9_Program
import argparse
from tqdm import tqdm

class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split

        self.dataset = self.load_logic_programs()
        program_executor_map = {'FOLIO': FOL_Prover9_Program}
        self.program_executor = program_executor_map[self.dataset_name]

    def load_logic_programs(self):
        with open('./outputs/{data}_{model}{method}_fol_finetuned.json', 'r') as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def save_results(self, outputs):
        # Change
        with open('./outputs/tool/{data}_{model}{method}_fol_finetuned-tool.json', 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def get_fol(self, logic_program, verifier=False):
        try:
            premises_string = logic_program.split("Conclusion_First-order:")[0].split("Premise_First-order:")[
                1].strip('\n;')  # When no premise, it fails
            conclusion_string = logic_program.split("Conclusion_First-order:")[-1].split('\n')[0].strip('\n;')

            # Extract each premise and the conclusion using regex
            premises = premises_string.strip().split(';')
            conclusion = conclusion_string.strip()

            logic_premises = [premise.split(':::')[0].split('::')[0].replace("\'", "").strip() for premise in
                              premises if ':::' in premise or '::' in premise]
            logic_conclusion = conclusion.split(':::')[0].split('::')[0].strip('*').strip()
            return logic_premises, logic_conclusion
        except Exception as e:
            return [],''

    def safe_execute_program(self, id, logic_program):
        logic_premises, logic_conclusion = self.get_fol(logic_program)[0], self.get_fol(logic_program)[1]
        program = self.program_executor(logic_program, logic_premises, logic_conclusion, self.dataset_name)
        print(f"ID: {id}")
        # cannot parse the program
        if program.flag == False:
            answer = 'None'
            print('parsing error')
            return answer, 'parsing error', ''
        # execute the program
        answer, error_message = program.execute_program()
        if error_message != "":
            print("ID| ", id, "Error message| ", str(error_message).strip())
        # not executable
        if answer is None:
            answer = 'None'
            print('execution error')
            return answer, 'execution error', error_message
        # successfully executed
        return answer, 'success', ''

    def inference_on_dataset(self):
        outputs = []
        error_count = 0
        for example in tqdm(self.dataset):
            answer, flag, error_message = self.safe_execute_program(example['id'], example['fol'][0].strip())
            if not flag == 'success':
                error_count += 1

            output = {'id': example['id'],
                      'input': example['input'],
                      'label': example['label'],
                      'predicted_logic': example['fol'][0],
                      'flag': flag,
                      'error': error_message,
                      'predicted_answer': answer}
            outputs.append(output)
        
        print(f"Error count: {error_count}")
        self.save_results(outputs)
        self.cleanup()

    def cleanup(self):
        complied_krb_dir = './compiled_krb'
        if os.path.exists(complied_krb_dir):
            print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='FOLIO')
    parser.add_argument('--split', type=str, default='dev')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()
