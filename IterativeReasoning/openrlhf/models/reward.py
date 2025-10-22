from ..trainer.select_utils import SelectedData
from ..datasets import SelectedDataset

class RewardModel:
    def __init__(self, dataset: SelectedDataset):
        self.evaluator = dataset.evaluator
        self.data_postprocess = dataset.data_postprocess
        self.dataset_postprocess = dataset.dataset_postprocess

    def evaluate(self, state):
        return self.evaluator.evaluate(state)
    
    def select(self, generated_data: SelectedData, select_policy='oracle'):
        """
        Select the data from the data_list according to the select_policy.
        Args:
            data_list: list of data
            select_policy: str, 'oracle' or 'llama3-instruct'
        """
        if select_policy == 'oracle':
            # Select the data with correct answer according to the ground truth
            generations = generated_data.answers
            gts = generated_data.gts
            preds = [self.data_postprocess(generation) if self.data_postprocess else generation for generation in generations]
            refs = [self.dataset_postprocess(gt) if self.dataset_postprocess else gt for gt in gts]
            result = self.evaluator.score(predictions=preds, references=refs)
            details = result['details']  # detail = {'pred': pred, 'answer': ref, 'correct': True or False}
            correct_indices = [i for i, detail in enumerate(details) if detail['correct']]
            selected_data = SelectedData(
                questions=[generated_data.questions[i] for i in correct_indices],
                prompts=[generated_data.prompts[i] for i in correct_indices],
                answers=[generated_data.answers[i] for i in correct_indices],
                gts=[generated_data.gts[i] for i in correct_indices],
                info={}
            )
            # selected_data = SelectedData()
            # for key, value in generated_data.items():
            #     if isinstance(value, list):
            #         selected_data[key] = [value[i] for i in correct_indices]
        else:
            raise NotImplementedError
        
        return selected_data