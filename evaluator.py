import os
import yaml
import glob
import torch
import logging
import numpy as np

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from typing import Dict, Union, List, Any, Tuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from utils import read_jsonl, write_json
from models.model_base import PreTrainedModelWrapper
from models.model_dpo import AutoDPOModelForCausalLM, AutoDPOModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mnlp-2024-auto-evaluator")

def repository_check():
    """
    Check if the repository is correctly set up.
    """
    assert os.path.isdir("checkpoints"), "You must have a directory named 'checkpoints' in the current directory."
    assert os.path.isdir("models"), "You must have a directory named 'models' in the current directory."

    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    assert 'main_config.yaml' in files, "main_config.yaml not found in the current directory."
    assert 'requirements.txt' in files, "requirements.txt not found in the current directory."
    assert 'utils.py' in files, "utils.py not found in the current directory."

    files = [f for f in os.listdir('models') if os.path.isfile('models/' + f)]
    assert 'model_base.py' in files, "model_base.py not found in the current directory."
    assert 'model_dpo.py' in files, "model_dpo.py not found in the current directory."

    # Assert that the documents are submitted in the Doc folder

class DPOModelEvaluator():
    def __init__(
        self,
        task_type: str="causal_lm",
        policy_model_path: str=None,
        reference_model_path: str=None,
        dpo_model_args: dict={}
    ):
        if task_type == "causal_lm":
            self.model_class = AutoDPOModelForCausalLM
        elif task_type == "seq2seq":
            self.model_class = AutoDPOModelForSeq2SeqLM
        else:
            raise ValueError("Invalid task type! Please choose from 'causal_lm' or 'seq2seq'.")

        self.policy_model_path = policy_model_path
        self.reference_model_path = reference_model_path
        if dpo_model_args is None:
            self.dpo_model_args = {}
        else:
            self.dpo_model_args = dpo_model_args

        self.policy_tokenizer = self.load_tokenizer(policy_model_path)
        if reference_model_path != None:
            self.reference_tokenizer = self.load_tokenizer(reference_model_path)
        else:
            self.reference_tokenizer = None

    def load_tokenizer(self, model_path: str):
        """Load the tokenizer for a given model.

        Args:
            model_path: The path to the model.
        Returns:
            The tokenizer for the model.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}!")
            logger.error("Important! You must properly save the tokenizer in the model directory!")

    def get_batch_predictions_mcqa(
        self,
        model: PreTrainedModelWrapper,
        tokenizer: PreTrainedTokenizerBase,
        batch: Dict[str, List[Union[str, List[str]]]]
    ):
        """Get the predictions of a student's DPO model for a batch of multiple-choice questions.

        Args:
            model (`PreTrainedModelWrapper`): A student's DPO model.
            tokenier (`PreTrainedTokenizerBase`): A tokenizer of the student's DPO or Reference model.
            batch (`dict` of `list`): A dictrionary containing the input MCQA questions data for the DPO model.
                   The data format is as follows:
                    {
                        "question": List[str], each <str> contains the question body and the choices.
                        "answer": List[str], each <str> is a single letter representing the correct answer.
                    }
        Returns:
           preds (`list` of `str`): A list of predicted choices for the MCQA questions.
        """
        output_dict = model.prediction_step_mcqa(batch, tokenizer)
        preds = output_dict["preds"]
        return preds

    def scoring_mcqa(self, test_dataloader: DataLoader):
        """Computing the accuracy of the multiple-choice question predictions from a DPO model.

        Args:
            test_dataloader (DataLoader): A pytorch dataLoader containing the test MCQA data.
        Returns:
            policy_accuracy (float): The mcqa accuracy of the policy model.
        """
        all_policy_preds = []
        all_labels = []

        # Load the policy model from the policy model path
        policy_model = self.model_class.from_pretrained(
            self.policy_model_path,
            **self.dpo_model_args)

        # Iterate over the test data and get the predictions from the policy model
        for _, batch in enumerate(test_dataloader):
            policy_preds = self.get_batch_predictions_mcqa(
                policy_model, self.policy_tokenizer, batch)

            all_policy_preds.extend(policy_preds)
            all_labels.extend(batch["answer"])

        # Clear the GPU memory allocated for the policy model
        del policy_model
        torch.cuda.empty_cache()

        policy_accuracy = accuracy_score(
            y_true=all_labels,
            y_pred=all_policy_preds)
        return policy_accuracy

    def get_batch_predictions_reward(
        self,
        model: PreTrainedModelWrapper,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """Get the reward estimations of a student's DPO model for a batch of preference pairs.

        Args:
            model (`PreTrainedModelWrapper`): A student's DPO or Reference model.
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
           chosen_rewards (`list` of `float`): A list of rewards for the chosen responses.
           rejected_rewards (`list` of `float`): A list of rewards for the rejected responses.
        """
        output_dict = model.prediction_step_reward(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        chosen_rewards = output_dict["chosen_rewards"]
        rejected_rewards = output_dict["rejected_rewards"]
        return chosen_rewards, rejected_rewards

    def compute_reference_logprobs(self, test_data: List[Dict]):
        """Compute the log probabilities of the reference model for the test data.

        Args:
            test_data (`list` of `dict`): A list of dictionaries containing the test preference data for reward computation.
                    The data format is as follows:
                        {
                            "prompt": str,
                            "chosen": str,
                            "rejected": str,
                        }

        Returns:
            new_test_data (`list` of `dict`): A list of dictionaries containing the test preference data with logprobs.
        """
        assert self.reference_model_path is not None, "You must provide the path to the reference model for reward computation!"
        assert self.reference_tokenizer is not None, "You must provide the tokenizer for the reference model for reward computation!"

        test_data_map = {}
        for data in test_data:
            test_data_map[data['prompt']] = {}
        test_dataloader = DataLoader(test_data, batch_size=8)
        reference_model = self.model_class.from_pretrained(
            self.reference_model_path)

        for idx, batch in enumerate(test_dataloader):
            try:
                chosen_logps, rejected_logps = reference_model.get_logprobs(batch, self.reference_tokenizer)
            except Exception as e:
                logger.error(f"Error in batch {idx}: {e}! \n Please check your implementation and the return format!")
                continue
            for prompt, chosen_logp, rejected_logp in zip(batch["prompt"], chosen_logps, rejected_logps):
                test_data_map[prompt]["chosen_logps"] = chosen_logp
                test_data_map[prompt]["rejected_logps"] = rejected_logp

        del reference_model
        torch.cuda.empty_cache()

        for data in test_data:
            data["chosen_logps"] = test_data_map[data['prompt']]["chosen_logps"]
            data["rejected_logps"] = test_data_map[data['prompt']]["rejected_logps"]

        del test_data_map

        # new_test_data = list(test_data_map.values())
        return test_data

    def scoring_reward_computation(self, test_dataloader: DataLoader):
        """Computing the accuracy of the preference reward estimations from a DPO model.

        Args:
            test_dataloader (`DataLoader`): A pytorch dataLoader containing the test preference data pairs.
                    The batch data format is as follows:
                        {
                            "prompt": List[str],
                            "chosen_response": List[str],
                            "rejected_response": List[str],
                            "chosen_logps": List[torch.FloatTensor],
                            "rejected_logps": List[torch.FloatTensor],
                        }

        Returns:
            policy_reward_accuracy (`float`): The accuracy of the policy model in estimating the rewards.
        """
        policy_chosen_rewards = []
        policy_rejected_rewards = []
        policy_model = self.model_class.from_pretrained(
            self.policy_model_path,
            **self.dpo_model_args
        )

        for idx, batch in enumerate(test_dataloader):
            try:
                reference_chosen_logps = batch["chosen_logps"]
                reference_rejected_logps = batch["rejected_logps"]
                policy_chosen_logps, policy_rejected_logps = policy_model.get_logprobs(
                    batch, self.policy_tokenizer)
                rewards = self.get_batch_predictions_reward(
                    policy_model,
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps)
            except Exception as e:
                logger.error(f"Error in batch {idx}: {e}! \n Please check your implementation and the return format!")
                continue
            policy_chosen_rewards.extend(rewards[0])
            policy_rejected_rewards.extend(rewards[1])

        policy_chosen_rewards = np.array(policy_chosen_rewards)
        policy_rejected_rewards = np.array(policy_rejected_rewards)
        policy_reward_accuracy = (policy_chosen_rewards > policy_rejected_rewards).sum()
        policy_reward_accuracy = policy_reward_accuracy / len(policy_chosen_rewards)

        del policy_model
        torch.cuda.empty_cache()

        return policy_reward_accuracy

class RAGModelEvaluator():
    def __init__(
        self,
        task_type: str="causal_lm",
        rag_policy_model_path: str=None,
        rag_model_args: dict={}
    ):
        self.rag_policy_model_path = rag_policy_model_path
        self.rag_model_args = rag_model_args

        self.rag_dpo_evaluator = DPOModelEvaluator(
            task_type=task_type,
            policy_model_path=rag_policy_model_path,
            dpo_model_args=rag_model_args
        )

    def scoring_rag(self, test_dataloader: DataLoader):
        """Computing the accuracy of the multiple-choice question predictions from a RAG model.

        Args:
            test_dataloader (DataLoader): A pytorch dataLoader containing the test MCQA data.

        Returns:
            rag_accuracy (float): The mcqa accuracy of the RAG model.
        """
        rag_accuracy = self.rag_dpo_evaluator.scoring_mcqa(test_dataloader)
        return rag_accuracy

class QuantizedEvaluator():
    def __init__(
        self,
        task_type: str="causal_lm",
        policy_model_path: str=None,
        quantized_model_path: str=None,
        policy_model_args: dict={},
        quantized_model_args: dict={}
    ):
        assert policy_model_path is not None, "You must provide the path to the policy model!"
        assert quantized_model_path is not None, "You must provide the path to the quantized model!"

        if task_type == "causal_lm":
            self.model_class = AutoDPOModelForCausalLM
        elif task_type == "seq2seq":
            self.model_class = AutoDPOModelForSeq2SeqLM
        else:
            raise ValueError("Invalid task type! Please choose from 'causal_lm' or 'seq2seq'.")

        self.policy_model_path = policy_model_path
        self.quantized_model_path = quantized_model_path

        self.quantized_model_args = quantized_model_args
        self.policy_model_args = policy_model_args

        self.quantized_dpo_evaluator = DPOModelEvaluator(
            task_type=task_type,
            policy_model_path=quantized_model_path,
            dpo_model_args=quantized_model_args
        )

    def scoring_quantization(self, test_dataloader: DataLoader):
        """Computing the accuracy of the multiple-choice question predictions from a quantized DPO model.

        Args:
            test_dataloader (DataLoader): A pytorch dataloader containing the test MCQA data.

        Returns:
            quantized_accuracy (float): The mcqa accuracy of the quantized policy model.
        """
        quantized_accuracy = self.quantized_dpo_evaluator.scoring_mcqa(test_dataloader)
        return quantized_accuracy

    def check_model_quantization(self):
        """Check if the model is quantized by comparing the model sizes (orig vs. quantized) on disk.

        Returns:
            compessed_model_size_on_disk (int): The size of the quantized model on disk.
            orig_model_size_on_disk (int): The size of the original model on disk.
            quantized (bool): A boolean indicating if the model is quantized.
        """
        quantized = False

        policy_model = self.model_class.from_pretrained(
            self.policy_model_path,
            **self.policy_model_args)
        policy_model_footprint = policy_model.pretrained_model.get_memory_footprint()
        del policy_model

        quantized_model = self.model_class.from_pretrained(
            self.quantized_model_path,
            **self.quantized_model_args)
        quantized_model_footprint = quantized_model.pretrained_model.get_memory_footprint()
        del quantized_model

        if quantized_model_footprint < policy_model_footprint:
            quantized = True

        return quantized_model_footprint, policy_model_footprint, quantized

if __name__ == '__main__':
    # Basic repository check to ensure the submission is correct
    repository_check()

    # Load the main configuration file
    main_config = {}
    with open("main_config.yaml") as f:
        try:
            main_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading main_config.yaml: {e}! Please check the file format.")

    # Load the task type to identify the model class
    task_type = main_config.get("task_type", "causal_lm")

    # Load the evaluation methods and the required paths
    eval_method = main_config.get("eval_method", ["mcqa"])
    policy_model_path = main_config["policy_model_path"]
    reference_model_path = main_config["reference_model_path"]
    test_data_path = main_config["test_data_path"]

    # Load the test data
    test_data = read_jsonl(test_data_path)

    # Load the model arguments
    dpo_model_args = main_config.get("dpo_model_args", {})
    rag_model_args = main_config.get("rag_model_args", {})
    quantized_model_args = main_config.get("quantized_model_args", {})

    # Initialize the metrics dictionary
    metrics = {
        "team_name": main_config.get("team_name", "Team Name"),
        "task_type": task_type,
    }

    # Ensure that the evaluation methods are not conflicting
    assert not ("reward" in eval_method and "mcqa" in eval_method), "You cannot evaluate both reward and mcqa at the same time!"

    # Initialize the evaluator based on the evaluation method and compute the metrics
    if "reward" in eval_method:
        evaluator = DPOModelEvaluator(
            task_type=task_type,
            policy_model_path=policy_model_path,
            reference_model_path=reference_model_path,
            dpo_model_args=dpo_model_args
        )
        # Compute the log probabilities of the reference model for the test data
        new_test_data = evaluator.compute_reference_logprobs(test_data)
        test_dataloader = DataLoader(new_test_data, batch_size=8)
        # compute the reward accuracy
        policy_reward_acc = evaluator.scoring_reward_computation(test_dataloader)
        metrics["policy_reward_accuracy"] = policy_reward_acc

    elif "mcqa" in eval_method:
        test_dataloader = DataLoader(test_data, batch_size=8)
        evaluator = DPOModelEvaluator(
            task_type=task_type,
            policy_model_path=policy_model_path,
            dpo_model_args=dpo_model_args
        )
        policy_acc= evaluator.scoring_mcqa(test_dataloader)
        eval_method.remove("mcqa")
        metrics["policy_acc"] = policy_acc

        # Loop over the remaining evaluation methods
        for method in eval_method:
            if method == "rag":
                # Check if the documents directory is present
                assert os.path.isdir("documents"), "You must have a directory named 'documents' with all the documents used for RAG, in the current directory."
                rag_policy_model_path = main_config["rag_policy_model_path"]
                evaluator = RAGModelEvaluator(
                    task_type,
                    rag_policy_model_path,
                    rag_model_args)
                rag_policy_acc = evaluator.scoring_rag(test_dataloader)
                metrics["rag_policy_acc"] = rag_policy_acc
            elif method == "quantiz":
                quantized_model_path = main_config["quantized_policy_model_path"]
                evaluator = QuantizedEvaluator(
                    task_type=task_type,
                    policy_model_path=policy_model_path,
                    quantized_model_path=quantized_model_path,
                    policy_model_args=dpo_model_args,
                    quantized_model_args=quantized_model_args)
                orig_size, quantized_size, quantized = evaluator.check_model_quantization()
                metrics["orig_model_size"] = orig_size
                metrics["quantized_model_size"] = quantized_size
                metrics["quantized"] = quantized
                if not quantized:
                    logger.error("Urgent! An error occurred that might result in 0 points!")
                    logger.error("Error: quantized model size should be less than the original model size!")
                quantized_policy_acc = evaluator.scoring_quantization(test_dataloader)
                metrics["quantized_policy_acc"] = quantized_policy_acc

        logger.info("Evaluation Completed! Results:")
        logger.info(metrics)

    # Write the metrics to a JSON file
    write_json(metrics, "metrics.json")
