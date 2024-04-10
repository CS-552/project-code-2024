# CS-552 MNLP 2024 Project M2 & M3 Description (Code Part)

## Project Description (Code Part)

- In this project, you will be working on developing large language models (LLMs) capable of communicating with humans and assisting with various tasks. You will be exploring various frontier methods to apply LLMs in real-world applications, including Direct Preference Optimization (DPO), Retrieval-Augmented Generation (RAG), and model quantization.
In Milestone 2, you will implement the alignment method Direct Preference Optimization (DPO), which is an effective way to align a foundation LLM with human preferences.
- In Milestone 3, you will continue to improve your DPO training pipeline to adapt your LLM model to support multiple-choice question answering (MCQA) tasks. Next, you will implement either **Retrieval-Augmented Generation (RAG)** to enhance the model's reasoning with external knowledge or **model quantization** to get lightweight and efficient models for easier real-world deployment. **Groups with 4 people must implement both RAG and quantization. Groups with 3 people can pick one of the directions. Every group must complete the development of the MCQA DPO model.**
- For the complete project description (including other deliverables) in addition to the code implementations, please refer to the [official document](https://docs.google.com/document/d/1SP8SCHPOZZGEhs2ay-38FjedRE1bS9Q99VJb28eHoYk/edit?usp=sharing)

## Codebase File Structure

```txt
.
├── checkpoints
│   ├── your_checkpoint_name
│   │   ├── config.json
│   │   |── model.safetensor
│   │   └── ...
├── datasets
│   │   ├── your_dataset_name
│   │   │   └── ...
│   │   └── ...
├── documents (For RAG only)
├── models
│       ├── model_base.py
│       └── model_dpo.py
├── utils.py
├── evaluator.py
├── main_config.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

### Setup via Conda Virtual Environment

```bash
# Replace <my-env> with the name of your environment.
conda create --name <my-env> python=3.10.11
conda activate <my-env>

# Install dependencies from a `requirements.txt`
pip install -r requirements.txt
# If you intend to use flash-attention for more efficient training and inference
pip install flash-attn --no-build-isolation
```

### Setup via Docker Container

```bash
# Replace <my-docker> with the name of your docker image.
docker build -f Dockerfile . -t <my-docker>
docker run <my-docker>
docker exec -it <my-docker> bash
# Continue any bash operations ...

# Replace <num-gpu> with the number of GPUs you have
sudo docker run --gpus <num-gpu> -it -d  \
    --name $NAME \
    --rm --shm-size=128gb \
    --network host \
    -v /pure-mlo-scratch:/pure-mlo-scratch \
    -v /home:/home meditron \
    -- /bin/bash -c 'bash'
```

## Codebase Introduction

- `model_base.py`: In this file, you will find a wrapper model class `PreTrainedModelWrapper` around a (`transformers.PreTrainedModel`) to be compatible with the (`~transformers.PreTrained`) class in order to keep some attributes and methods of the (`~transformers.PreTrainedModel`) class. You can save a checkpoint through `PreTrainedModelWrapper.save_pretrained(model_name_or_path)` and load a checkpoint locally or from HuggingFace hub through the method `PreTrainedModelWrapper.from_pretrained(model_name_or_path)`.
- `model_dpo.py`: In this file, you will implement your DPO model. Read and complete the TODOs. Note that TODO (Optional) is not required; You only need to do these if you want to add custom modules to your model. If you are working with a causal language model like GPT-2 or LLama2, use the `AutoDPOModelForCausalLM` class. If you are working with a sequence-to-sequence language model like T5 or Bart, use the `AutoDPOModelForSeq2SeqLM` class. The functions that are required for all to implement including `forward`, `prediction_step_reward`, and `prediction_step_mcqa`.
- In addition to a transformer model, you can add custom modules to the `AutoDPOModel` classes. Below is an example custom module. You can follow the `TODO (Optional)` to integrate a custom module into the main model.

### Basic Model functionalities

For `AutoDPOModelForCausalLM` and `AutoDPOModelForSeq2SeqLM`, which both inherit `PreTrainedModelWrapper`, you have the following basic operations:

**Load from a pre-trained model listed in [Huggingface Hub](https://huggingface.co/models)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.model_dpo import AutoDPOModelForCausalLM

# Download the pre-trained model and tokenizer from the Hub
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize your model class and import the pre-trained model into your class
# Note that if you have a custom module in your class
# You should initialize the weights of this module in the `__init__` function
model_wrapper = AutoDPOModelForCausalLM(pretrained_model=model)
```

**Save your model as a Huggingface transformers compatible checkpoint**

```python
# Save your model and tokenizer to the checkpoint directory `models/dpo_gpt2`
checkpoint_dir = "models/dpo_gpt2"
model_wrapper.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)
```

**Load from your local checkpoint**

```python
checkpoint_dir = "models/dpo_gpt2"
model_wrapper = AutoDPOModelForCausalLM.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
```

### Custom Module Example

```python
class CustomModule(nn.Module):
    """
    This is only a dummy example of a custom module. You can replace this with your own custom module.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.summary(output)
        return output
```

- `evaluator.py` is the main evaluation script that we will use to collect your model performance and implementation quality in order to assign you a grade. To execute this script, you first need to specify details in the `main_confi.yaml` configuration file. The details in this config file will be used by the evaluation script to execute the grading properly. Make sure you fill all the important information in the config file.

### Main Configuration Arguments

```yaml
"team_name": "Team 1" # Your team name
"eval_method": ["mcqa", "rag"] # Tells the evaluator which evaluations need to be executed. choices = [mcqa, reward, rag, compression]
"task_type": "causal_lm" # Identifies which model class you use. choices = [causal_lm, seq2seq]
"policy_model_path": "./checkpoints/best_model/" # Your path to the final checkpoint
"reference_model_path": "microsoft/phi-2" # The repo id of your pretrained DPO reference model
"quantized_policy_model_path": "./checkpoints/best_model_quantized/" # Your path to the final quantized checkpoint
"rag_policy_model_path": "./checkpoints/best_model_rag/" # Your path to the final RAG checkpoints
"test_data_path": "./data/test.json" # Your path to the test data. (We will replace it with the official test sets when grading)
"dpo_model_args": null # Any required arguments to load your dpo model using "from_pretrained"
"rag_model_args": # Any required arguments to load your rag model using "from_pretrained" For example:
    "encoder_model_path": "facebook/bart-large"
    "retriever_model_path": "./checkpoints/rag_retriever"
    "document_dir": "./data/documents"
"quantized_model_args": null # Any required arguments to load your quantized model using "from_pretrained"
```

- Note: `eval_method`'s value must be a list object.

- Note: `reward` and `mcqa` cannot co-exist in the `eval_method` list at the same time.

Please review the evaluation script code for detailed evaluation methods and the input and output of each evaluation function.

## Deliverables (for the coding part of the project)

- [ ] The dpo model file: `model_dpo.py`
  - [ ] `forward`,
  - [ ] `predict_step_reward`,
  - [ ] `predict_step_mcqa`,
  - [ ] Any code related to your `custom_module` (optional, only if you need custom modules)
  - [ ] Any helper or utility functions required for your implementation.
- [ ] The YAML file contains information for evaluation
- [ ] The `checkpoints` directory with all your checkpoints in it. (Note: a group of 3 only needs to deliver either the RAG or the Quantized checkpoints in addition to the DPO checkpoint, which is required for all groups)
  - [ ] DPO model checkpoint
  - [ ] RAG model checkpoints
  - [ ] Quantized model checkpoints
  - [ ] Any support model checkpoints (if required for your implementation)
  - [ ] All tokenizer models used by your implementation
- [ ] `requirements.txt` includes all the dependencies required by your implementation
- [ ] (Required for groups doing RAG) `documents` directory that contains all the documents you need to retrieve from. You can take a look at some example documents in this [Kaggle Challenge](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset?resource=download).
