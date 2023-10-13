# Med-HALT: Medical Domain Hallucination Test for Large Language Models

This is the code repository used in the [Med-HALT](https://arxiv.org/abs/2307.15343) research paper. This research paper focuses on the challenges posed by hallucinations in large language models (LLMs), particularly in the context of the medical domain. We propose a new benchmark and dataset, Med-HALT (Medical Domain Hallucination Test), designed specifically to evaluate hallucinations. 

Med-HALT provides a diverse multinational dataset derived from medical examinations across various countries and includes multiple innovative testing modalities. Med-HALT includes two categories of tests reasoning and memory-based hallucination tests, designed to assess LLMs' problem-solving and information retrieval abilities. Our study evaluated leading LLMs, including Text Davinci, GPT-3.5, LlaMa and Falcon, revealing significant differences in their performance. The paper provides detailed insights into the dataset, promoting transparency and reproducibility. Through this work, we aim to contribute to the development of safer and more reliable language models in healthcare. Our benchmark can be found at https://github.com/medhalt/medhalt

## Benchmark

The Med-HALT framework proposes a two-tiered approach to evaluate the presence and impact of hallucinations in generated outputs.

#### Reasoning Hallucination Tests (RHTs)

<details>
<summary>False Confidence Test (FCT)</summary>

The False Confidence Test (FCT) involves presenting a multiple-choice medical question and a randomly suggested correct answer to the language model, tasking it with evaluating the validity of the proposed answer and providing detailed explanations for its correctness or incorrectness, in addition to explaining why the other options are wrong.

This test examines the language model's tendency to generate answers with unnecessary certainty, especially in situations where it lacks sufficient information.
</details>

<details>
<summary>None of the Above Test (Nota)</summary>

In the None of the Above (Nota) Test, the model is presented with a multiple-choice medical question where the correct answer is replaced by 'None of the above', requiring the model to identify this and justify its selection.

It tests the model's ability to distinguish irrelevant or incorrect information.
</details>

<details>
<summary>Fake Questions Test (FQT)</summary>

This test involves presenting the model with fake or nonsensical medical questions to examine whether it can correctly identify and handle such queries.

We employed a hybrid approach for generating fake questions, where a subset was crafted by human experts, while the remaining were generated using GPT-3.5.
</details>

#### Memory Hallucination Tests (MHTs)

<details>
<summary>Abstract-to-Link Test</summary>

Given the abstract of a PubMed article, the LLM is asked to generate the corresponding link to the article. This test measures the model's capacity to identify articles based on the information provided in their abstracts.
</details>

<details>
<summary>PMID-to-Title Test</summary>

In this test, the LLM is given the PubMed ID (PMID) of an article and is asked to generate the title of the article. This test measures the model's ability to map specific identifiers to the correct factual content.
</details>

<details>
<summary>Title-to-Link Test</summary>

Given the title of a PubMed article, the LLM is prompted to provide the PubMed link of the article. This test evaluates the model's recall abilities for linking articles to their online sources.
</details>

<details>
<summary>Link-to-Title Test</summary>

Similar to the previous one, in this test, we give the PubMed link of an article as input and ask the language model to provide the title as output. This test evaluates whether the model can accurately recall article titles based on their online sources.
</details>

# Dataset

Datasets are in `medhalt/datasets` directory. Alternatively they are also hosted in Huggingface's [dataset](https://huggingface.co/datasets/MedHALT/Med-HALT)

## Evaluation Instructions

1. Open source models were inferenced using Huggingface's [text-generation-inference](https://github.com/huggingface/text-generation-inference) library . Spin up an TGI inference server using the below command:

```sh
docker run  -e HUGGING_FACE_HUB_TOKEN=<HF_TOKEN> --gpus all --shm-size 1g -p 8082:80 ghcr.io/huggingface/text-generation-inference:0.8.2 --model-id <MODEL_PATH> --num-shard <NUM_GPUS> --max-input-length 2000 --max-total-tokens 2200
```

2. Run inference of the model

```sh
sh run_inference.sh <model_id>
```

3. Run evaluation

```sh

sh run_eval.sh <path_to_dataset_folder> <path_to_save_predictions>

```

## Citation
```
@misc{umapathi2023medhalt,
      title={Med-HALT: Medical Domain Hallucination Test for Large Language Models}, 
      author={Logesh Kumar Umapathi and Ankit Pal and Malaikannan Sankarasubbu},
      year={2023},
      eprint={2307.15343},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
