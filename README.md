![alt text](https://github.com/ziye2chen/LLMs-for-Mathematical-Analysis/blob/main/img/head.png)

# Large Language Models for Mathematical Analysis

Mathematical problem-solving is a key field in artificial intelligence (AI) and a critical benchmark for evaluating the capabilities of large language models (LLMs). While extensive research has focused on computational aspects of mathematics, there remains a significant gap in addressing proof-based tasks—particularly those found in mathematical analysis. Rigorous reasoning, precise definitions, and formal proofs are essential components of real analysis, yet these aspects have been largely overlooked in existing datasets and methodologies.

To bridge this gap, we introduce the **DEMI-MathAnalysis** dataset, a collection of proof-oriented problems drawn from topics such as Sequences and Limits, Infinite Series, and Convex Functions. Accompanying this dataset is a guiding framework designed to help LLMs navigate the complexities of formal reasoning, ensuring that they not only produce answers but also justify their steps in a logically consistent and mathematically sound manner.

By fine-tuning LLMs on DEMI-MathAnalysis and employing our framework, we have observed significant improvements in their capacity to generate logical, complete, and elegant proofs. This work not only highlights critical areas of mathematical reasoning previously underserved by AI research but also contributes to the development of more trustworthy AI models capable of understanding and producing formalized mathematical language.

## Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Framework and Model](#framework-and-model)
- [To Run the Models](#to-run-the-models)
- [Reference](#reference)

## Overview

The DEMI-MathAnalysis project provides:
1. A **Dataset**: A curated collection of proof-based real analysis problems sourced from authoritative texts.
2. A **Framework**: A pipeline to guide LLMs in understanding problems, retrieving relevant theorems and definitions, and generating logically sound proofs.

## Motivation

Most mathematical datasets for LLMs focus on computational questions, leaving a significant gap in training for formal reasoning tasks. DEMI-MathAnalysis addresses this gap by supplying proof-oriented challenges. By doing so, we help models develop the precise and structured thinking required in advanced mathematics.

## Dataset

**DEMI-MathAnalysis** covers topics often studied in undergraduate real analysis courses:
- Sequences and Limits
- Infinite Series
- Continuous Functions
- Differentiation
- Integration and Improper Integrals
- Series of Functions
- Approximation by Polynomials
- Convex Functions

The dataset is split into:
- **Training Set**: Used for fine-tuning models.
- **Benchmark Set**: Reserved for unbiased evaluation.

Each problem includes a detailed statement and metadata. Data files are available [**here**](https://people.eecs.berkeley.edu/~hendrycks/MATH.tar).

## Framework and Model

A special framework is built to guide LLMs through a structured approach to solving proof-based mathematical problems. It consists of the following key steps:

1. **Problem Identification**:  
   The LLM classifies each problem into a specific category (e.g., Sequences and Limits, Infinite Series, Convex Functions). This step ensures that the reasoning strategies and retrieved knowledge are tailored to the nature of the problem.

2. **Prompt Construction**:  
   Once categorized, the system constructs a comprehensive prompt. This includes the original problem statement, the identified category, and relevant domain knowledge from a curated knowledge base (KB). These structured prompts provide the model with the precise context and definitions needed for logical reasoning.

3. **Knowledge Base Integration**:  
   The KB contains essential definitions, theorems, and lemmas from real analysis. By dynamically retrieving the most relevant pieces of information, the framework equips the LLM with a foundation of reliable mathematical principles, ensuring that solutions rest on correct formal underpinnings.

4. **Solution Generation**:  
   The fine-tuned LLM, now guided by explicit reasoning paths and precise knowledge, produces a step-by-step proof. This involves leveraging methods like ε–δ arguments, limit definitions, and convergence criteria. The aim is to ensure that the solution is not only correct but also rigorously and formally justified.

<p align="center">
  <img src="https://github.com/ziye2chen/LLMs-for-Mathematical-Analysis/blob/main/img/framework.png" alt="head" width="600px" />
</p>

## To Run the Models

Install the required packages：

```bash
pip install -r requirements.txt
```

Run the code in [**RealAnalysis_Final_Code.ipynb**](https://github.com/ziye2chen/LLMs-for-Mathematical-Analysis/blob/main/RealAnalysis_Final_Code.ipynb)

If you want to change the model, change the **model_name** below:

```python
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "MathAnalysis_Qwen_Classifier", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 10240,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
```
```python
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "MathAnalysis_Qwen_ProblemSolver", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 10240,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
```

- **"MathAnalysis_Qwen_Classifier"** and **"MathAnalysis_Qwen_Classifier"**: fine-tuned Qwen2.5-Math-7B-bnb-4bit
- **"MathAnalysis_Llama_Classifier"** and **"MathAnalysis_Llama_Classifier"**: fine-tuned Llama-3.2-3B-Instruct

If you want to choose another large language models and fine-tune it, run the code in [**pretraining.ipynb**](https://github.com/ziye2chen/LLMs-for-Mathematical-Analysis/blob/main/pretraining.ipynb).

## Reference

```bibtex
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {http://github.com/unslothai/unsloth},
  year = {2023}
}

@book{demidovich1964problems,
    title={Problems in Mathematical Analysis. Edited by B. Demidovich. Translated From the Russian by G. Yankovsky},
    author={Demidovich, B.P.},
    series={Russian Monographs and Texts on Advanced Mathematics and Physics},
    url={https://books.google.com/books?id=XdmpwgEACAAJ},
    year={1964},
    publisher = {Mir Publishers}  
}

@book{hata2007problems,
  title={Problems and Solutions in Real Analysis},
  author={Hata, M.},
  isbn={9789812776013},
  lccn={2008295629},
  series={Series on number theory and its applications},
  url={https://books.google.com/books?id=vSxkRgQe0AcC},
  year={2007},
  publisher={World Scientific}
}

@misc{dubey2024llama3herdmodels,
      title={The Llama 3 Herd of Models}, 
      author={Meta},
      year={2024},
      eprint={2407.21783},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.21783}, 
}

@misc{yang2024qwen2technicalreport,
      title={Qwen2 Technical Report}, 
      author={Alibaba},
      year={2024},
      eprint={2407.10671},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.10671}, 
}

@techreport{GPT4o,
    author = {OpenAI},
    title = {GPT-4o System Card},
    year = {2024}, 
    url = {https://cdn.openai.com/gpt-4o-system-card.pdf}
}

@techreport{OpenAIo1,
    author = {OpenAI},
    title = {OpenAI o1 System Card},
    year = {2024}, 
    url = {https://assets.ctfassets.net/kftzwdyauwt9/67qJD51Aur3eIc96iOfeOP/71551c3d223cd97e591aa89567306912/o1_system_card.pdf}
}
```

