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
- [Pretraining & Fine-Tuning](#pretraining--fine-tuning)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

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

Each problem includes a detailed statement and metadata. Data files are available in the [**DEMI-MathAnalysis**](https://people.eecs.berkeley.edu/~hendrycks/MATH.tar).


## Framework and Model

The DEMI-MathAnalysis framework is built to guide LLMs through a structured approach to solving proof-based mathematical problems. It consists of the following key steps:

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


**Model Choices**:  
We have experimented with several LLMs, including both open-source and commercial solutions. Fine-tuning these models on DEMI-MathAnalysis significantly boosts their ability to handle rigor in mathematical analysis. Detailed instructions for model selection, training, and inference are provided in the sections below.

```bash
# Example command to integrate the framework into a training pipeline
python scripts/run_framework.py --model <MODEL_NAME> --data data/train --kb data/knowledge_base.json
```
