# README

This repository contains the code and data for our anonymous submission.

## Table of Contents
- [Kowledge Environment Construct](#Knowledge-Environment-Construct)
- [Behavioral Analysis](#Behavioral-Analysis)
- [CoT Editing](#CoT-Editing)
- [Data](#Data)



## knowledge-Environment-Construct

Knowledge environment constructing scripts are in `1_exp_knowledge_environment_construct`

- **With Misleading Information:**  
  - `init_rfc_database-chroma.py`: Initialize the RFC database
  - `pipeline_facuallyIncorrect.py`: Pipeline with *Type Ⅱ* data


- **Without Misleading Information:**  
  - `hallucination/pipeline_factuallyCorrect_Hallu.py`: Pipeline with *Type Ⅰ* data
  - `hallucination/rfc_index.json`: RFC index file in json format
  - `no_halllucination/pipeline_factuallyCorrect_withoutHallu`: Experiment pipeline with *Type Ⅱ* data


## Behavioral-Analysis

 Hallucination behavioral analysis scripts are in `2_exp_dataset_analyze/`
  - `data_correct.py`: Analyze *Type Ⅰ* data
  - `data_incorrect.py`: Analyze *Type Ⅱ* data


## CoT-Editing
Chain-of-Thought (CoT) editing scripts are in `3_exp_cot_editing/`
  - `edit_cot.py`: CoT editing script
  - `think_template.jinjia`: COT editing template


## Data
Click [here](https://drive.google.com/file/d/1KZDBerCz05VKk-_q1r4mbvL5qnughgtG/view?usp=drive_link) to get datset.