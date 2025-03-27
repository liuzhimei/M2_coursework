# M2 Coursework
This is my coursework for the MPhil Data Intensive Science M2 course. 

## Directory Structure
```
M2_coursework/
│ 
├── src/
│   ├── __init__.py
│   ├── preprocessor.py
│   ├── qwen.py
│
├── data/
│   ├── lotka_volterra_data.h5
│
├── report/
│   ├── M2_Coursework_Report.pdf
│
├── pdf/
│   ├── llmtime.pdf
│   ├── main.pdf
│
├── lora/
│   ├── __init__.py
│   ├── lora_skeleton.py
│
├── image/
│
├── README.md
├── pyproject.toml
├── baseline.ipynb
├── lora.ipynb
├── flops.ipynb
├── param_search.ipynb
```


## Installation
To install and run this project, follow these steps:

1. Clone the repository:
```
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m2_coursework/zl474.git
```

2. Navigate to the project directory:
```
cd zl474
```

## Module Overview
`src/`
- `preprocessor.py`: defines functions for preprocessing sequences using LLMTIME scheme and Qwen's tokenization.
- `qwen.py`: defines functions for Qwen.

`data/`: 
- `lotka_volterra_data.h5`: dataset containing 1000 predator-prey systems. 

`report/`:
- `M2_Coursework_Report.pdf`: report for this coursework.

`pdf/`:
- `llmtime.pdf`: a paper explaining LLMTIME scheme.
- `main.pdf`: coursework description.

`lora/`: 
- `lora_skeleton.py`: defines functions to train Qwen with LoRA linear layers.

`image/`: a directory that contains the images I put in the report.

`baseline.ipynb`: Jupyter notebook that runs the untrained Qwen's model.

`lora.ipynb`: Jupyter notebook that trains Qwen with LoRA linear layers.

`flops.ipynb`: Jupyter notebook that calculates FLOPS.

`param_search.ipynb`: Jupyter notebook that performs hyperparameter searches and evaluate the best model.

