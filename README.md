## Folder Structure
### Download the data as shown in the Folder Structure below:

``` bash
.
│── analysis/ # Analysis scripts for data exploration and visualization
│── assets/ # Assets such as images, pdfs, etc.
├── reports/ # reports to summarize findings
├── papers/ # Research papers related to the project
├── data/
│   └── mine_impact_data_2019.mat
│   └── phase2_data_20220215.mat
├── data_classes/ # Data classes in this repository prepare data and extract features from the dataset
│   └── decomposition.py           
├── models/
│   └── models.py # Model defintions
│   └── loops.py  # Training and evaluation loops
│   └── classification.py  # Classification model definitions
├── evaluation/ # Evaluation scripts for assessing model performance
├── requirements.txt      # List of Python dependencies
└── README.md             # Project documentation
```
### Python Setup
### Setup Python3 virtual enviroment
``` bash
python3 -m venv venv
```
### Install dependencies
``` bash
pip install -r requirements.txt
```