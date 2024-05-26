 # De-Identification of Clinical Documents Using Deep Learning
 ### Master's Thesis at UCLouvain, 2024

 ## Project Overview

 This repository contains the work of a master's thesis conducted at UCLouvain in 2024, focusing on the de-identification of clinical documents through the application of deep learning techniques. This project developed a robust system capable of automatically detecting protected health information (PHI) within clinical documents, thus preserving patient confidentiality while enabling the utilization of clinical data for research purposes.

[![ILA Demo](https://i9.ytimg.com/vi/PJTIBT_-VHk/mqdefault.jpg?sqp=CKT4zbIG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGGUgZShlMA8=&rs=AOn4CLD_0_xNbOkYX6pn5jQPYzVxmiVzPg)](https://youtu.be/PJTIBT_-VHk "Demo of the ILA")


 ## Table of Contents

 1. [Installation](#installation)
 2. [Dataset Preparation](#dataset-preparation)
 3. [Usage](#usage)
     - [Using the Manager](#using-the-manager)
     - [Manual Execution](#manual-execution)
 4. [Evaluation](#evaluation)
 5. [Plot Results](#plot-results)
 6. [Acknowledgements](#acknowledgements)

 ## Installation

 To get started with this project, you need to set up your environment. Follow the steps below:

 1. **Clone the repository**
     ```bash
     git clone https://github.com/NicolasHuberty/DeepDeId-ClinicalDocs
     cd DeepDeId-ClinicalDocs
     ```

 2. **Create a virtual environment**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

 3. **Install the required packages**
     ```bash
     pip install -r requirements.txt
     ```

 ## Dataset Preparation
### Supported Input Formats
- **CoNLL-U**: A format used for annotated datasets in NLP tasks, particularly for named entity recognition.
- **XML**: An XML-based format for annotated clinical documents.
- **BRAT**: A format originating from the BRAT annotation tool, used for named entity recognition tasks.
- **TSV**: Tab-separated values format, which aligns each word with its corresponding label.

 The datasets used in this project should be placed in the `datasets/formatted` directory. Ensure that the datasets are properly formatted according to the required structure, then adapt all the fields to redirect to your unformatted dataset in dataset/formatDataset.py file.
 1. **Format the datasets**
    ```bash
    python formatDataset.py
    ```

 ## Usage

 ### Using the Manager

 To simplify the operation, use the project management script:

 1. **Create the project and automatically manage files**
    ```bash
    python src/create_project.py --project_name wikiNER --dataset dataset/formatted/wikiNER/train.tsv --labels PERSON LOCATION DATE ID --model_name roberta --eval_percentage 30 --training_steps 10 --num_predictions 50 --start_from 20
    ```
This creation of project will create a new folder in projects/ containing the configuration file, the database and the potential supplementary evaluation set.

 2. **Simulate project operation**
    ```bash
    python simulation.py --steps 10 --to 500
    ```
This code will simulate the annotation process of 500 documents with a step size of 10.
All performances of the model are present in the folder of the project.
 ### Manual Execution

 All functions of the project can be called separately without using the manager.

 1. **Create project structure**
    ```bash
    python src/create_project.py --project_name wikiNER --dataset dataset/formatted/wikiNER/train.tsv --labels PERSON LOCATION DATE ID --model_name roberta --eval_percentage 30 --training_steps 10 --num_predictions 50 --start_from 20
    ```

 2. **Simulate manual annotations**
    ```bash
    python src/fake_manual_annotation.py --project_name wikiNER --num_records 10
    ```

    This code will simulate the annotation of 10 records. The use of this function is possible only if the dataset is provided with existing labels, otherwise, only the human annotation from the webUI or directly from SQLite is possible.

 3. **Train the model**
    ```bash
    python train.py --project_name wikiNER
    ```

    This function will launch the training of the model.
    This code will automatically discover all available documents for the model.

 4. **Run predictions**
    ```bash
    python prediction.py --project_name wikiNER --num_records 100
    ```

    This code will predict the next 100 documents of the project.

 5. **Evaluate model performance**
    ```bash
    python evaluation.py
    ```

 ## Evaluation

 Evaluate the performance of the trained models using the `evaluation.py` script.
    ```bash
    python evaluation.py --project_name wikiNER --supplementary_dataset dataset/formatted/wikiNER/test.tsv
    ```

This file will evaluate the model using the supplementary dataset instead of using the records dedicated for evaluation. If the supplementary_dataset need to systematically used, it is needed to manually integrate it from the config.json file, otherwise it can be added from the webUI.

If no annotated files exist this code will retrieve all evaluation records coming from the annotation process.

```bash
python evaluation.py --project_name wikiNER
```
 ## Plot Results

 To plot all results, use the `plotResults.py` script.

```bash
python plotResults.py
```
It is manual, it is needed to manually adapt the path of the results to use.

 ## Acknowledgements

 This project was conducted as part of a master's thesis at UCLouvain, 2024. Special thanks to my supervisor, assistant, and the research community for their support and contributions.
