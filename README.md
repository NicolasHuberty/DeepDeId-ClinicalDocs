 # De-Identification of Clinical Documents Using Deep Learning

 ### Master's Thesis at UCLouvain, 2024

 ## Project Overview

 This repository contains the work of a master's thesis conducted at UCLouvain in 2024, focusing on the de-identification of clinical documents through the application of deep learning techniques. This project developped a robust system capable of automatically detecting protected health information (PHI) within clinical documents, thus preserving patient confidentiality while enabling the utilization of clinical data for research purposes.

 ## Table of Contents

 1. [Installation](#installation)
 2. [Dataset Preparation](#dataset-preparation)
 3. [Training](#training)
 4. [Evaluation](#evaluation)
 5. [Transfer Learning](#transfer-learning)
 6. [Acknowledgements](#acknowledgements)

 ## Installation

 To get started with this project, you need to set up your environment. Follow the steps below:

 1. **Clone the repository**

    ```bash
    git clone https://github.com/NicolasHuberty/DeepDeId-ClinicalDocs/tree/evaluation
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

 The datasets used in this project should be placed in the `datasets/formatted` directory. Ensure that the datasets are properly formatted according to the required structure.

 1. **Format the datasets**
### Supported Input Formats
- **CoNLL-U**: A format used for annotated datasets in NLP tasks, particularly for named entity recognition.
- **XML**: An XML-based format for annotated clinical documents.
- **BRAT**: A format originating from the BRAT annotation tool, used for named entity recognition tasks.
- **TSV**: Tab-separated values format, which aligns each word with its corresponding label.

 Use the provided `formatDataset.py` script to convert datasets into the required format.

```bash
python datasets/formatDataset.py --input_path path/to/raw_datasets --output_path datasets/formatted
```
The dataset statistics can be explored using this file:

```bash
python src/dataset_statistics.py --datasets_path wikiNER/train.tsv wikiNER/test.tsv
```
This command should give plots, all supports and statistics.

2. **Define Mappings**

If the labels definition is not what you expected, you can map your own labels and place the file in the mapping folder with name like name_mapping.json
Here is an example of correct JSON mapping file:
```json
{
    "I-PER": "PERSON",
    "B-PER": "PERSON",
    "I-LOC": "LOCATION",
    "B-LOC": "LOCATION",
    "B-MISC": "MISC",
    "I-MISC": "MISC",
    "B-ORG": "ORG",
    "I-ORG": "ORG"
}
```
 ## Training

 To train and evaluate the deep learning models for de-identification, follow these steps:

```bash
python src/main.py --train_set wikiNER/train.tsv --eval_set wikiNER/test.tsv --epochs 5 --batch_size 4 --mapping None --dataset_size -1 --variant_name roberta
```

 You can customize the training parameters by specifying additional arguments and place your dataset path instead of wikiNER.
 
 This python command will create a folder 
```bash
 models_save/roberta-wikiNER_-1-mapping_None-epochs_5-batch_size_4
```


 ## Evaluation

 Evaluate the performance of the trained models using the `evaluation.py` script.

```bash
python evaluation.py --eval_set wikiNER/test.tsv --batch_size 4 --mapping None --variant_name roberta --model_path roberta-wikiNER_-1-mapping_None-epochs_5-batch_size_4
```

# Transfer Learning
For transfer learning, you need to define an other argument --transfer_learning_path that redirect to the model folder
    
```bash
python src/main.py --train_set wikiNER/train.tsv --eval_set wikiNER/test.tsv --epochs 5 --batch_size 4 --mapping None --dataset_size -1 --variant_name roberta --transfer_learning_path roberta-i2b2_2006_-1-mapping_None-epochs_5-batch_size_4
```

 ## Acknowledgements

 This project was conducted as part of a master's thesis at UCLouvain, 2024. Special thanks to my supervisor, assistant, and the research community for their support and contributions.
