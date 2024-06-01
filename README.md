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
 6. [Plot Results](#plot-results)
 7. [Acknowledgements](#acknowledgements)

 ## Installation

 To get started with this project, you need to set up your environment. Follow the steps below:

 1. **Clone the repository**

    ```bash
    git clone --branch evaluation --single-branch https://github.com/NicolasHuberty/DeepDeId-ClinicalDocs.git
    cd DeepDeId-ClinicalDocs
    ```

 2. **Create a virtual environment**

    ```bash
    python -m venv venv
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
For the next commands example, we provided the wikiNER dataset to test all the commands on this dataset.
**Dataset URL**: [wikiner dataset](https://metatext.io/datasets/wikiner)

**Citation**:
Joel Nothman et al., "Learning Multilingual Named Entity Recognition from Wikipedia," 
*Artificial Intelligence*, vol. 194, pp. 151-175, Jan 2013, DOI: [10.1016/j.artint.2012.03.006](https://linkinghub.elsevier.com/retrieve/pii/S0004370212000276).


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
python src/main.py --train_set wikiNER/train.tsv --eval_set wikiNER/test.tsv --epochs 5 --batch_size 4 --mapping None --dataset_size 100 --variant_name mbert
```

 You can customize the training parameters by specifying additional arguments and place your dataset path instead of wikiNER.
 Different models such as roberta,bert or camembert can be used instead of mbert.
 This python command will create a folder in `models_save/mbert-wikiNER_100-mapping_None-epochs_5-batch_size_4` and a result file here: `results/mbert-wikiNER_100-mapping_None-epochs_5-batch_size_4.csv`


 ## Evaluation

 Evaluate the performance of the trained models using the `evaluation.py` script.

```bash
python src/evaluation.py --eval_set wikiNER/test.tsv --batch_size 4 --mapping None --variant_name mbert --model_path mbert-wikiNER_100-mapping_None-epochs_5-batch_size_4
```
 This file will evaluate the performances of an existing model, this evaluation is already done from the main execution.
# Transfer Learning
For transfer learning, you need to define an other argument --transfer_learning_path that redirect to the model folder
    
```bash
python src/main.py --train_set wikiNER/train.tsv --eval_set wikiNER/test.tsv --epochs 5 --batch_size 4 --mapping None --dataset_size 100 --variant_name mbert --transfer_learning_path models_save/mbert-wikiNER_100-mapping_None-epochs_5-batch_size_4
```
This command will create a new folder `models_save/mbert-wikiNER_100-mapping_None-epochs_5-batch_size_4_wikiNER`
# Plot Results
To plot all results, you can use this call function
```bash
python src/plot_performances.py --model_path mbert-wikiNER_100-mapping_None-epochs_5-batch_size_4
```
That provides plots like here

<div style="text-align: center;">
    <img src="results/plots/labelsF1-n2c2.png" width="70%" alt="Labels F1">
</div>
<table>
<tr>
    <td><img src="results/plots/confusionMatrix-roberta-n2c2.png" alt="Confusion Matrix" style="width:80%;"></td>
    <td><img src="results/plots/macro-n2c2.png" alt="Macro Avg F1" style="width:100%;"></td>
</tr>
</table>


# Acknowledgements

 This project was conducted as part of a master's thesis at UCLouvain, 2024. Special thanks to my supervisor, assistant, and the research community for their support and contributions.

