# This file is part of DeepDeId-ClinicalDocs project and is released under the GNU General Public License v3.0.
# See "LICENSE" for more information or visit https://www.gnu.org/licenses/gpl-3.0.html.
from transformers import AutoModelForTokenClassification, AutoTokenizer, RobertaTokenizerFast
from models import RobertaCustomForTokenClassification
from utils import load_config_field
import json


def load_model_and_tokenizer(project_name):
    # This function will return the model and tokenizer depending on the model name
    base_model = load_config_field(project_name,"modelName")
    model_path = load_config_field(project_name,"model_path")
    tokenizer_path = load_config_field(project_name,"tokenizer_path")
    labels = load_config_field(project_name,"labels")
    num_labels = len(labels)
    base_model_map = {
        "mbert": "bert-base-multilingual-uncased",
        "roberta": "Jean-Baptiste/roberta-large-ner-english",
        "bert": "bert-large-uncased",
        "beto": "dccuchile/bert-base-spanish-wwm-cased",
        "xlm-roberta": "xlm-roberta-large",
    }
    
    base_model_name = base_model_map.get(base_model, "bert-base-multilingual-uncased")
    if(base_model == "roberta"):
        if(model_path):
            model = RobertaCustomForTokenClassification.from_pretrained(model_path)
            tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        else:
            model = RobertaCustomForTokenClassification(num_labels=len(labels))
            tokenizer = RobertaTokenizerFast.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    elif model_path:
        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        model = AutoModelForTokenClassification.from_pretrained(base_model_name, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer
