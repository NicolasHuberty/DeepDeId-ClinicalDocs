from transformers import Trainer, TrainingArguments, RobertaTokenizerFast,  BertTokenizerFast, BertForTokenClassification

def tokenize_text(text):
    print(text[0])
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    tokenized_inputs = tokenizer(text,max_length=512, padding="max_length", truncation=True, is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt")
    text_inputs = []
    for idx, input_ids in enumerate(tokenized_inputs["input_ids"]):
        text_inputs.append(tokenizer.decode(input_ids, skip_special_tokens=True))
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0])}")
    print(f"Offset mapping: {tokenized_inputs['offset_mapping'][0]}")
    for off in tokenized_inputs['offset_mapping'][0]:
        print(off)
    return tokenized_inputs,text_inputs
