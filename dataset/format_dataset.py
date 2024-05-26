import os
import xml.etree.ElementTree as ET
import re

def save_formatted(filePath,formattedLines):
    print(f"Save dataset format in a new file...")
    filePath = filePath.replace("unformatted","formatted")
    outputDir = os.path.dirname(filePath)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    with open(filePath,"w",encoding="utf-8") as file:
        for line in formattedLines:
            file.write(line)

def format_conllu_dataset(filePath, save = False):
    with open(filePath,"r",encoding="utf-8") as file:
        formattedLines = []
        for line in file:
            if line.strip():
                parts = line.strip().split('\t')
                token = parts[1]
                label = parts[2]
                formattedLines.append(f"{token}\t{label}\n")
            else:
                formattedLines.append(f"\n")
    if save:
        save_formatted(filePath, formattedLines)


def format_xml_dataset(directory_path, save=False):
    formattedLines = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory_path, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            text_element = root.find('TEXT')
            clinical_text = text_element.text if text_element is not None else ""
            annotations = []
            tags_element = root.find('TAGS')
            if tags_element is not None:
                for tag in tags_element:
                    start = int(tag.attrib['start'])
                    end = int(tag.attrib['end'])
                    tag_type = tag.attrib['TYPE']
                    annotations.append((start, end, tag_type))
            annotations.sort(key=lambda x: x[0])
            tokens = [(m.start(), m.end()) for m in re.finditer(r'\S+', clinical_text)]
            labels = ['O'] * len(tokens)
            current_entity_type = None            
            for i, (token_start, token_end) in enumerate(tokens):
                for start, end, tag_type in annotations:
                    if start <= token_start < end:
                        if current_entity_type != tag_type: 
                            labels[i] = "B-" + tag_type
                            current_entity_type = tag_type
                        else: 
                            labels[i] = "I-" + tag_type
                        break
                else:
                    current_entity_type = None
            for (token_start, token_end), label in zip(tokens, labels):
                token_text = clinical_text[token_start:token_end]
                formattedLines.append(f"{token_text}\t{label}\n")
        formattedLines.append(f"\n")
    if save:
        saveFile = directory_path.replace("unformatted", "formatted") + ".tsv"
        save_formatted(saveFile,formattedLines)

def format_xml_merged_dataset(filename, save=False):
    formattedLines = []
    if filename.endswith(".xml"):
        tree = ET.parse(filename)
        root = tree.getroot()

        for record in root.findall('RECORD'):
            full_text = []
            annotations = []
            for element in list(record.find('TEXT')):
                if element.tail:
                    full_text.append(element.tail)
                if element.tag == 'PHI':
                    start = sum(len(t) for t in full_text)
                    full_text.append(element.text)
                    end = start + len(element.text)
                    tag_type = element.attrib['TYPE']
                    annotations.append((start, end, tag_type))
            full_text = ''.join(full_text)
            annotations.sort(key=lambda x: x[0])
            tokens = [(m.start(), m.end()) for m in re.finditer(r'\S+', full_text)]
            labels = ['O'] * len(tokens)
            for i, (token_start, token_end) in enumerate(tokens):
                for start, end, tag_type in annotations:
                    if start <= token_start < end:
                        prefix = "B-" if token_start == start else "I-"
                        labels[i] = prefix + tag_type
                        break
            for (token_start, token_end), label in zip(tokens, labels):
                token_text = full_text[token_start:token_end]
                formattedLines.append(f"{token_text}\t{label}\n")
            formattedLines.append("\n") 
    if save:
        saveFile = filename.split(".xml")[0] + ".tsv"
        save_formatted(saveFile,formattedLines)

def format_brat_dataset(directory_path, save=False):
    import os

    def saveFile(path, lines):
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    all_formatted_lines = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            base_filename = filename[:-4]
            txt_path = os.path.join(directory_path, f"{base_filename}.txt")
            ann_path = os.path.join(directory_path, f"{base_filename}.ann")

            if os.path.exists(ann_path):
                formattedLines = []
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    text_content = txt_file.read()

                tokens = []
                token_spans = []
                current_pos = 0
                for word in text_content.split():
                    start_pos = text_content.find(word, current_pos)
                    end_pos = start_pos + len(word)
                    tokens.append(word)
                    token_spans.append((start_pos, end_pos))
                    current_pos = end_pos

                labels = ['O'] * len(tokens)
                annotations = []

                with open(ann_path, 'r', encoding='utf-8') as ann_file:
                    for line in ann_file:
                        if line.startswith('T'):
                            parts = line.split('\t')
                            ann_info = parts[1].split(' ')
                            label = ann_info[0]
                            span_info = ann_info[1:]

                            spans = []
                            for i, span in enumerate(span_info):
                                if i == 0:
                                    start_span = int(span)
                                elif ';' in span:
                                    for part in span.split(';'):
                                        spans.append((start_span, int(part)))
                                        if i < len(span_info) - 1:
                                            start_span = int(span_info[i + 1])
                                else:
                                    spans.append((start_span, int(span)))

                            annotations.append((spans, label))

                for i, (token_start, token_end) in enumerate(token_spans):
                    for spans, ann_label in annotations:
                        for ann_start, ann_end in spans:
                            if not (token_end <= ann_start or token_start >= ann_end):
                                labels[i] = ann_label
                                break

                for token, label in zip(tokens, labels):
                    formattedLines.append(f"{token}\t{label}\n")
                formattedLines.append("\n")

                all_formatted_lines.extend(formattedLines)
    if save and all_formatted_lines:
        saveFile = directory_path.replace("unformatted", "formatted") + ".tsv"
        save_formatted(saveFile, all_formatted_lines)


# This function allows to map all labels of a dataset an save this new mapping as a new tsv file
def map_labels_and_save(input_file_path, output_file_path, sep=' '):
    # Label Mapping for your new file
    label_mapping = {
        'O': 'O',
        'B-PER': 'PER',
        'I-PER': 'PER',
    }
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
        previous_line_was_empty = False
        
        for line in input_file:
            if not line.strip():
                if not previous_line_was_empty: 
                    output_file.write("\n")
                previous_line_was_empty = True 
                continue
            else:
                previous_line_was_empty = False 
            parts = line.strip().split(sep, 1) 
            if len(parts) == 2:
                record, label = parts
                mapped_label = label_mapping.get(label, label) if label_mapping else label
                output_file.write(f"{record}{sep}{mapped_label}\n")
            else:
                output_file.write(line)


if __name__ == "__main__":
    #format_conllu_dataset("./datasets/unformatted/multiNERD/train_en.tsv", save=True)
    #format_n2c2_2014_dataset("./datasets/unformatted/meddocan/test",save=True)
    #format_n2c2_dataset("./dataset/unformatted/n2c2",save=True)
    #format_brat_dataset("./datasets/unformatted/QUAERO_FrenchMed/test",save=True)
    input_file_path = 'dataset/formatted/multiNERD/train_en.tsv'
    output_file_path = 'dataset/formatted/multiNERD/train_enMap.tsv'
    map_labels_and_save(input_file_path,output_file_path,sep='\t')
