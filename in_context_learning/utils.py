import re


ZERO_SHOT_TRIPLE_EXTRACTION_REGEX = re.compile(r'<triplet>.*?<subj>\s*(.*?)\s*<\/subj>.*?<obj>\s*(.*?)\s*<\/obj>.*?<relation>\s*(.*?)\s*<\/relation>.*?<\/triplet>')


THREE_SHOT_TRIPLE_EXTRACTION_REGEX = re.compile(r'<triplet>(.+)<subj>(.+)<obj>(.+)')


def extract_triples_zero_shot(text):
    matches = re.findall(ZERO_SHOT_TRIPLE_EXTRACTION_REGEX, text, re.DOTALL)
    triples = [(subj.strip(), obj.strip(), relation.strip()) for subj, obj, relation in matches]

    triples = list(set([i for i in triples if len(i) == 3]))
    new_triples = set()
    for triple in triples:
        if len(triple[0]) > 0 and len(triple[1]) > 0 and len(triple[2]) > 0:
            if triple[-1].lower() in ['positive', 'negative']:
                new_triples.add(triple)
            elif 'positive' in triple[-1].lower():
                new_triples.add((triple[0], triple[1], 'positive'))
            elif 'negative' in triple[-1].lower():
                new_triples.add((triple[0], triple[1], 'negative'))
            else:
                pass
        else:
            pass

    return list(new_triples)


def extract_triples_three_shot(text):
    matches = re.findall(THREE_SHOT_TRIPLE_EXTRACTION_REGEX, text)
    triples = set()
    # print(matches)
    for match in matches:
        subj = match[0].strip()  # Extracting subject
        obj = match[1].strip()   # Extracting object
        rel = match[2].strip()   # Extracting relationship
        triples.add((subj, rel, obj))
    
    return list(triples) if triples else [None]


def get_triples_from_zero_shot_model_output(model_output):
    model_output = model_output.split('[INST]')[-1]

    if '[Input' in model_output:
        model_output = model_output.split('[Input')[0]
    
    if '[input' in model_output:
        model_output = model_output.split('[input')[0]
    
    extracted_triples = extract_triples_zero_shot(model_output)

    extracted_triples_json = []
    for item in list(extracted_triples):
        if item:
            extracted_triples_json.append(
                {
                    'src': item[0],
                    'tgt': item[1],
                    'direction': item[2].lower()
                }
            )
    return extracted_triples_json


def get_triples_from_model_three_shot_model_output(model_output):
    model_output = model_output.split('[INST]')[-1]

    if '[Input' in model_output:
        model_output = model_output.split('[Input')[0]
    
    if '[input' in model_output:
        model_output = model_output.split('[input')[0]
    
    extracted_triples = []
    for sentence in model_output.split('\n'):
        sentence = sentence.strip()
        output_triple = extract_triples_three_shot(sentence)
        extracted_triples.extend(output_triple)

    new_triples = set()
    for triple in extracted_triples:
        if triple:
            if len(triple[0]) > 0 and len(triple[1]) > 0 and len(triple[2]) > 0:
                if triple[-1].lower() in ['positive', 'negative']:
                    new_triples.add(triple)
                elif 'positive' in triple[1].lower():
                    new_triples.add((triple[0], 'positive', triple[2]))
                elif 'negative' in triple[1].lower():
                    new_triples.add((triple[0], 'negative', triple[2]))
                else:
                    pass

    extracted_triples_json = []
    for item in list(new_triples):
        if item:
            extracted_triples_json.append(
                {
                    'src': item[0],
                    'tgt': item[2],
                    'direction': item[1]
                }
            )
    return extracted_triples_json
