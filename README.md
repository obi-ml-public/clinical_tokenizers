# clinical_tokenizers
This repository provides spaCy-compatible tokenizers with custom rules for medical text and word vectors trained on MIMIC-3 notes using fastText.


## Getting started

If necessary, create a conda environment:
```bash
conda env create -f environment.yml
conda activate clinical_tokenizer
```

Install the python package to make use of customized tokenizers for clinical text:
```bash
cd clinical_tokenizers
python setup.py develop
```

Install word vectors trained on MIMIC-3 with fasttext
```bash

# external URL
pip install https://github.com/obi-ml-public/clinical_tokenizers/blob/main/spacy_models/3.3.0/en_mimic_fasttext-0.0.1/dist/en_mimic_fasttext-0.0.1.tar.gz

# or using local file
pip install en_mimic_fasttext-0.0.1.tar.gz 
```

## Usage

```python

# this uses spacy word vectors
# word_vectors='en_core_web_sm'

# this uses fasttext vectors based on mimic
word_vectors='en_mimic_fasttext'
from clinical_tokenizers.clinical_spacy_tokenizer import ClinicalSpacyTokenizer
tokenizer = ClinicalSpacyTokenizer(word_vectors).get_nlp()

note_text1 = """
Physician Discharge Summary    Patient Information   78 yof Hospitalization Summary   Diagnoses: Unstable Angina
Brief Summary/Assessment: 78yo F w/ h/o CAD s/p PCI to LAD presented to the ER with acute onset chest pain. 
She describes the pain as similar pain to the previous one that she experienced before the last PCI. The difference 
is that the pain did not resolve with rest this time. Hospital Course    Hospital Course:  HPI: Patient was in her 
usual state of health until this morning, when she suddenly felt pain in the chest, which was similar to the pain that 
she had before the last PCI. She sat down and rested for 15 min and realized that the pain was not getting better. She 
immediately called 911 and was brought to the ED.   On arrival to the ED, she still had chest pain at the same level. 
ECG showed elevated ST segment. The cath lab was consulted and she was transferred for CAG/PCI.
CAG revealed total occlusion in the RCA #1. PCI was performed and a stent was placed.
Her symptoms disappeared  after the procedure and the patient was in good condition until discharged.
"""

note_text2 = """
Physician Discharge Summary  Patient Information  60 y.o.          Hospitalization Summary   Diagnoses: for PCI (stable)
Brief Summary/Assessment: PCI was performed. RCA #1 75%->0% and a DES (XIENCE 2.5mm x 15 mm) was placed.
Medication: Aspirin 100 mg/day, Clopidogrel 75 mg/day, atorvastatin 80 mg /day
"""

doc1 = tokenizer(note_text1)
features = []
# these are the token ids
for token in doc1:
    vector_id = token.vocab.vectors.find(key=token.orth)
    features.append(vector_id)

# this is the overall similarity based on the word vectors
doc1.similarity(tokenizer(note_text2))

```