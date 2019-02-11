# nlu_datasets
Datasets for intent classification and entity extraction including converters.

**Note that all data has been decapitalized** 

## Data
Data is not split in train / test, train / dev / test to allow for using k-fold cross 
validation and allow users to define their own splits. It is advised to use fixed random 
seed for code reproducibility. The `NLU-evaluation-corpora` define a fixed train test split.
This is stored in these files using the `training` column. Note that the splits are odd.
In `AskUbuntuCorpus.json` 53 sentences have `"training": true` while 109 sentences have 
`"training" : false`. In other words, the test set is twice as large as training set. The
only reason to use this predefined split is to compare accuracy against 
NLU-evaluation-corpora paper.

## Annotation standard
The datasets are in Rasa NLU Markdown training format and not in BIO or BIO2 annotation standard. 
Reason for this is to improve readability of datasets for humans. Code to convert
from annotated sentence to a more convenient representation is provided by the Rasa NLU 
project (available as Python PIP package). The Rasa format makes it more convenient to
store sentence information (for example, intent of sentence) in the same file.

For comparison the following sentence is annotated using the BIO2 standard.
```
Stanford University located at California .
B-ORG    I-ORG      O       O  B-LOC      O
```

Typically this information will be stored in a sentence and annotation file. Having one 
respectively token or annotation per line. 

The same sentence in Rasa NLU training format is:
```
[Stanford University](ORG) located at [California](LOC)
```
This can be stored in one file having one line per sentence. Note that sentences 
containing round or square brackets will cause problems. This can be
solved by either removing them from the sentences before processing or escaping the 
characters. The former is not expected to affect the classification performance 
significantly. The latter seems like the best approach. 

Alternative NER annotation standards exist.  
[https://arxiv.org/pdf/1511.08308.pdf](BIOES Chui Nichols, 2016)
[https://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/](BMEWO, BMEWO+)
These seem aimed at performance based on the 2009 post. This is no issue for us since
the classifier will probably use more computing power.

Another representation from `CoNLL-2003 v2` dataset:
```
SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O
```
