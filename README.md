# qa2hypo
Transfer question and answer pairs into assertive sentences.

## Quick start
1. Download and install the nltk package that is used in the project to parse parse-trees.
2. Download the Nodebox English Linguistics library and put it in the root directory of the project.
3. Run ```python ./stanford-corenlp-python/corenlp.py``` for using the stanford corenlp package for constituency parsing.
4. Call ```qa2hypo(question, answer, corenlp)``` in ```qa2hypo.py``` to get the transformation in ```str``` structure. To optimize for speed in compensation for performance, set argument ```corenlp``` as False, and accordingly there is no need for step 2.
