# qa2hypo
Transfer question and answer pairs into assertive sentences.

## Quick start
1. Download and install the nltk package that is used in the project to parse parse-trees.
2. Download the Nodebox English Linguistics library and put it in the root directory of the project.
3. Run ```python ./stanford-corenlp-python/corenlp.py``` for using the stanford corenlp package for constituency parsing.
4. Call ```qa2hypo(question, answer, corenlp)``` in ```qa2hypo.py``` to get the transformation in ```str``` structure. To optimize for speed in compensation for performance, set argument ```corenlp``` as False, and accordingly there is no need for step 2.
5. This module is part of DQA-net. To work with [DQA-net](https://github.com/seominjoon/dqa-net), tensorflow installation is required. Common issues include an outdated version of libstdc++6, and the solutions can be found [here](http://stackoverflow.com/questions/16605623/where-can-i-get-a-copy-of-the-file-libstdc-so-6-0-15) and [here](http://askubuntu.com/questions/575505/glibcxx-3-4-20-not-found-how-to-fix-this-error).
6. This module also works with [Math Question Answering](https://gitlab.cs.washington.edu/amini91/MultiEQProbs), which is under heavy development, aiming at automating algebraic question answering.
