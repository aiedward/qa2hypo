
###############################################################
# beginning of the global variables
###############################################################

# auxiliary verbs, from https://en.wikipedia.org/wiki/Auxiliary_verb
AUX_V = [r'\bam\b', r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b', r'\bbe\b', r'\bbeen\b', r'\bcan\b', r'\bcould\b', r'\bdare\b', r'\bdo\b', r'\bdoes\b', r'\bdid\b', r'\bhave\b', r'\bhad\b', r'\bmay\b', r'\bmight\b', r'\bmust\b', r'\bneed\b', r'\bshall\b', r'\bshould\b', r'\bwill\b', r'\bwould\b']
AUX_V_REGEX = '('+'|'.join(['('+AUX_V[i]+')' for i in range(len(AUX_V))])+')'
AUX_V_BE = [r'\bam\b', r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b', r'\bbe\b', r'\bbeen\b']
AUX_V_BE_REGEX = '('+'|'.join(['('+AUX_V_BE[i]+')' for i in range(len(AUX_V_BE))])+')'
AUX_V_DOES = [r'\bcan\b', r'\bcould\b', r'\bdare\b', r'\bdoes\b', r'\bdid\b', r'\bhave\b', r'\bhad\b', r'\bmay\b', r'\bmight\b', r'\bmust\b', r'\bneed\b', r'\bshall\b', r'\bshould\b', r'\bwill\b', r'\bwould\b']
AUX_V_DOES_REGEX = '('+'|'.join(['('+AUX_V_DOES[i]+')' for i in range(len(AUX_V_DOES))])+')'
AUX_V_DOESONLY = [r'\bdoes\b', r'\bdid\b', r'\bdo\b']
AUX_V_DOESONLY_REGEX = '('+'|'.join(['('+AUX_V_DOESONLY[i]+')' for i in range(len(AUX_V_DOESONLY))])+')'

# question types
QUESTION_TYPES = ['__+', \
'(when '+AUX_V_REGEX+'.*)|(when\?)', \
'(where '+AUX_V_REGEX+'.*)|(where\?)', \
r'\bwhat\b', \
r'\bwhich\b', \
'(whom '+AUX_V_REGEX+'.*)|(who '+AUX_V_REGEX+'.*)|(who\?)|(whom\?)', \
r'\bwhy\b', \
r'\bhow\b', \
# '(how many)|(how much)', \
# '(\Ahow [^(many)(much)])|(\W+how [^(many)(much)])', \
'(name)|(choose)|(identify)', \
'(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )'
]

Q_ALIAS = 'question'
A_ALIAS = 'ans'
S_ALIAS = 'statement'

# SAMPLE_TYPE:
# -1: sample by question type
# 0: sample the complementary set of the listed question types
# not -1 or 0: sample randomly, the value denoting the sample size. QUESTION_TYPE ignored
SAMPLE_TYPE = 80

# used when SAMPLE_TYPE == -1
QUESTION_TYPE = 9

# whether to use the Stanford parser in the transformation
corenlp = True

# whether to print things when running
QUEIT = False

###############################################################
# end of the global variables
###############################################################