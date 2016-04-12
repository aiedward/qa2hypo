import numpy as np
import json
import argparse
import os
import random
import re
import string

from helper import *



###############################################################
# beginning of the global variables
###############################################################

# auxiliary verbs, from https://en.wikipedia.org/wiki/Auxiliary_verb
AUX_V = ['am', 'is', 'are', 'can', 'could', 'dare', 'do', 'does', 'did', 'have', 'had', 'may', 'might', 'must', 'need', 'shall', 'should', 'will', 'would']
AUX_V_REGEX = '('+'|'.join(['('+AUX_V[i]+')' for i in range(len(AUX_V))])+')'
AUX_V_BE = ['am', 'is', 'are']
AUX_V_BE_REGEX = '('+'|'.join(['('+AUX_V_BE[i]+')' for i in range(len(AUX_V_BE))])+')'
AUX_V_DOES = ['can', 'could', 'dare', 'does', 'did', 'have', 'had', 'may', 'might', 'must', 'need', 'shall', 'should', 'will', 'would']
AUX_V_DOES_REGEX = '('+'|'.join(['('+AUX_V_DOES[i]+')' for i in range(len(AUX_V_DOES))])+')'
AUX_V_DOESONLY = ['does', 'did']
AUX_V_DOESONLY_REGEX = '('+'|'.join(['('+AUX_V_DOESONLY[i]+')' for i in range(len(AUX_V_DOESONLY))])+')'
AUX_V_DO_REGEX = '(do) '

# question types
QUESTION_TYPES = ['__+', \
'(when '+AUX_V_REGEX+'.*)|(when\?)', \
'(where '+AUX_V_REGEX+'.*)|(where\?)', \
'what ', \
'which ', \
'(whom '+AUX_V_REGEX+'.*)|(who '+AUX_V_REGEX+'.*)|(who\?)|(whom\?)', \
'why ', \
'(how many)|(how much)', \
'(\Ahow [^(many)(much)])|(\W+how [^(many)(much)])', \
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
SAMPLE_TYPE = 50

# used when SAMPLE_TYPE == -1
QUESTION_TYPE = 10

# whether to use the Stanford parser in the transformation
corenlp = True

# whether to print things when running
QUEIT = False

###############################################################
# end of the global variables
###############################################################


# parse the arguments of the program
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir')
    ARGS = parser.parse_args()
    return ARGS

# pre-processing
# using ~/csehomedir/projects/dqa/dqa-data/shining3-vqa for diagram question answering
# using ~/csehomedir/projects/dqa/math-data for math question answering
def pre_proc(args, domain):
    root_dir = args.root_dir
    qa_path = os.path.join(root_dir, 'qa_pairs.json')
    qa_res_path = os.path.join(root_dir, 'qa_res.json')

    print("Loading json files ...")
    qa_pairs = json.load(open(qa_path, 'rb'))

    if domain == 'diagram':
        qa_pairs_list = qa_pairs['qa_pairs']
    elif domain == 'math':
        qa_pairs_list = qa_pairs

    return qa_pairs_list

# post-processing
# using ~/csehomedir/projects/dqa/dqa-data/shining3-vqa for diagram question answering
# using ~/csehomedir/projects/dqa/math-data for math question answering
def post_proc(args, res, domain):
    root_dir = args.root_dir
    qa_path = os.path.join(root_dir, 'qa_pairs.json')
    qa_res_path = os.path.join(root_dir, 'qa_res.json')

    print("Dumping json files ...")
    json.dump(res, open(qa_res_path, 'wb'))

    qa_res_path_2 = os.path.join(root_dir, 'qa_res.txt')
    with open(qa_res_path_2, 'wb') as fw:
        for i in res:
            fw.write('\nquestion: ')
            fw.write((i[Q_ALIAS]).encode('utf-8').strip())
            fw.write('\nanswer: ')
            fw.write(str(i[A_ALIAS]))
            fw.write('\nresult: ')
            fw.write((i[S_ALIAS]).encode('utf-8').strip())
            fw.write('\n-----------------')



# turn qa_pairs into hypotheses, test (can sample questions)
def qa2hypo_test(qa_pairs_list):
    # number of samples and the types of questions to sample
    k = SAMPLE_TYPE
    
    # sampling question type (for examining the result)
    q_type = QUESTION_TYPES[QUESTION_TYPE]
    qa_pairs_list = sample_qa(qa_pairs_list, k, q_type) # set the case lower in the function for questions
    
    # result file
    res = []

    ctr = 0
    for item in qa_pairs_list:
        question = item[Q_ALIAS]
        ans = item[A_ALIAS]

        # sample by question type when k = -1 
        if k != -1 and k != 0:
            q_type = get_question_type(question)

        ###if not re.search('what '+AUX_V_DOESONLY_REGEX, question) and not re.search('what '+AUX_V_DO_REGEX, question):
        sent = rule_based_transform(question, ans, q_type, corenlp, QUEIT)

        res.append({Q_ALIAS:question, A_ALIAS:ans, S_ALIAS:sent})

        ctr += 1
        ###
            
    print(ctr)
    return res
    

# turn qa_pairs into hypotheses (core module)
def qa2hypo(question, answer, corenlp, quiet):
    
    # determine the question type:
    q_type = get_question_type(question)

    # transform the question answer pair into a statement
    sent = rule_based_transform(question, answer, q_type, corenlp, quiet)

    return sent


# determine the question type
def get_question_type(question):
    question = q_norm(question)

    for q_type in QUESTION_TYPES:
        if re.search(q_type, question):
            return q_type
    return 'none of these'

# rule based qa2hypo transformation
def rule_based_transform(question, ans, q_type, corenlp, quiet):
    if not quiet:
        print('Question:', question)
        print('Answer:', ans)

    # question and answer normalization
    question = q_norm(question)
    ans = a_norm(ans)

    # type based transformation
    if q_type == QUESTION_TYPES[0]:
        s, e = test_pattern(q_type, question)
        hypo = replace(question, s, e, ans)
    else:
        if q_type == QUESTION_TYPES[1]:
            s, e = test_pattern('when', question)
            if re.search('when '+AUX_V_DOES_REGEX, question):
                s2, e2 = test_pattern('when '+AUX_V_DOES_REGEX, question)
                hypo = replace(question, s2, e2, '')
                hypo = strip_nonalnum_re(hypo)+' in '+ans
            elif re.search('when '+AUX_V_DO_REGEX, question):
                s3, e3 = test_pattern('when '+AUX_V_DO_REGEX, question)
                hypo = replace(question, s3, e3, '')
                hypo = strip_nonalnum_re(hypo)+' in '+ans
            else:
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[2]:
            s, e = test_pattern('where', question)
            if re.search('where '+AUX_V_DOES_REGEX, question):
                s2, e2 = test_pattern('where '+AUX_V_DOES_REGEX, question)
                hypo = replace(question, s2, e2, '')
                hypo = strip_nonalnum_re(hypo)+' at '+ans
            elif re.search('where '+AUX_V_DO_REGEX, question):
                s3, e3 = test_pattern('where '+AUX_V_DO_REGEX, question)
                hypo = replace(question, s3, e3, '')
                hypo = strip_nonalnum_re(hypo)+' at '+ans
            else:
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[3]:
            if corenlp:
                if re.search('what '+AUX_V_DOESONLY_REGEX, question):
                    s_aux, e_aux, s_vp, e_vp, first_VP=find_np_pos(question, ans, 'what '+AUX_V_DOESONLY_REGEX, node_type='VP', if_root_node=True)
                    hypo = replace(question, e_vp, e_vp, ' '+ans+' ')
                    # print('hypo:', hypo)
                    hypo = replace(hypo, s_aux, e_aux, '')
                    hypo = strip_nonalnum_re(hypo)
                    
                elif re.search('what '+AUX_V_DO_REGEX, question):
                    s_aux, e_aux, s_vp, e_vp, first_VP=find_np_pos(question, ans, 'what '+AUX_V_DO_REGEX, node_type='VP', if_root_node=True)
                    hypo = replace(question, e_vp, e_vp, ' '+ans+' ')
                    # print('hypo:', hypo)
                    hypo = replace(hypo, s_aux, e_aux, '')
                    hypo = strip_nonalnum_re(hypo)
                    
                else:
                    s, e = find_whnp_pos(question)
                    if not s and not e:
                        s, e = test_pattern('what', question)
                    hypo = replace(question, s, e, ans)

            else:
                s, e = test_pattern('what', question)
                hypo = replace(question, s, e, ans)
                hypo = strip_nonalnum_re(hypo)

        elif q_type == QUESTION_TYPES[4]:
            if corenlp:
                s, e = find_whnp_pos(question)
                if not s and not e:
                    s, e = test_pattern('which', question)
                hypo = replace(question, s, e, ans)
            else:
                s, e = test_pattern('which', question)
                hypo = replace(question, s, e, ans)
                hypo = strip_nonalnum_re(hypo)

        elif q_type == QUESTION_TYPES[5]:
            s, e = test_pattern('(who)|(whom)', question)
            hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[6]:
            s, e = test_pattern('why', question)
            hypo = question+', '+ans
            if not re.search('because', ans, re.IGNORECASE):
                hypo = question+', because '+ans

        elif q_type == QUESTION_TYPES[7]:
            if corenlp:
                s, e = find_whnp_pos(question)
                if not s and not e:
                    s, e = test_pattern('(how many)|(how much)', question)
                hypo = replace(question, s, e, ans)

                exp_how_many(question, q_type)
  
            else:
                s, e = test_pattern('(how many)|(how much)', question)
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[8]:
            s, e = test_pattern('(\Ahow )|(\W+how )', question)
            hypo = replace(question, s, e, ' '+ans+' is how ')

        elif q_type == QUESTION_TYPES[9]:
            s, e = test_pattern('(name)|(choose)|(identify)', question)
            hypo = replace(question, s, e, ans+' is')

        # if starting with aux_v, exchange the Verb and Noun
        # if it is an or question, choose the one that heuristically matches the answer
        elif q_type == QUESTION_TYPES[10]:
            if not corenlp:
                if re.search('(yes, )|(no, )', ans):
                    s, e = test_pattern('(yes, )|(no, )', ans)
                    hypo = replace(ans, s, e, '')
                elif ' or ' in question:
                    hypo = ans
                elif re.search('yes\W??', ans):
                    s, e = test_pattern('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
                    hypo = replace(question, s, e, "")
                elif re.search('no\W??', ans):
                    s, e = test_pattern('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
                    hypo = "not "+replace(question, s, e, "")
                else:
                    s, e = test_pattern('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
                    hypo = replace(question, s, e, "")
                    hypo = strip_nonalnum_re(hypo)+' '+ans
            else:
                if re.search('\A(((yes)\W+)|((yes)$))', ans):
                    s_aux, e_aux, s_np, e_np, first_NP = find_np_pos(question, ans, q_type)
                    hypo = replace(question, s_aux, e_np, first_NP + ' ' + question[s_aux:e_aux-1] + ' ')
                
                elif re.search('\A(((no)\W+)|((no)$))', ans):
                    s_aux, e_aux, s_np, e_np, first_NP = find_np_pos(question, ans, q_type)
                    hypo = replace(question, s_aux, e_np, first_NP + ' ' + question[s_aux:e_aux] + 'not ')

                elif re.search(' or ', question):
                    s_aux, e_aux, s_np, e_np, first_NP = find_np_pos(question, ans, q_type)
                    hypo = replace(question, s_aux, e_np, first_NP + ' ' + question[s_aux:e_aux-1] + ' ')
                    s_candidate, e_candidate, candidate = find_or_pos(hypo, ans, q_type)
                    hypo = replace(hypo, s_candidate, e_candidate, candidate)

                else:
                    s, e = test_pattern('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
                    hypo = replace(question, s, e, "")
                    hypo = strip_nonalnum_re(hypo)+' '+ans

        else:
            hypo = strip_nonalnum_re(question)+' '+ans

    if not quiet:
        print('Result:', hypo)
        print("--------------------------------------")

    return hypo

# sample sentences
def sample_qa(qa_pairs_list, k, q_type):
    l = range(len(qa_pairs_list))
    l_sampled = []

    # random sampling
    if k != -1 and k != 0:
        l_sampled = random.sample(l, k)

    # sampling the complementary set
    elif k == 0:
        return sample_qa_complementary(qa_pairs_list)

    # sample by question type (k == -1)
    else:
        for num in l:
            q = qa_pairs_list[num]['question'].lower() # use the lower case for all
            # --- regex ---
            if re.search(q_type, q):
                l_sampled.append(num)

    return [qa_pairs_list[i] for i in l_sampled]

# sample sentences -- the complementary set; this is a helper to sample_qa
def sample_qa_complementary(qa_pairs_list):
    l = range(len(qa_pairs_list))
    l_sampled = []

    for num in l:
        q = qa_pairs_list[num]['question'].lower() # use the lower case for all
        flag = 0
        for q_type in QUESTION_TYPES:
            # --- regex ---
            if re.search(q_type, q) != None:
                flag = 1
                break
        if flag == 0:
            l_sampled.append(num)

    return [qa_pairs_list[i] for i in l_sampled]



if __name__ == "__main__":
    ############################
    # test on single sentence
    ############################

    # question = "what does he refer to when he says that?"
    # answer = "ice"
    # sent = qa2hypo(question, answer)

    ############################
    # test on single sentence
    ############################
    a = get_args()
    qa_pairs_list = pre_proc(a, 'math')
    res = qa2hypo_test(qa_pairs_list)
    post_proc(a, res, 'math')

