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
SAMPLE_TYPE = 50

# used when SAMPLE_TYPE == -1
QUESTION_TYPE = 9

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
        s, e = find_regex(q_type, question)
        hypo = replace(question, s, e, ans)
    else:
        if q_type == QUESTION_TYPES[1]:
            s, e = find_regex('when', question)
            if re.search('when '+AUX_V_DOES_REGEX, question):
                s2, e2 = find_regex('when '+AUX_V_DOES_REGEX, question)
                hypo = replace(question, s2, e2, '')
                hypo = strip_nonalnum_re(hypo)+' in '+ans
            # elif re.search('when '+AUX_V_DO_REGEX, question):
            #     s3, e3 = find_regex('when '+AUX_V_DO_REGEX, question)
            #     hypo = replace(question, s3, e3, '')
            #     hypo = strip_nonalnum_re(hypo)+' in '+ans
            else:
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[2]:
            s, e = find_regex('where', question)
            if re.search('where '+AUX_V_DOES_REGEX, question):
                s2, e2 = find_regex('where '+AUX_V_DOES_REGEX, question)
                hypo = replace(question, s2, e2, '')
                hypo = strip_nonalnum_re(hypo)+' at '+ans
            # elif re.search('where '+AUX_V_DO_REGEX, question):
            #     s3, e3 = find_regex('where '+AUX_V_DO_REGEX, question)
            #     hypo = replace(question, s3, e3, '')
            #     hypo = strip_nonalnum_re(hypo)+' at '+ans
            else:
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[3]:
            if corenlp:
                # do
                if re.search('what '+AUX_V_DOESONLY_REGEX, question):
                    s_aux, e_aux, s_vp, e_vp, first_VP=find_np_pos(question, ans, 'what '+AUX_V_DOESONLY_REGEX, node_type='VP', if_root_node=True)
                    hypo = replace(question, e_vp, e_vp, ' '+ans+' ')
                    # print('hypo:', hypo)
                    hypo = replace(hypo, s_aux, e_aux, '')
                    hypo = strip_nonalnum_re(hypo)
                # # does
                # elif re.search('what '+AUX_V_DO_REGEX, question):
                #     s_aux, e_aux, s_vp, e_vp, first_VP=find_np_pos(question, ans, 'what '+AUX_V_DO_REGEX, node_type='VP', if_root_node=True)
                #     hypo = replace(question, e_vp, e_vp, ' '+ans+' ')
                #     # print('hypo:', hypo)
                #     hypo = replace(hypo, s_aux, e_aux, '')
                #     hypo = strip_nonalnum_re(hypo)
                # be
                else:
                    s, e = find_type_position(question, 'WHNP')
                    if not s and not e:
                        s, e = find_regex('what', question)
                    hypo = replace(question, s, e, ans)

            else:
                s, e = find_regex('what', question)
                hypo = replace(question, s, e, ans)
                hypo = strip_nonalnum_re(hypo)

        elif q_type == QUESTION_TYPES[4]:
            if corenlp:
                s, e = find_type_position(question, 'WHNP')
                if not s and not e:
                    s, e = find_regex('which', question)
                hypo = replace(question, s, e, ans)
            else:
                s, e = find_regex('which', question)
                hypo = replace(question, s, e, ans)
                hypo = strip_nonalnum_re(hypo)

        elif q_type == QUESTION_TYPES[5]:
            s, e = find_regex('(who)|(whom)', question)
            hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[6]:
            s, e = find_regex('why', question)
            hypo = question+', '+ans
            if not re.search('because', ans, re.IGNORECASE):
                hypo = question+', because '+ans

        # how
        elif q_type == QUESTION_TYPES[7]:
            s, e = find_regex(q_type, question)

            question_head = question[:s]
            question_rear = question[s:]

            # find 'how'
            s_how, e_how = find_regex(q_type, question_rear)

            # find the word adjacent to 'how'
            how_next = ((question_rear[e_how:]).strip().split(' '))[0]
            # print 'how_next: ', how_next

            if corenlp:
                # how [aux_v] question
                if re.search(AUX_V_REGEX, how_next):
                    hypo = replace(question_rear, s_how, e_how, ' '+ans+' is how ')
                    hypo = question_head + hypo
                    hypo = strip_nonalnum_re(hypo)
                # how [adj]
                else:
                    # find the position of WHADJP
                    s_whadjp, e_whadjp = find_type_position(question_rear, 'WHADJP')

                    # find the position of WHNP
                    s_whnp, e_whnp = find_type_position(question_rear, 'WHNP')
                    # print 'whnp: ', question_rear[s_whnp:e_whnp]

                    s_wh = s_whadjp
                    if s_whadjp == None:
                        s_wh = s_whnp

                    e_wh = e_whadjp
                    if e_whadjp == None:
                        e_wh = e_whnp

                    # find the first auxiliary verb
                    s_aux, e_aux = find_regex(AUX_V_REGEX, question_rear)
                    aux_be = question_rear[s_aux:e_aux]

                    # an auxiliary verb immediately follows a [whnp]
                    ###### e.g., how many apples are on the table ######
                    ###### e.g., how many apples are those that Joe bought ######
                    ###### e.g., how many apples will be served ######
                    if (e_whnp != None) and (s_aux - e_whnp <= 2):
                        hypo = replace(question_rear, s_wh, e_wh, ans)
                        hypo = question_head + hypo
                        hypo = strip_nonalnum_re(hypo)
                    ###### e.g., how complicated is the problem ######
                    ###### e.g., how much money do you have ######
                    ###### e.g., how much money am I going to earn ######
                    else:
                        # find [np] right after [aux_v]
                        question_trimmed = question_rear[e_aux:]
                        s_np, e_np = find_type_position(question_trimmed, 'NP')
                        if e_np != None:
                            s_np += e_aux
                            e_np += e_aux
                        s_vp, e_vp = find_type_position(question_trimmed, 'VP')
                        if e_vp != None:
                            s_vp += e_aux
                            e_np += e_aux

                        # find the existence of [does]
                        s_does, e_does = find_regex(AUX_V_DOESONLY_REGEX, question_rear[s_aux:e_aux])

                        # [will] as the [aux_v], no need for changing the tense
                        if s_does == e_does:
                            # [be]
                            if e_vp == None:
                                hypo = replace(question_rear, e_np, e_np, " "+aux_be+" "+ans+" "+question_rear[s_wh+3:s_aux])
                            # [will]
                            else:
                                hypo = replace(question_rear, e_vp, e_vp, " "+ans+" "+question_rear[s_wh+3:s_aux])
                                hypo = replace(hypo, e_np, e_np, " "+aux_be+" ")

                        # [does] as the [aux_v], need for changing the tense
                        else:
                            hypo = replace(question_rear, e_vp, e_vp, " "+ans+" "+question_rear[s_wh+3:s_aux])
                            v_old = question_rear[s_vp:e_vp]
                            v_new = v_transform(v_old, question_rear[s_np:e_np], aux_be)
                            hypo = replace(hypo, s_vp, e_vp, v_new)

                        hypo = hypo[s_np:]
                        hypo = question_head + hypo
                        hypo = strip_nonalnum_re(hypo)




                    

                    e_wh = find_min([e_whadjp, e_whadvp, e_whnp])

                    
                    # find if the neighboring node is a noun
                    aux_node = find_node_by_word(question_rear, s_aux, e_aux)

                    # the node right to the auxiliary verb
                    aux_right = aux_node.right_sibling()




                    # be
                    if s_aux_be != e_aux_be:

                        

                        # find the position of the first auxiliary verb
                        s_aux_any, e_aux_any = find_regex(AUX_V_REGEX, question_rear)
                        # print 'aux: ', question_rear[s_aux_any:e_aux_any]

                        # find the position of the first noun in NP form
                        s_np, e_np = find_type_position(question_rear, 'NP')
                        # print 'np: ', question_rear[s_np:e_np]

                        # noun follows [how many]
                        ###### e.g., how many apples are in the fridge ######
                        if s_whnp != e_whnp:
                            if s_whadjp!=None:
                                hypo = replace(question_rear, s_whadjp, e_whadjp, ans)
                            else:
                                hypo = replace(question_rear, s_whadvp, e_whadvp, ans)
                            hypo = question_head + hypo

                        # no noun follows [how many]
                        else:
                            # find first [vp]
                            s_vp, e_vp = find_type_position(question_rear, 'VP')

                            be = question_rear[s_aux_any:e_aux_any]

                            ###### e.g., how big is the apple (no vp) ######
                            if e_vp == None:
                                if s_whadjp!= None:
                                    hypo = replace(question_rear, e_np, e_np, ' '+be+' '+ans+' '+question_rear[s_whadjp+3:s_aux_any])
                                else:
                                    hypo = replace(question_rear, e_np, e_np, ' '+be+' '+ans+' '+question_rear[s_whadvp+3:s_aux_any])
                            ###### e.g., how far did he go (vp) ######
                            else:
                                if s_whadjp!= None:
                                    hypo = replace(question_rear, e_vp, e_vp, ' '+ans+' '+question_rear[s_whadjp+3:s_aux_any])
                                    # print question_rear[s_whadjp+3:s_aux_any]
                                else:
                                    hypo = replace(question_rear, e_vp, e_vp, ' '+ans+' '+question_rear[s_whadvp+3:s_aux_any])
                                # put [will] after [np]
                                hypo = replace(hypo, e_np, e_np, ' '+be+' ')

                            hypo = hypo[s_np:]
                            hypo = question_head + hypo
                            hypo = strip_nonalnum_re(hypo)

                    # do
                    else:
                        # find AUX_V_DOESONLY_REGEX
                        s_aux_do, e_aux_do = find_regex(AUX_V_DOESONLY_REGEX, question_rear)
                        # non-do
                        ###### e.g., how many apples will he eat ######
                        if s_aux_do == e_aux_do:
                            # find AUX_V_DOES_REGEX
                            s_aux, e_aux = find_regex(AUX_V_DOES_REGEX, question_rear)
                            # find the first verb
                            s_0, e_0, s_vp, e_vp, first_VP=find_np_pos(question_rear, ans, AUX_V_DOES_REGEX, node_type='VP', if_root_node=True)
                            question_np = question_rear[e_whadjp:s_aux]
                            hypo = replace(question_rear, e_vp, e_vp, ' '+ans+' '+question_np+' ')
                            hypo = replace(hypo, s_vp, s_vp, ' '+question_rear[s_aux:e_aux]+' ')
                            hypo = hypo[e_aux:]
                            hypo = question_head + hypo
                            hypo = strip_nonalnum_re(hypo)

                        # do
                        ###### e.g., how many apples did he eat ######
                        else:
                            # find the first verb
                            s_0, e_0, s_vp, e_vp, first_VP=find_np_pos(question_rear, ans, AUX_V_DOES_REGEX, node_type='VP', if_root_node=True)
                            question_np = question_rear[e_whadjp:s_aux_do]
                            # print "question_np: ", question_np
                            
                            hypo = replace(question_rear, e_vp, e_vp, ' '+ans+' '+question_np+' ')
                            hypo = hypo[e_aux_do:]
                            # print('hypo:', hypo)
                            hypo = question_head + hypo
                            hypo = strip_nonalnum_re(hypo)
            else:
                hypo = replace(question, s, e, ' '+ans+' is how ')


        elif q_type == QUESTION_TYPES[8]:
            s, e = find_regex('(name)|(choose)|(identify)', question)
            hypo = replace(question, s, e, ans+' is')

        # if starting with aux_v, exchange the Verb and Noun
        # if it is an or question, choose the one that heuristically matches the answer
        elif q_type == QUESTION_TYPES[9]:
            if not corenlp:
                if re.search('(yes, )|(no, )', ans):
                    s, e = find_regex('(yes, )|(no, )', ans)
                    hypo = replace(ans, s, e, '')
                elif ' or ' in question:
                    hypo = ans
                elif re.search('yes\W??', ans):
                    s, e = find_regex('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
                    hypo = replace(question, s, e, "")
                elif re.search('no\W??', ans):
                    s, e = find_regex('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
                    hypo = "not "+replace(question, s, e, "")
                else:
                    s, e = find_regex('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
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
                    s, e = find_regex('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
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
    question = "How much oil exactly is under the ground?"
    # question = "How much longer is Oscar's bus ride than Charlie's?"
    # question = "How more complicated is the problem?"
    # question = "How heavy is the apple?"
    # question = "How heavier is the apple?"
    # question = "How far away is the town?"
    # question = "How big is the apple going to be?"
    # question = "How big is the apple becoming?"
    # question = "How big will the apple be?"
    # question = "How many apples are in the fridge?"
    # question = "How severely was he injured?"
    # question = "How many sheep are on the hill?"
    # question = "How many children are in the classroom?"
    # question = "How much milk is in the bottle?"
    # question = "How long did he run?"
    # question = "How many pairs of shoes are on the shelf?"
    answer = "0.5"
    tree = get_parse_tree(question)
    tree.pretty_print()
    for subtree in tree.subtrees():
        print subtree.label() + ":" + ' '.join(subtree.leaves())
    
    # sent = qa2hypo(question, answer, True, False)

    ############################
    # test on single sentence
    ############################
    # a = get_args()
    # qa_pairs_list = pre_proc(a, 'math')
    # res = qa2hypo_test(qa_pairs_list)
    # post_proc(a, res, 'math') # includes writing to file

