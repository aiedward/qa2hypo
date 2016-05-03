import numpy as np

import random
import re
import string

from helper import *
from globe import *
from pre_post_proc import *


# turn qa_pairs into hypotheses, test (can sample questions)
def qa2hypo_test(qa_pairs_list):
    # number of samples and the types of questions to sample
    k = SAMPLE_TYPE
    
    # sampling question type (for examining the result)
    q_type = QUESTION_TYPES[QUESTION_TYPE]
    if k != -2:
        qa_pairs_list = sample_qa(qa_pairs_list, k, q_type) # set the case lower in the function for questions
    
    # result file
    res = []

    ctr = 0
    for item in qa_pairs_list:
        # get question
        question = item[Q_ALIAS]
        # get answer
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
    # question normalization
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

    try:
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
                        
                        # find the position of WHADVP
                        s_whadvp, e_whadvp = find_type_position(question_rear, 'WHADVP')

                        e_list = [e_whadjp, e_whnp, e_whadvp]
                        s_list = [s_whadjp, s_whnp, s_whadvp]
                        e_wh, i_min = find_min(e_list)
                        s_wh = s_list[i_min]
                        # print question_rear[s_wh:e_wh]

                        # find the first auxiliary verb
                        s_aux, e_aux = find_regex(AUX_V_REGEX, question_rear)
                        aux_be = question_rear[s_aux:e_aux]
                        # print aux_be

                        # find the existence of [does]
                        s_does, e_does = find_regex(AUX_V_DOESONLY_REGEX, question_rear[s_aux:e_aux])

                        # find the first [vp]
                        s_vp, e_vp = find_type_root(question_rear, 'VP')
                        # print 'vp: ', question_rear[s_vp:e_vp]

                        # find the [np] after [aux_v]
                        s_np = e_aux + 1
                        if s_vp != None:
                            e_np = s_vp - 1
                        else:
                            question_trim = question_rear[s_np:]
                            s_tmp, e_tmp = find_type_position(question_trim, 'NP')
                            if e_tmp != None:
                                e_np = s_np + e_tmp - s_tmp
                            else:
                                e_np = len(question_rear)-1
                        # print 'np: ', question_rear[s_np:e_np]


                        # determine if there is [adv] between [wh] and [np]
                        s_adv = 0
                        e_adv = 0
                        if (e_whnp != None):
                            adv = question_rear[e_whnp:s_aux]
                            if adv.strip() != "":
                                s_adv, e_adv = find_type_position(adv, 'ADVP')


                        # [whnp] + [aux_v] without [vp] or without [np]
                        ######### how many applies are peeled
                        ######### how many applies are on the table
                        if (e_whnp != None) and ((s_aux - e_whnp <= 2) or (s_adv != e_adv)) and ((e_vp == None) or (e_np - s_np <=0)):
                            hypo = replace(question_rear, s_wh, e_wh, ans)
                        # need change of word order
                        else:
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

                else:
                    hypo = replace(question, s, e, ' '+ans+' is how ')


            elif q_type == QUESTION_TYPES[8]:
                s, e = find_regex('(name)|(choose)|(identify)|(find)|(determine)', question)
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
    except:
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





