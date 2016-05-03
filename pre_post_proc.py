import os
import json
import re
from globe import *

# pre-processing
# use ~/csehomedir/projects/dqa/dqa-data/shining3-vqa for diagram question answering
# use ~/csehomedir/projects/dqa/math-data for math question answering
# use ~/csehomedir/projects/dqa/MultiEQProbs for math question answering updated
def pre_proc(args, domain):
    print("Loading json files ...")
    root_dir = args.root_dir

    # Min's diagram qa dataset
    if domain == 'diagram':
        qa_path = os.path.join(root_dir, 'qa_pairs.json')
        qa_pairs = json.load(open(qa_path, 'rb'))
        qa_pairs_list = qa_pairs['qa_pairs']

    # Rik's math qa dataset
    elif domain == 'math_rik':
        
        qa_path = os.path.join(root_dir, 'qa_pairs.json')
        qa_pairs = json.load(open(qa_path, 'rb'))
        qa_pairs_list = qa_pairs
        # for item in qa_pairs_list:
        #     question = item[Q_ALIAS]
        #     s, e = find_regex(QUESTION_TYPES[7], question)
        #     item[Q_ALIAS] = question[s:]

    # Aida's math qa dataset
    elif domain == 'math_aida':
        qa_path = os.path.join(root_dir, 'data.txt')
        qa_pairs_list = []
        ctr = 0

        list_len = []
        list_head = []
        with open(os.path.join(root_dir, 'data_clean.txt'), 'wb') as fw:
            with open(qa_path, 'r') as f:
                for line in f:
                    q = line.strip()
                    if q != '4' and q != "":
                        
                        q_head, q_real = q_aida_extract(q)
                        q_list = q_aida_identify(q_real)
                        for q_sub in q_list:
                            item = {}
                            item[Q_ALIAS] = q_sub
                            item[A_ALIAS] = '4'
                            qa_pairs_list.append(item)

                            # print "sent: ", q
                            # fw.write("sent: "+q+'\n')
                            # print "real: ", q_sub
                            # fw.write('real: '+q_sub+'\n')
                            # print 
                            # fw.write('\n')

                        list_len.append(len(q_list))
                        list_head.append(q_head)

                    ctr += 1

    return qa_pairs_list

# post-processing
# use ~/csehomedir/projects/dqa/dqa-data/shining3-vqa for diagram question answering
# use ~/csehomedir/projects/dqa/math-data for math question answering
# use ~/csehomedir/projects/dqa/MultiEQProbs for math question answering updated
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

# isolate questions and descriptions heuristically
def q_aida_extract(q):
    q_list = q.split('.')
    l = len(q_list)
    q_real = ""
    q_head = ""
    if l > 1:
        if q_list[-1] != "":
            i = 0
            q_real = q_list[l-1]
            while re.match('\d', q_list[l-2-i][-1]) and re.match('\d', q_list[l-2-i+1][0]) and l-2-i>=0:
                q_real = q_list[l-2-i]+'.'+q_real
                i+=1
            q_head = '.'.join(q_list[:l-2-i+1])+'.'
        else:
            q_real = q_list[-2]+'.'
            q_head = '.'.join(q_list[:-2])+'.'
    else:
        q_real = q
        q_head = ""
    return q_head, q_real

# identify individual questions heuristically
def q_aida_identify(q_real):
    q_real_list = q_real.split('?')
    ans = []
    for q in q_real_list:
        if q != "":
            ans.append(q+'?')
    return ans



