import os
import json

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
        print qa_path
        with open(qa_path, 'r') as f:
            a = f.readline()
            b = f.readline()
            c = f.readline()
            print a
            print b
            print c
        qa_pairs_list = None

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