from parser import *
import re
import en
# from nltk.tokenize import TweetTokenizer

# strip any non alnum characters in the end
def strip_nonalnum_re(sent):
    return re.sub(r"^\W+|\W+$", "", sent)

# replace 
def replace(text, start, end, replacement):
    text_left = text[:start]
    text_right = text[end:]

    return text_left+replacement+text_right

# for print purpose
def test_patterns(patterns, text):
    """Given source text and a list of patterns, look for
    matches for each pattern within the text and print
    them to stdout.
    """
    # Show the character positions and input text
    # print
    # print ''.join(str(i/10 or ' ') for i in range(len(text)))
    # print ''.join(str(i%10) for i in range(len(text)))
    # print text

    # Look for each pattern in the text and print the results
    for pattern in patterns:
        # print
        # print 'Matching "%s"' % pattern
        # --- regex ---
        for match in re.finditer(pattern, text):
            s = match.start()
            e = match.end()
            # print '  %2d : %2d = "%s"' % (s, e-1, text[s:e])
            
            # print '    Groups:', match.groups()
            # if match.groupdict():
            #     print '    Named groups:', match.groupdict()
            # print
    return

# for return purpose
def find_regex(pattern, text):
    match = re.search(pattern, text)
    pos = len(text)-1
    if not match:
        return pos, pos
    s = match.start()
    e = match.end()
    # print '  %2d : %2d = "%s"' % (s, e-1, text[s:e])
    return s, e

# find the positions of the NPs or VPs around 'or'
def find_or_pos(question, ans, q_type):

    # sent_parse = loads(server.parse(question))
    # parse_tree = sent_parse['sentences'][0]['parsetree']
    # tree = ParentedTree.fromstring(parse_tree)
    tree = get_parse_tree(question)
    # tree.pretty_print()

    or_node = None
    for subtree in tree.subtrees(filter=lambda x: x.label() == 'CC'):
        # print "or position:", subtree.leaves()
        or_node = subtree
        break

    # left_siblings = []
    # l = or_node.left_sibling()
    # while l:
    #     left_siblings.append(l)
    #     l = l.left_sibling()

    # right_siblings = []
    # r = or_node.right_sibling()
    # while r:
    #     right_siblings.append(r)
    #     r = r.right_sibling()

    # print left_siblings
    # print right_siblings

    or_parent = or_node.parent()
    candidates_tok = or_parent.leaves()
    candidates_len = len(' '.join(candidates_tok))

    candidates_list = []
    item = ''
    for tok in candidates_tok:
        if (tok != ',') and (tok != 'or'):
            item = item + ' ' + tok
        else:
            candidates_list.append(item)
            item = ''
    candidates_list.append(item)

    candidate_chosen = candidates_list[0]
    for candidate in candidates_list:
        if ans in candidate:
            candidate_chosen = candidate
            break
    # print("candidates_list:", candidates_list)
    # print("candidate_chosen:", candidate_chosen)

    s0, e0 = find_regex(candidates_list[0].strip(), question)
    s1, e1 = find_regex(candidates_list[-1].strip(), question)

    return s0, e1, candidate_chosen

# find the minimum of a list that contains None elements
def find_min(noneList):
    if noneList == []:
        return None, None
    
    i = 0
    while (noneList[i]==None) and (i<len(noneList)):
        i += 1
    if i >= len(noneList):
        return None, None

    i_min = i
    a = noneList[i]

    while i < len(noneList):
        if noneList[i] == None:
            i+=1
        else:
            if noneList[i] < a:
                a = noneList[i]
                i_min = i
            i+=1

    return a, i_min

# transform a verb into an appropriate tense
def v_transform(v_old, np, aux_v):
    v_new = v_old

    if aux_v.strip() == "did":
        v_new = en.verb.past(v_old)
    elif aux_v.strip() == "does":
        v_new = en.verb.present(v_old, person=3, negate=False)

    return v_new


# find the first subtree of a certain node type
def find_first_subtree(tree, node_type):
    for subtree in tree.subtrees(filter=lambda x: x.label() == node_type):
        return subtree

# find all the subtrees of a certain node type
def find_all_subtree(tree, node_type):
    subtreeList = []
    subtreePos = []
    for subtree in tree.subtrees(filter=lambda x: x.label() == node_type):
        subtreeList.append(subtree)
    return subtreeList

# find the root of the first subtree of a certain node type  
def find_first_root(tree, node_type):
    a = find_first_subtree(tree, node_type)
    if a == None:
        return None
    # print a[0].label()
    return a[0]

# find the first appearance of a node root
def find_type_root(sent, node_type):
    tree = get_parse_tree(sent)
    subtree = find_first_root(tree, node_type)
    if subtree != None:
        subtree_str = ' '.join(subtree.leaves())
        tmp = ((subtree_str.strip()).split(' '))[0]
        s, e_tmp = find_regex(tmp, sent)
        e = s+len(subtree_str)
    else:
        s, e = None, None
    return s, e

# find the first appearance of a node
def find_type_position(sent, node_type):
    tree = get_parse_tree(sent)
    subtree = find_first_subtree(tree, node_type)
    if subtree != None:
        # print subtree.label()
        subtree_str = ' '.join(subtree.leaves())
        # print 'subtree: ', subtree_str
        tmp = ((subtree_str.strip()).split(' '))[0]
        s, e_tmp = find_regex(tmp, sent)

        s_than, e_than = find_regex(r'\bthan\b', sent)
        if s_than == e_than:
            e = s+len(subtree_str)
        else:
            e = s_than
    else:
        s, e = None, None
    return s, e


# find the positions of the aux_v and the first noun
def find_np_pos(question, ans, q_type, node_type='NP', if_root_node=False):
    s_aux, e_aux = find_regex(q_type, question)
    # print '  %2d : %2d = "%s"' % (s_aux, e_aux-1, question[s_aux:e_aux])
    
    if node_type=='NP':
        question = question[s_aux:]
    elif node_type=='VP':
        question = question[e_aux:]

    # print "Shortened question:", question

    tree = get_parse_tree(question)
    # tree.pretty_print()

    first_NP = None

    if (node_type == 'NP') or (node_type == 'VP'):
        # get the whole node
        if not if_root_node:
            subtree = find_first_subtree(tree, node_type)
            # print "whole node:\n", subtree.pretty_print()
            first_NP = ' '.join(subtree.leaves())

        # get the root node
        else:
            root = find_first_root(tree, node_type)
            # print "root node:\n", root.pretty_print()
            if root != None:
                first_NP = ' '.join(root.leaves())

        # print node_type+':', first_NP
        # print

    first_NP_len = 0
    if first_NP:
        first_NP_len = len(first_NP)
        s_np, e_np = find_regex((first_NP.split(' '))[0], question)
    else:
        s_np = len(question)-1
    # s_np = e_aux+1
    e_np = s_np + first_NP_len

    if node_type == 'NP':
        return s_aux, e_aux, s_np+s_aux, e_np+s_aux, first_NP
    elif node_type == 'VP':
        return s_aux, e_aux, s_np+e_aux, e_np+e_aux, first_NP
    else:
        return s_aux, e_aux, s_np, e_np, first_NP


# # find the WHNP node
# def find_node_pos(question, node_type):
#     tree = get_parse_tree(question)
#     # tree.pretty_print()
#     subtree = find_first_subtree(tree, node_type)
#     if subtree == None:
#         return None, None
#     text = subtree.leaves()
#     text = ' '.join(text)
#     s, e = find_regex(text, question)
#     if s == e:
#         return None, None
#     else:
#         return s, e


# question normalization
def q_norm(question):
    question = question.lower()

    # if question.endswith('?') or question.endswith(':'):
    #     question = question[:-1]

    return question

# answer normalization
def a_norm(answer):
    if isinstance(answer, float) or isinstance(answer, int):
        answer = str(answer)
    answer = answer.lower().strip('.')
    return answer