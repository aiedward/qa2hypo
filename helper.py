from parser import *
import re

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
def test_pattern(pattern, text):
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

    sent_parse = loads(server.parse(question))
    parse_tree = sent_parse['sentences'][0]['parsetree']
    tree = ParentedTree.fromstring(parse_tree)
    # print the tree
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

    s0, e0 = test_pattern(candidates_list[0].strip(), question)
    s1, e1 = test_pattern(candidates_list[-1].strip(), question)

    return s0, e1, candidate_chosen

# find the first subtree of a certain node type
def find_first_subtree(tree, node_type):
    for subtree in tree.subtrees(filter=lambda x: x.label() == node_type):
        return subtree

# find the root of the first subtree of a certain node type  
def find_first_root(tree, node_type):
    a = find_first_subtree(tree, node_type)
    if a == None:
        return None
    # print a[0].label()
    return a[0]

# find the positions of the aux_v and the first noun
def find_np_pos(question, ans, q_type, node_type='NP', if_root_node=False):
    s_aux, e_aux = test_pattern(q_type, question)
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
        s_np, e_np = test_pattern((first_NP.split(' '))[0], question)
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


# find the WHNP node
def find_whnp_pos(question):
    tree = get_parse_tree(question)
    # tree.pretty_print()
    subtree = find_first_subtree(tree, 'WHNP')
    if subtree == None:
        return None, None
    text = subtree.leaves()
    text = ' '.join(text)
    s, e = test_pattern(text, question)
    if s == e:
        return None, None
    else:
        return s, e

# how many/how much transformation
def exp_how_many(q_type, question):
    s_aux, e_aux = test_pattern(q_type, question)
    question_head = question[:s_aux]
    question = question[s_aux:]
    tree = get_parse_tree(question)
    tree.pretty_print()


# question normalization
def q_norm(question):
    question = question.lower()

    if question.endswith('?') or question.endswith(':'):
        question = question[:-1]

    return question

# answer normalization
def a_norm(answer):
    if isinstance(answer, float) or isinstance(answer, int):
        answer = str(answer)
    answer = answer.lower().strip('.')
    return answer