# for parsing sentences using the stanford core nlp package with a python wrapper
# from https://github.com/dasmith/stanford-corenlp-python
# this is effective after launching the server by in parallel doing
# python corenlp.py
import sys
from nltk.tree import *
sys.path.insert(0, './stanford-corenlp-python')
import jsonrpc
from simplejson import loads
server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(), jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))


# get an nltk tree structure
def get_parse_tree(sent):
    sent_parse = loads(server.parse(sent))
    parse_tree = sent_parse['sentences'][0]['parsetree']
    tree = Tree.fromstring(parse_tree)
    return tree