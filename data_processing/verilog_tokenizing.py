import tokenize
import random
import json
from functools import reduce
from operator import add


def getLines(tokens, types):
    idxs = [i for i, j in enumerate(types) if j == 'NEWLINE']
    token_lines = [tokens[i: j] for i, j in zip([0] + [idx+1 for idx in idxs], [idx for idx in idxs] + [None])]
    type_lines = [types[i: j] for i, j in zip([0] + [idx+1 for idx in idxs], [idx for idx in idxs] + [None])]
    return token_lines, type_lines


def get_pair(tokens, types, input_stmt_len):
    assert len(tokens) == len(types)
    token_lines, type_lines = getLines(tokens, types)
    input_stmt_len = min(input_stmt_len, len(token_lines)-1)
    if input_stmt_len == 0:
        return [], [], [], []
    token_inp = []
    type_inp = []
    for i in range(len(token_lines) - input_stmt_len):
        token_inp.append(reduce(add, token_lines[i:i+input_stmt_len]))
        type_inp.append(reduce(add, type_lines[i:i+input_stmt_len]))
    token_tar = token_lines[input_stmt_len:]
    type_tar = type_lines[input_stmt_len:]

    assert len(token_inp) == len(token_tar)
    assert len(type_inp) == len(type_tar)
    assert len(token_inp) == len(type_inp)
    return token_inp, type_inp, token_tar, type_tar


def create_instance(file_path):
    with open(file_path, 'rb') as f:
        data = json.loads(f.read())

    all_data = list(data.values())
    random.shuffle(all_data)
    token_inp, token_tar, type_inp, type_tar = [], [], [], []
    for tokens, types in all_data:
        token_x, type_x, token_y, type_y = get_pair(tokens, types, 10)
        if len(token_x) > 0:
            token_inp.extend(token_x)
            token_tar.extend(token_y)
            type_inp.extend(type_x)
            type_tar.extend(type_y)
    return token_inp, token_tar, type_inp, type_tar


def write_data(data_instance, source_token_file=None, source_type_file=None, target_token_file=None, target_type_file=None):
    sf_token = open(source_token_file, 'w', encoding='utf-8')
    sf_type = open(source_type_file, 'w', encoding='utf-8')
    tf_token = open(target_token_file, 'w', encoding='utf-8')
    tf_type = open(target_type_file, 'w', encoding='utf-8')
    input_token_data, target_token_data, input_type_data, target_type_data = data_instance
    assert len(input_token_data)==len(input_type_data)==len(target_token_data)==len(target_type_data)
    cnt = 0
    for i in range(len(input_token_data)):
        cnt += 1
        if cnt%1000==0:
            print('{}/{}'.format(cnt, len(input_token_data)))
        # print(' '.join(input_token_data[i]))
        # print(' '.join(target_token_data[i]))
        input_token_data[i] = [token.replace(' ', '_').replace('\n', '').replace('\r', '') for token in input_token_data[i]]
        target_token_data[i] = [token.replace(' ', '_').replace('\n', '').replace('\r', '') for token in target_token_data[i]]
        input_token_stmts = ' '.join(input_token_data[i]).replace('\n', '').replace('\r', '')
        target_token_stmts = ' '.join(target_token_data[i]).replace('\n', '').replace('\r', '')
        input_type_stmts = ' '.join(input_type_data[i])
        target_type_stmts = ' '.join(target_type_data[i])

        assert len(input_token_stmts.split(' ')) == len(input_type_stmts.split(' '))
        assert len(target_token_stmts.split(' ')) == len(target_type_stmts.split(' '))
        if len(input_token_stmts) == 0 or len(target_token_stmts)==0:
            continue
        sf_token.write(input_token_stmts + '\n')
        sf_type.write(input_type_stmts + '\n')
        tf_token.write(target_token_stmts + '\n')
        tf_type.write(target_type_stmts + '\n')


if __name__=="__main__":
    data_path = "D:/source/PycharmProjects/SANAR/data_processing/verilog/"
    eval_instances = create_instance(data_path + "eval_tokenize_res.txt")
    # write_data(eval_instances, 'verilog/py150/token_and_type/eval.token.x', 'verilog/py150/token_and_type/eval.type.x', 'verilog/py150/token_and_type/eval.token.y', 'verilog/py150/token_and_type/eval.type.y')
    write_data(eval_instances, 'verilog/token_and_type/eval.xtoken',
               'verilog/token_and_type/eval.xtype', 'verilog/token_and_type/eval.ytoken',
               'verilog/token_and_type/eval.ytype')

    eval_instances = create_instance(data_path + "train_tokenize_res.txt")
    # write_data(eval_instances, 'verilog/py150/token_and_type/train.token.x', 'verilog/py150/token_and_type/train.type.x', 'verilog/py150/token_and_type/train.token.y', 'verilog/py150/token_and_type/train.type.y')
    write_data(eval_instances, 'verilog/token_and_type/train.xtoken', 'verilog/token_and_type/train.xtype',
               'verilog/token_and_type/train.ytoken', 'verilog/token_and_type/train.ytype')

    test_instances = create_instance(data_path + "test_tokenize_res.txt")
    write_data(test_instances, 'verilog/token_and_type/test.xtoken', 'verilog/token_and_type/test.xtype',
               'verilog/token_and_type/test.ytoken', 'verilog/token_and_type/test.ytype')