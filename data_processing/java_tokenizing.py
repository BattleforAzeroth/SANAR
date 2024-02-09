
import javalang
import os
import json
import random
import functools
from operator import add

def tokenize(code_file):
    code = open(code_file, 'r').read()
    token_gen = javalang.tokenizer.tokenize(code)
    data = []
    while(True):
        try:
            token = next(token_gen)
            tokval = token.value
            _type = str(type(token))[:-2].split(".")[-1]
            if _type == 'String':
                tokval = '<STR>'
            elif _type.__contains__('Integer') or _type.__contains__('Float'):
                tokval = '<NUM>'
            data.append([tokval, _type, token.position[0]])
        except:
            break
    
    return data # [token, type, line_num]

def getLines(data):
    token_lines = []
    type_lines = []
    lines = []
    for tokens, types, lineno in data:
        if lineno not in lines:
            if len(token_lines) > 0:
                assert len(token_lines[-1]) == len(type_lines[-1])
            lines.append(lineno)
            token_lines.append([])
            type_lines.append([])
        token_lines[-1].append(tokens)
        type_lines[-1].append(types)
    assert len(token_lines) == len(type_lines)
    return token_lines, type_lines

def iter_files(rootDir):
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            if file_name.endswith('.java'):
                yield file_name

# save token and type information
def data_process(data_path, save_path):
    save_data = {}
    error_files = []
    cnt = 0
    for file_path in iter_files(data_path):
        if cnt % 100 == 0:
            print(cnt)
        try:
            data = tokenize(file_path)
            token_lines, type_lines = getLines(data)
            if len(token_lines) > 0:
                save_data[file_path] = {'token_lines': token_lines, 'type_lines': type_lines}
                cnt += 1
        except:
            error_files.append(file_path)
        
    with open(save_path, 'w') as wf:
        wf.write(json.dumps(save_data))



def get_pair(token_lines, type_lines, input_stmt_len):
    input_stmt_len = min(input_stmt_len, len(token_lines)-1)
    if input_stmt_len == 0:
        return [], [], [], []
    token_inp = []
    type_inp = []
    for i in range(len(token_lines) - input_stmt_len):
        token_inp.append(functools.reduce(add, token_lines[i:i+input_stmt_len]))
        type_inp.append(functools.reduce(add, type_lines[i:i+input_stmt_len]))
    token_tar = token_lines[input_stmt_len:]
    type_tar = type_lines[input_stmt_len:]

    assert len(token_inp) == len(token_tar)
    assert len(type_inp) == len(type_tar)
    assert len(token_inp) == len(type_inp)
    return token_inp, type_inp, token_tar, type_tar


def create_instance(file_path):
    with open(file_path, 'r') as f:
        data = json.loads(f.read())
    
    all_data = list(data.values())
    random.shuffle(all_data)
    token_inp, token_tar, type_inp, type_tar = [], [], [], []
    for data in all_data:
        tokens, types = data['token_lines'], data['type_lines']
        token_x, type_x, token_y, type_y = get_pair(tokens, types, 10)
        if len(token_x) > 0:
            token_inp.extend(token_x)
            token_tar.extend(token_y)
            type_inp.extend(type_x)
            type_tar.extend(type_y)
    return token_inp, token_tar, type_inp, type_tar


def write_data(data_instance, source_token_file=None, source_type_file=None, target_token_file=None, target_type_file=None):
    sf_token = open(source_token_file, 'w')
    sf_type = open(source_type_file, 'w')
    tf_token = open(target_token_file, 'w')
    tf_type = open(target_type_file, 'w')
    input_token_data, target_token_data, input_type_data, target_type_data = data_instance
    assert len(input_token_data)==len(input_type_data)==len(target_token_data)==len(target_type_data)
    cnt = 0
    for i in range(len(input_token_data)):
        cnt += 1
        if cnt%1000==0:
            print('{}/{}'.format(cnt, len(input_token_data)))

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
        # print(input_token_stmts)
        # print(input_type_stmts)
        sf_token.write(input_token_stmts + '\n')
        sf_type.write(input_type_stmts + '\n')
        tf_token.write(target_token_stmts + '\n')
        tf_type.write(target_type_stmts + '\n')

if __name__=="__main__":

    data_path = 'nat_java_data/'
    train_path = 'java_train_projects/'
    valid_path = 'java_valid_projects/'
    test_path = 'java_test_projects/'
    
    data_process(data_path+train_path, data_path + "java_train_tokenize_res.txt")
    data_process(data_path+valid_path, data_path + "java_valid_tokenize_res.txt")
    data_process(data_path+test_path, data_path + "java_test_tokenize_res.txt")

    train_instances = create_instance(data_path + "java_train_tokenize_res.txt")
    valid_instances = create_instance(data_path + "java_valid_tokenize_res.txt")
    test_instances = create_instance(data_path + "java_test_tokenize_res.txt")
    
    write_data(valid_instances, 'nat_java_data/token_and_type/eval.token.x', 'nat_java_data/token_and_type/eval.type.x', 'nat_java_data/token_and_type/eval.token.y', 'nat_java_data/token_and_type/eval.type.y')
    write_data(test_instances, 'nat_java_data/token_and_type/test.token.x', 'nat_java_data/token_and_type/test.type.x', 'nat_java_data/token_and_type/test.token.y', 'nat_java_data/token_and_type/test.type.y')
    write_data(train_instances, 'nat_java_data/token_and_type/train.token.x', 'nat_java_data/token_and_type/train.type.x', 'nat_java_data/token_and_type/train.token.y', 'nat_java_data/token_and_type/train.type.y')
    