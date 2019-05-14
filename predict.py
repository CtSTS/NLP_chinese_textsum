import tensorflow as tf
from model import InferenceModel
import re
import jieba
import sys
from collections import Counter

def pred(source_str):
    with open("int_to_vocab_50000.txt","r",encoding='utf8',errors="ignore") as f:
        #dic_str = f.read()
        #int_to_vocab = eval("{"+", ".join([": '".join(line.split(' '))+"'" for line in dic_str.split('\n')])+"}")
        int_to_vocab = eval(f.read())
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    test_source_str = jieba.cut(re.sub(u'[0-9a-zA-Z\+\-\*\/\\\_&^%$#@~\|`\?!\'\";:<>\.,\(\)\[\]\{\}\s]',"",re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|��.*?��|[0-9]+��|[0-9]+��|[0-9]+��", "", source_str)), cut_all=False)
    test_target_str = ''
    test_str = ' '.join(test_source_str)
    #test_source_input = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in test_str.split()]
    text_source_input=[]
    text_source_unknow=[]
    for word in test_str.split():
        x=vocab_to_int.get(word,None)
        if x == None:
            text_source_input.append(vocab_to_int['<UNK>'])
            text_source_unknow.append(word)
        else:
            text_source_input.append(x)
    vocab_size = len(int_to_vocab)
    embedding_size = 128
    num_units = 256
    num_layers = 3
    # rnn����
    # �����󳤶�
    max_target_sequence_length = 15
    #
    max_gradient_norm = 5
    learning_rate = 0.01    # 0.01
    batch_size = 64
    infer_batch_size = 1
    epochs = 50
    infer_step = 100
    beam_size  = 5
    segment_to_int = vocab_to_int
    # ����ģʽ
    infer_mode = 'beam_search'
    #infer_mode = 'greedy'

    infer_graph = tf.Graph()

    with infer_graph.as_default():
        infer_model = InferenceModel(vocab_size,embedding_size,num_units,num_layers,
                                     max_target_sequence_length, infer_batch_size, beam_size, segment_to_int, infer_mode)

    checkpoints_path = "model2/"

    infer_sess = tf.Session(graph=infer_graph)

    infer_batch = ([text_source_input],[len(text_source_input)])

    infer_model.saver.restore(infer_sess, tf.train.latest_checkpoint(checkpoints_path))
    current_predict = infer_model.infer(infer_sess, infer_batch)
    #print(current_predict[0])
    # greedy
    #result = ''.join([int_to_vocab[idxes] for idxes in current_predict[0][0]]).replace('<EOS>', '')
    # beam_search
    #result = ''.join([int_to_vocab[Counter(idxes).most_common(1)[0][0]] for idxes in current_predict[0][0]]).replace('<EOS>','')
    #print(result)
    print(''.join([int_to_vocab[idxes[0]] for idxes in current_predict[0][0]]).replace('<EOS>', ''))

    infer_mode = 'greedy'

    infer_graph = tf.Graph()

    with infer_graph.as_default():
        infer_model = InferenceModel(vocab_size, embedding_size, num_units, num_layers,
                                     max_target_sequence_length, infer_batch_size, beam_size, segment_to_int,
                                     infer_mode)

    checkpoints_path = "model2/"

    infer_sess = tf.Session(graph=infer_graph)

    infer_batch = ([text_source_input], [len(text_source_input)])

    infer_model.saver.restore(infer_sess, tf.train.latest_checkpoint(checkpoints_path))
    current_predict = infer_model.infer(infer_sess, infer_batch)
    # print(current_predict[0])
    # greedy
    text_unknow_num = 0
    words = [int_to_vocab[idxes] for idxes in current_predict[0][0]]
    for index in range(len(words)):
        if words[index] == '<UNK>':
            words[index] = text_source_unknow[text_unknow_num]
            text_unknow_num = (text_unknow_num+1)%len(text_source_unknow)
    words = words + text_source_unknow
    result = ''.join(words).replace('<EOS>', '')
    print(result)

if __name__ == "__main__":
    print('begin:')
    init = jieba.cut('��ͳ�ʼ��','')

    pred(source_str = sys.stdin.readline())

