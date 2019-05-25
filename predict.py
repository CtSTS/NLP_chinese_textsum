import tensorflow as tf
from model import InferenceModel
import re
import jieba
import sys
from collections import Counter

 # 读取的数据类型为str


    #test_source_input = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in test_words]

def init():
    # ÉèÖÃ»ù±¾²ÎÊý
    # ´Ê±í´óÐ¡
    with open("int_to_vocab_50000.txt","r",encoding='utf8',errors="ignore") as f:
        #dic_str = f.read()
        #int_to_vocab = eval("{"+", ".join([": '".join(line.split(' '))+"'" for line in dic_str.split('\n')])+"}")
        int_to_vocab = eval(f.read())
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    vocab_size = len(int_to_vocab) 
    # embeddingÎ¬¶È
    embedding_size = 128
    # rnnÒþ²Øµ¥ÔªÊý
    num_units = 256
    # rnn²ãÊý
    num_layers = 3
    # Êä³ö×î´ó³¤¶È
    max_target_sequence_length = 60
    #
    max_gradient_norm = 5
    # Ñ§Ï°ÂÊ
    learning_rate = 0.01    # 0.01
    # Åú´Î´óÐ¡
    batch_size = 64
    # ÍÆÀíÃ¿ÅúÒ»‚€¾ä×Ó
    infer_batch_size = 1
    # ÑµÁ·¶àÉÙ´ú
    epochs = 50
    # ¶àÉÙ²½Ô¤²âÒ»ÏÂ
    infer_step = 100
    # beam ´óÐ¡
    beam_size = 5
    # ·Ö´ÊÓ³Éä
    segment_to_int = vocab_to_int
    # ÍÆÀíÄ£Ê½
    infer_mode = 'beam_search'
    #infer_mode = 'greedy'

    infer_graph = tf.Graph()

    with infer_graph.as_default():
        infer_model = InferenceModel(vocab_size,embedding_size,num_units,num_layers,
                                     max_target_sequence_length, infer_batch_size, beam_size, segment_to_int, infer_mode)

    checkpoints_path = "model2/"

    infer_sess = tf.Session(graph=infer_graph)

    

    infer_model.saver.restore(infer_sess, tf.train.latest_checkpoint(checkpoints_path))
    #print(current_predict[0])
    # greedy
    #result = ''.join([int_to_vocab[idxes] for idxes in current_predict[0][0]]).replace('<EOS>', '')
    # beam_search
    #result = ''.join([int_to_vocab[Counter(idxes).most_common(1)[0][0]] for idxes in current_predict[0][0]]).replace('<EOS>','')
    #print(result)
    return infer_model,infer_sess

def pred(source_str,infer_model,infer_sess):
    with open("int_to_vocab_50000.txt","r",encoding='utf8',errors="ignore") as f:
        #dic_str = f.read()
        #int_to_vocab = eval("{"+", ".join([": '".join(line.split(' '))+"'" for line in dic_str.split('\n')])+"}")
        int_to_vocab = eval(f.read())
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    test_source_str = jieba.cut(re.sub(u'[0-9a-zA-Z\+\-\*\/\\\_&^%$#@~\|`\?!\'\";:<>\.,\(\)\[\]\{\}\s]',"",re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]|£¨.*?£©|[0-9]+Äê|[0-9]+ÔÂ|[0-9]+ÈÕ", "", source_str)), cut_all=False)
    test_target_str = ''
    test_str = ' '.join(list(test_source_str)[:100])
    test_words = test_str.split()
    test_source_input = []
    unk_list = []
    for word in test_words:
        try:
            test_source_input.append(vocab_to_int[word])
        except:
            test_source_input.append(vocab_to_int['<UNK>'])
            unk_list.append(word)
    #test_source_input = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in test_words]
    # ÉèÖÃ»ù±¾²ÎÊý
    # ´Ê±í´óÐ¡
    vocab_size = len(int_to_vocab) 
    infer_batch = ([test_source_input],[len(test_source_input)])
    current_predict = infer_model.infer(infer_sess, infer_batch)
    words = [int_to_vocab[idxes[0]] for idxes in current_predict[0][0]]

    unk_index = 0
    for i in range(len(words)):
        if words[i] == "<UNK>":
            if unk_index < len(unk_list):
                words[i] = unk_list[unk_index]
                unk_index += 1
            else:
                words[i] = '<EOS>'
    i = 0
    bound = len(words)
    while i < bound-1:
        if words[i] == words[i+1]:
            del words[i+1]
            bound -= 1
        else:
            i += 1
    words += ['***']+unk_list[unk_index:]

    result = ''.join(words).replace('<EOS>', '')
    return result


if __name__ == "__main__":
    init = jieba.cut('½á°Í³õÊ¼»¯','')
    infer_model,infer_sess = init()
    print(pred(source_str = sys.stdin.readline()),infer_model=infer_mode,infer_sess=infer_sess)
