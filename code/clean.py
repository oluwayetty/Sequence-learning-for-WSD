from datetime import datetime
import pickle

from lxml import etree
from collections import namedtuple
from nltk.corpus import wordnet

from config import XML_FILEPATH, GOLD_FILE, FOUND_VOCAB_PATH, NOTFOUND_VOCAB_PATH, \
    WND_DOMAINS_PATH, OUTPUT_DATA

TrainSet = namedtuple("TrainSet", "word_id word_pos word_lemma is_instance")
GoldSet = namedtuple("GoldSet", "word_id word_senses")


def log_message(file_handle, message, to_stdout=True, with_time=True):
    if with_time:
        message = "%s [%s]" % (message, str(datetime.now()))

    if file_handle is not None:
        file_handle.write("%s\n" % message)
        file_handle.flush()

    if to_stdout:
        print("%s" % message)


def read_train(train_path=None):
    sentence_list = []
    train_file = train_path if train_path else XML_FILEPATH
    for event, sentence in etree.iterparse(train_file, tag="sentence"):
        if event == "end":
            for element in sentence:
                if element.tag == "wf":
                    entry = TrainSet(word_id=None, word_pos=element.get("pos"), word_lemma=element.get("lemma"), is_instance=False)
                    sentence_list.append(entry)
                elif element.tag == "instance":
                    entry = TrainSet(element.get("id"), element.get("pos"), element.get("lemma"), True)
                    sentence_list.append(entry)
                    if not sentence_list[0].is_instance:
                        current_entry = sentence_list[0]
                        sentence_list[0] = TrainSet(current_entry.word_id, current_entry.word_pos, current_entry.word_lemma, True)
            yield sentence_list

            sentence.clear()


def get_sentence_id(val):
    return ".".join(val.split(".")[:-1])


def build_wordnet_id(sense):
    syn = wordnet.lemma_from_key(sense).synset()
    offset = str(syn.offset())
    offset = "0" * (8 - len(offset)) + offset
    wn_id = "wn:%s%s" % (offset, syn.pos())
    return wn_id


def read_gold_file():
    output = []
    gold_file = GOLD_FILE
    with open(gold_file, mode="r") as gold:
        for line in gold:
            line = line.strip().split(' ')
            word_id, sense_id = line[0], line[1:]
            for sense in sense_id:
                wordnet_id = build_wordnet_id(sense)
                output.append(wordnet_id)
        return output


def read_gold(gold_path=None):
    sent = []
    forgotten_sent_id = None
    forgotten_entry = None
    prev_sent_id = None
    gold_file = gold_path if gold_path else GOLD_FILE
    word_senses = set()
    with open(gold_file, mode="r") as gold:
        for line in gold:
            line = line.strip().split(' ')
            word_id, sense_id = line[0], line[1:]
            sent_id = get_sentence_id(word_id)

            for sense in sense_id:
                wordnet_id = build_wordnet_id(sense)
                word_senses.add(wordnet_id)
            entry = GoldSet(word_id=word_id, word_senses=list(word_senses))

            if forgotten_entry is not None and len(sent) == 0:
                sent.append(forgotten_entry)
                if forgotten_sent_id != sent_id:
                    prev_sent_id = forgotten_sent_id
                    forgotten_sent_id = sent_id
                    forgotten_entry = entry

                    yield sent

            if prev_sent_id is None or prev_sent_id == sent_id:
                sent.append(entry)
                prev_sent_id = sent_id
            else:
                forgotten_sent_id = sent_id
                forgotten_entry = entry
                yield sent


def get_lemma(word_lemma):
    lemma_map = {"NUM": "<NUM>", "SYM": "<SYMBOL>", "PUNCT": "<PUNCTUATION>", ".": "<PUNCTUATION>"}
    return lemma_map.get(word_lemma) if word_lemma in list(lemma_map.keys()) else word_lemma


def build_input():
    sentences = read_train()
    instances = set()
    word_counts = {}
    vocab = []
    non_vocab = set()
    vocab_map = {"<PAD>": 0, "<UNK>": 1, "<S>": 2, "</S>": 3, "<SUB>": 4}
    all_vocab = ["<PAD>", "<UNK>", "<S>", "</S>", "<SUB>"]
    total = 0
    for sentence in sentences:
        for word in sentence:
            lemma = get_lemma(word.word_lemma)
            if word.word_id is not None:
                if word.is_instance:
                    instances.add(lemma)
            word_counts[lemma] = word_counts.get(lemma, 0) + 1
            total += 1
        with open(FOUND_VOCAB_PATH, 'wb') as vocab_file, open(NOTFOUND_VOCAB_PATH, 'w') as non_vocab_file:
            for w in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
                if w[0] in instances:
                    vocab.append(w[0])
                    pickle.dump(vocab, vocab_file)
                else:
                    non_vocab.add(w[0])
                    pickle.dump(non_vocab, non_vocab_file)

    for word in vocab:
        word = word.strip()
        if word not in vocab_map.keys():
            vocab_map[word] = len(vocab)
            all_vocab.append(word)


def build_output():
    gold = read_gold_file()
    output = []
    mapping, reverse_mapping = read_mapping(WND_DOMAINS_PATH)

    for goldId in gold:
        for mappingID in reverse_mapping[goldId]:
            try:
                v = mapping[mappingID]
            except KeyError:
                v = []
            output.extend(v)
            print(output)
            with open(OUTPUT_DATA, 'wb') as output_file:
                pickle.dump(output, output_file)


def read_mapping(file_path):
    mapping = {}
    reverse_mapping = {}

    with open(file_path, mode="r") as f:
        for line in f:
            line = line.strip().split("\t")

            if len(line) < 2:
                continue

            key = line[0]
            if key not in mapping:
                mapping[key] = []
                for value in line[1:]:
                    mapping[key].append(value)
            else:
                for value in line[1:]:
                    mapping[key].append(value)

            for value in line[1:]:
                if value not in reverse_mapping:
                    reverse_mapping[value] = [key]
                else:
                    reverse_mapping[value].append(key)

    return mapping, reverse_mapping
