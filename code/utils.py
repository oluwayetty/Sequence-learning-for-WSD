import csv
import string
import json
import pandas as pd
from typing import List, Dict
from nltk.corpus import wordnet as wn
from lxml import etree as ET
from nltk.corpus import stopwords
from nltk import word_tokenize
stoplist = stopwords.words('english')
stop = list(string.punctuation)
from config import XML_FILEPATH, DATA_FILE, GOLD_FILE, BABEL_WORDNET, JSON_FILE


class Error(Exception):
    """Base class for other exceptions"""
    pass

class EmptyTagError(Error):
    """Raised when the text lang tag is empty"""
    print("Empty tag detected")

def parse_to_dict(filename: str, gold = True) -> Dict:
    """
    Read each line of gold text file or babelnet/wordnetID.
    :param filename
    :returns a dictionary
    """
    dict = {}
    with open(filename, 'r') as f:
        content = f.readlines()
        for line in content:
            if gold:
                unpacked = line.split(" ")
                instance_id, sense_key = unpacked[0], unpacked[1]
            else:
                unpacked = line.strip().split("\t")
                instance_id, sense_key = unpacked[1], unpacked[0]

            dict[instance_id] = sense_key
    return dict


def read_xml(filename: str) -> List[str]:
    """
    Read the xml sentence by sentence.
    :param filepath - xml path of our training corpus
    :returns sentences and sentences replaced with babelnetID
    """

    xml_content = ET.iterparse(filename, events=('end',), tag='sentence')
    print("Parsing xml file.......")

    dict_gold = parse_to_dict(GOLD_FILE)
    dict_babelnet = parse_to_dict(BABEL_WORDNET, gold = False)

    sentence_array = []
    for event, element in xml_content:
        try:

            for elem in element.iter():
                if elem.tag == 'sentence':
                    data_dict = {}
                    arr_sen = ["".join(child.attrib['lemma']) for child in elem]
                    for child in elem:
                        info_array = []
                        if 'id' in child.attrib:
                            word = child.attrib['lemma']
                            sensekey = dict_gold.get(child.attrib['id']).strip()
                            synset = wn.lemma_from_key(sensekey).synset()
                            synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
                            babelnetID = dict_babelnet.get(synset_id)
                            info_array.extend([babelnetID, child.attrib['pos']])
                            data_dict[word] = info_array
                        else:
                             # word, value = [child.attrib['lemma'], child.attrib['pos']]
                             # word, value = [child.attrib['lemma'], 'UNK']
                             data_dict[child.attrib['lemma']] = 'UNK'
                    sentence_array.append(data_dict)

        except EmptyTagError:
            continue

        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]


    print("Parsing all done.......")
    return sentence_array
    # return english_texts, ids

def remove_stop_words(data):
    print("Removing stop words using nltk toolkit")
    all_clean = []
    for each in data:
        clean = [i for i in word_tokenize(each.lower()) if i not in stop]
        clean = " ".join(clean)
        all_clean.append(clean)
    return all_clean


def parse_to_csv(sentence: List, babel_sentence: List) -> str:
    """
    Read each sentence in the array of sentences.
    :param sentences - lists of sentences
    :returns csv format of sentences & babel_sentence
    """
    header = ['sentence', 'babel_sentence']
    rows = zip(sentence,babel_sentence)
    with open(DATA_FILE, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(i for i in header)
        wr.writerows(rows)

def parse_to_json(sentences: List) -> None:
    with open(JSON_FILE, 'w') as fout:
        json.dump(sentences, fout)


if __name__ == "__main__":
    print("Loading xml path")
    x = read_xml(XML_FILEPATH)
    parse_to_json(x)
    # import ipdb; ipdb.set_trace()
    # clean_sentences, clean_babel_sentences = remove_stop_words(sentences), remove_stop_words(babel_sentences)
    # parse_to_csv(clean_sentences, clean_babel_sentences)
    print("Data saved to CSV path")
