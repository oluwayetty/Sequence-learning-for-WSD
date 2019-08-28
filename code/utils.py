import csv
import pandas as pd
from typing import List, Dict
from nltk.corpus import wordnet as wn
from lxml import etree as ET
from config import XML_FILEPATH, DATA_FILE, GOLD_FILE, BABEL_WORDNET


class Error(Exception):
    """Base class for other exceptions"""
    pass

class EmptyTagError(Error):
    """Raised when the text lang tag is empty"""
    print("Empty tag detected")

def read_xml(filename: str) -> List[str]:
    """
    Read the xml sentence by sentence.
    :param filepath - xml path of our training corpus
    :returns sentences and sentences replaced with babelnetID
    """
    english_texts, ids = [], []
    xml_content = ET.iterparse(filename, events=('end',), tag='sentence')
    print("Parsing xml file.......")

    for event, element in xml_content:
        try:
            for elem in element.iter():
                if elem.tag == 'sentence':
                    id = []
                    arr_sen = ["".join(child.text) for child in elem]
                    for child in elem:
                        if 'id' in child.attrib:
                            dict_gold = parse_to_dict(GOLD_FILE)
                            sensekey = dict_gold.get(child.attrib['id']).strip()
                            babelnetID = sense_key_to_babelID(sensekey)
                            id.append(babelnetID)
                        else:
                            id.append(child.attrib['lemma'])
                    ids.append(" ".join(id))

            english_texts.append(" ".join(arr_sen))

        except EmptyTagError:
            continue

        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]

    assert len(english_texts) == len(ids)
    print("Parsing all done.......")
    return english_texts, ids


def sense_key_to_babelID(sense_key: str) -> str:
    synset = wn.lemma_from_key(sense_key).synset()
    synset_id = "wn:" + str(synset.offset()).zfill( 8) + synset.pos()
    dict_wordnet = parse_to_dict(BABEL_WORDNET, gold = False)
    return dict_wordnet.get(synset_id)


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


def read_file(filename: str):
    """
    Read the csv dataset line by line.
    :param filename: file to read
    :return:
    """
    data = pd.read_csv(DATA_FILE)
    print(data.describe())


if __name__ == "__main__":
    print("Loading xml path")
    sentences, babel_sentences = read_xml(XML_FILEPATH)
    parse_to_csv(sentences, babel_sentences)
    print("Data saved to CSV path")
    # read_file(DATA_FILE)
