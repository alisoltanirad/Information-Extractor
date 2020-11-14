# https://github.com/alisoltanirad/Information-Extractor.git
# Dependencies: nltk
import ssl
import nltk

def main():
    download_resources()
    #noun_phrase = 'NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}'
    #verb_to_verb = 'VtV: {<V.*><TO><V.*>}'


def chunk(sentence, chunk_grammar):
    chunk_parser = nltk.RegexpParser(chunk_grammar)
    parsed_sentence = chunk_parser.parse(sentence)
    return parsed_sentence


def preprocess_text(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    sentences = [nltk.pos_tag(sentence) for sentence in sentences]
    return sentences


def download_resources():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('averaged_perceptron_tagger')
    nltk.download('brown')


if __name__ == '__main__':
    main()