# https://github.com/alisoltanirad/Information-Extractor.git
# Dependencies: nltk
import ssl
import nltk

def main():
    download_resources()
    #noun_phrase = 'NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}'
    noun_phrase = 'NP: { < [CDJNP]. * > +}'
    #verb_to_verb = 'VtV: {<V.*><TO><V.*>}'
    print(evaluate_chunker(get_chunker(noun_phrase)))


def evaluate_chunker(chunker):
    evaluation_data = nltk.corpus.conll2000.chunked_sents(chunk_types=['NP'])
    return chunker.evaluate(evaluation_data)


def chunk(chunker, sentence):
    return chunker.parse(sentence)


def get_chunker(chunk_grammar):
    return nltk.RegexpParser(chunk_grammar)


def preprocess_text(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    sentences = [nltk.pos_tag(sentence) for sentence in sentences]
    return sentences


def download_resources():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('averaged_perceptron_tagger')
    nltk.download('conll2000')
    nltk.download('brown')


if __name__ == '__main__':
    main()