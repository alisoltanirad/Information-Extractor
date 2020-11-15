# https://github.com/alisoltanirad/Information-Extractor.git
# Dependencies: nltk
import ssl
import nltk

class BigramChunker(nltk.ChunkParserI):

    def __init__(self, train_sentences):
        train_data = [[(pos, chunk)
                       for _, pos, chunk in nltk.chunk.tree2conlltags(sentence)]
                      for sentence in train_sentences]
        self.tagger = nltk.BigramTagger(train_data)


    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        pos_chunk_tags = self.tagger.tag(pos_tags)
        chunk_tags = [chunk for (pos, chunk) in pos_chunk_tags]
        conll_tags = [(word, pos, chunk)
                      for ((word, pos), chunk) in zip(sentence, chunk_tags)]
        return nltk.chunk.conlltags2tree(conll_tags)


def main():
    download_resources()
    print_evaluation_scores()


def print_evaluation_scores():
    train_set, test_set = get_data_set()
    print(BigramChunker(train_set).evaluate(test_set))


def get_data_set():
    train = nltk.corpus.conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    test = nltk.corpus.conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    return train, test


def evaluate_re_chunker(chunker):
    evaluation_data = nltk.corpus.conll2000.chunked_sents('test.txt',
                                                          chunk_types=['NP'])
    return chunker.evaluate(evaluation_data)


def re_chunk(chunker, sentence):
    return chunker.parse(sentence)


def get_re_chunker(chunk_grammar):
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