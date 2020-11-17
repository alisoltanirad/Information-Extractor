# https://github.com/alisoltanirad/Information-Extractor.git
# Dependencies: nltk
import ssl
import nltk

class BigramChunker(nltk.ChunkParserI):

    def __init__(self, train_sentences):
        train_data = [[(pos, chunk)
                       for _, pos, chunk in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sentences]
        self.tagger = nltk.BigramTagger(train_data)


    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        pos_chunk_tags = self.tagger.tag(pos_tags)
        chunk_tags = [chunk for (pos, chunk) in pos_chunk_tags]
        conll_tags = [(word, pos, chunk)
                      for ((word, pos), chunk) in zip(sentence, chunk_tags)]
        return nltk.chunk.conlltags2tree(conll_tags)


class ConsecutiveChunker(nltk.ChunkParserI):

    def __init__(self, train_sentences):
        train_data = [[((word, pos), chunk)
                       for word, pos, chunk in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sentences]
        self.tagger = ConsecutiveTagger(train_data)


    def parse(self, sentence):
        tagged_sentence = self.tagger.tag(sentence)
        conll_tags = [(word, pos, chunk)
                      for ((word, pos), chunk) in tagged_sentence]
        return nltk.chunk.conlltags2tree(conll_tags)


class ConsecutiveTagger(nltk.TaggerI):

    def __init__(self, train_data):
        train_set = []
        for tagged_sentence in train_data:
            sentence = nltk.tag.untag(tagged_sentence)
            history = []
            for i, (word, tag) in enumerate(tagged_sentence):
                features = self.__get_features(sentence, i)
                train_set.append((features, tag))
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)


    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            features = self.__get_features(sentence, i)
            tag = self.classifier.classify(features)
            history.append(tag)
        return zip(sentence, history)


    def __get_features(self, sentence, i):
        features = {}

        word, pos = sentence[i]
        features['word'] = word
        features['pos'] = pos

        if i == 0:
            previous_word, previous_pos = '_', '<START>'
        else:
            previous_word, previous_pos = sentence[i-1]
        features['previous_pos'] = previous_pos

        if i == len(sentence)-1:
            next_word, next_pos = '_', '<END>'
        else:
            next_word, next_pos = sentence[i+1]
        features['next_pos'] = next_pos

        features['previous_pos+pos'] = '%s+%s' % (previous_pos, pos)
        features['pos+next_pos'] = '%s+%s' % (pos, next_pos)
        features['tags_since_dt'] = self.__get_tags_since_dt(sentence, i)

        return features


    def __get_tags_since_dt(self, sentence, i):
        tags = set()
        for word, pos in sentence[:i]:
            if pos == 'DT':
                tags = set()
            else:
                tags.add(pos)
        return '+'.join(sorted(tags))


class RegexpChunker():

    def __init__(self):
        self.parser = nltk.RegexpParser(self.__get_chunk_grammar(), loop=2)
        self.evaluate = self.parser.evaluate


    def parse(self, sentence):
        return self.parser.parse(sentence)


    def __get_chunk_grammar(self):
        grammar = r'''
        NP: {<DT|JJ|NN.*>+}           # Noun Phrase
        PP: {<IN><NP>}                # Prepositional Phrase
        VP: {<VB.*><NP|PP|CLAUSE>+$}  # Verb Phrase
        CLAUSE: {<NP><VP>}            # Sentence
        '''
        return grammar


def main():
    download_resources()


def get_data_set():
    train = nltk.corpus.conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    test = nltk.corpus.conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    return train, test


def preprocess_text(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    sentences = [nltk.pos_tag(sentence) for sentence in sentences]
    return sentences


def download_resources():
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('averaged_perceptron_tagger')
    nltk.download('conll2000')


if __name__ == '__main__':
    main()