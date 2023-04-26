from more_itertools import chunked
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import SnowballStemmer,WordNetLemmatizer
import nltk

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else: 
        return wordnet.NOUN

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
# string = input('Enter a sentence : ')
string = """
The crew of the USS Discovery discovered many discoveries.
Discovering is what explorers do.
"""
# string = 'The friends of DeSoto love scarves.'
words_of_sentence = word_tokenize(string)

stop_words = set(stopwords.words('english'))

filtered_words = [word for word in words_of_sentence if word.casefold() not in stop_words]

stemmed_words = [stemmer.stem(word) for word in words_of_sentence]

pos_tag_list = nltk.pos_tag(words_of_sentence)

lemmatized_words = [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in words_of_sentence]

grammer = 'NP: {<DT>?<JJ>*<NN>}'
chunk_parser = nltk.RegexpParser(grammer)

tree = chunk_parser.parse(pos_tag_list)

print(filtered_words)
print(stemmed_words)
print(pos_tag_list)
print(lemmatized_words)
tree.draw()