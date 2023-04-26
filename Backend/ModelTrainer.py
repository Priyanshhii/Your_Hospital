# importing modules
import json
import numpy as np
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer
from sklearn import svm

Stemmer = LancasterStemmer()

# one hot encoding function


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [Stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def chat(inp):
    result = model.predict([bag_of_words(inp, words)])
    max_result_index = np.argmax(result)

    tag = labels[max_result_index]
    return tag


# loading data
with open("intents.json") as json_data:
    data = json.load(json_data)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # creating variables
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # extracting all words possible and labels possible
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [Stemmer.stem(word.lower()) for word in words if word != "?"]

    words = sorted(list(set(words)))
    labels = sorted(labels)
    # creating training and testing data
    training = []
    output = []

    output_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        encoding = []
        wrds = [Stemmer.stem(w) for w in doc]

        for word in words:
            if word in wrds:
                encoding.append(1)
            else:
                encoding.append(0)

        out = output_empty[:]
        out[labels.index(docs_y[x])] = 1

        training.append(encoding)
        output.append(labels.index(docs_y[x]))
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
    pickle.dump(words, open("words.pkl", "wb"))

# print(output)
inp = "Sweating Chills and shivering Muscle aches"
model = svm.SVC()
model.fit(training, output)
print(labels[model.predict([bag_of_words(inp, words)])[0]])

# creating model
# with tf.Graph().as_default():

#     net = tflearn.input_data(shape=[None, len(training[0])])
#     net = tflearn.fully_connected(net, 8)
#     net = tflearn.fully_connected(net, 8)
#     net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
#     net = tflearn.regression(net)

#     present_files = os.listdir(os.getcwd())
#     # training model
#     model = tflearn.DNN(net)
#     if "checkpoint" in present_files:
#         model.load("chatbot_model.tflearn")
#     else:
#         model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
#         model.save("chatbot_model.tflearn")
#     pickle.dump(model, open("model.pkl", "wb"))
