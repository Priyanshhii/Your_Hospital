from flask import Flask, request, jsonify, Response
from firebase_admin import credentials, db
import firebase_admin
from cryptography.fernet import Fernet
import json
import numpy as np
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer
from sklearn import svm

nltk.download("punkt")
app = Flask(__name__)
Stemmer = LancasterStemmer()
cred = credentials.Certificate("Servicekey.json")
default_app = firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://yourhospital-5d8a7-default-rtdb.asia-southeast1.firebasedatabase.app/"
    },
)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [Stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


@app.route("/predict", methods=["GET"])
def chat():
    gender = request.args["gender"]
    age = request.args["age"]
    inp = request.args["symptoms"]
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

    try:
        model = pickle.load("model.pkl", "rb")
    except:
        model = svm.SVC()
        model.fit(training, output)
        pickle.dump(model, open("model.pkl", "wb"))

    resp = Response(labels[model.predict([bag_of_words(inp, words)])[0]])
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify(["hello"])


@app.route("/register", methods=["GET"])
def register():
    key = Fernet.generate_key()
    fernet = Fernet(key)
    email = request.args["email"]
    username = request.args["username"]
    password = request.args["password"]
    gender = request.args["gender"]
    encryption = fernet.encrypt(password.encode())
    encryption = key + encryption
    print(len(key))
    ref = db.reference("/users")
    if ref.get() == None:
        structure = {
            "password": encryption.decode(),
            "username": username,
            "gender": gender,
        }
        ref.set({email: structure})
    else:
        data = ref.get()
        if email not in data.keys():
            structure = {
                "password": encryption.decode(),
                "username": username,
                "gender": gender,
            }
            data[email] = structure
            ref.update(data)
        else:
            resp = Response("user found")
            resp.headers["Access-Control-Allow-Origin"] = "*"
            return resp
    resp = Response("success")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def retrive():
    ref = db.reference("/users")
    data = ref.get()
    for key in data.keys():
        encoded = data[key]["password"].encode()
        fernet_key = encoded[:44]
        fernet = Fernet(fernet_key)
        data[key]["password"] = fernet.decrypt(encoded[44:]).decode()

    return data


@app.route("/login")
def login():
    data = retrive()
    email = request.args["email"]
    password = request.args["password"]
    found = False
    for key in data:
        if key == email:
            if password == data[key]["password"]:
                found = True
    resp = jsonify(found)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/response", methods=["GET"])
def response():
    name = request.args["name"]
    email = request.args["email"]
    message = request.args["message"]

    ref = db.reference("/response")

    if ref.get() == None:
        ref.set({"responses": [{"name": name, "email": email, "message": message}]})
    else:

        data = ref.get()
        data["responses"].append({"name": name, "email": email, "message": message})
        print(data)
        ref.update(data)
        print(data)
    resp = jsonify("done")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    app.run(port=8080)
    chat()
# one hot encoding function
