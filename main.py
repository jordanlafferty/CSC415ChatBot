# packages needed
import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer
import tensorflow as tensorF
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

# botData essentially acts as a json file/python dictionary
botData = {"intents": [

    {"tag": "table",
     "patterns": ["table", "I would like a table", "Can I make a reservation?", "reservation",
                  "make a reservation"],
     "responses": ["Bot: If you want to be added to the waiting list, type how many people are in your party"
                   ", otherwise type 'no'"]
     },
    {"tag": "waiting",
     "patterns": ["See waiting list", "people waiting", "available", "how long is the wait", "reservations",
                  "make a reservation"],
     "responses": ["Bot: Here is the current waiting list: "]
     },
    {"tag": "greeting",
     "patterns": ["Hi", "Hello", "Hey", "Sup", "What's up?"],
     "responses": ["Bot: Hi there, how may I help you?",
                   "Bot: Welcome to Computer Science Cafe, what can I do to help?",
                   "Bot: Hi, what can I do for you?"],
     },
    {"tag": "goodbye",
     "patterns": ["bye", "later", "done", "exit", "finished", "see ya", "no"],
     "responses": ["Bot: Bye, have a good day!", "Bot: Take care!"]
     },
    {"tag": "menu",
     "patterns": ["what's your menu?", "menu", "eat", "food"],
     "responses": ["Bot: Let me give you the menu: \nAPPETIZERS\n----------\nNachos $7\nSoft Pretzels $5"
                   "\nCheese Curds $6\n\nMEALS\n----------\nFish Tacos $10\nBurger $11\nCaesar Salad $7\n\n"
                   "DESSERTS\n----------\nIce Cream Sundae $5\n\nFor full menu visit computersciencecafe.com/menu"
                   "\n\nBot: Anything else I can help you with?"]
     },
    {
        "tag": "location",
        "patterns": ["where are you located?", "location", "directions", "where are you?", "address"],
        "responses": ["Bot: Our address is 1010 N AI Street\n\nBot: Anything else I can help you with?"]
    },
    {
        "tag": "hours",
        "patterns": ["when are you opened", "time", "hours", "closed", "what are your hours of operation"],
        "responses": ["Bot: Our hours are 11am to 11pm everyday, except we are closed on major holidays."]
    },
    {
        "tag": "order",
        "patterns": ["order", "can I buy food", "I would like to order", "take out", "buy", "delivery"],
        "responses": ["Bot: You can order carryout or delivery by calling (123)-456-7890 or by going to "
                      "computersciencecafe.com/order\n\nBot: Anything else I can help you with?"]
    }

]}

# lists of everything that will make the bot be able to detect patterns and match them with the tags
getWords = WordNetLemmatizer()
newWords = []
patternArr = []
tagClasses = []
tagArr = []
reservationList = []
waitingList = []

for intent in botData["intents"]:
    for pattern in intent["patterns"]:
        patternTokens = nltk.word_tokenize(pattern)  # splits up the words
        newWords.extend(patternTokens)  # add the new words to the pattern
        patternArr.append(pattern)
        tagArr.append(intent["tag"])

    if intent["tag"] not in tagClasses:
        tagClasses.append(intent["tag"])  # add each category by its tag

newWords = [getWords.lemmatize(word.lower()) for word in newWords if
            word not in string.punctuation]  # make all words lowercase
newWords = sorted(set(newWords))  # sorting words
tagClasses = sorted(set(tagClasses))  # sorting  tag classes

# this uses neural networks to train the bot
trainingData = []
outEmpty = [0] * len(tagClasses)

for index, doc in enumerate(patternArr):
    wordChoices = []
    text = getWords.lemmatize(doc.lower())
    for word in newWords:
        wordChoices.append(1) if word in text else wordChoices.append(0)
        # adds a one if the word appears and a zero if not

    outputRow = list(outEmpty)
    outputRow[tagClasses.index(tagArr[index])] = 1
    trainingData.append([wordChoices, outputRow])

random.shuffle(trainingData)
trainingData = num.array(trainingData, dtype=object)  # making the data an array after being shuffled

x = num.array(list(trainingData[:, 0]))  # train data
y = num.array(list(trainingData[:, 1]))  # train data part 2

iShape = (len(x[0]),)
oShape = len(y[0])
organizedData = Sequential()  # organizes the data

# dense gives the neural network an output layer and dropout prevents over-fitting
# activation function takes nodes' summed weight into the activation of the current node
organizedData.add(Dense(128, input_shape=iShape, activation="relu"))
organizedData.add(Dropout(0.5))
organizedData.add(Dense(64, activation="relu"))
organizedData.add(Dropout(0.3))
organizedData.add(Dense(oShape, activation="softmax"))

md = tensorF.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
print(md)
organizedData.compile(loss='categorical_crossentropy',
                      optimizer=md,
                      metrics=["accuracy"])

organizedData.fit(x, y, epochs=200, verbose=1)  # 200 epochs used to rerun data


# separates and adds new words for the chatbot to be able to work
# adds additional features

def ourText(text):
    newTokens = nltk.word_tokenize(text)
    newTokens = [getWords.lemmatize(word) for word in newTokens]
    return newTokens


# checking for words
def wordBag(input, vocab):
    newTokens = ourText(input)
    wordBank = [0] * len(vocab)
    for w in newTokens:
        for idx, word in enumerate(vocab):
            if word == w:
                wordBank[idx] = 1
    return num.array(wordBank)


def botResponse(checkList, fJson):
    # goes through and gets the correct response
    # if the person is making a reservation it is a special case that takes you through a different loop
    tag = checkList[0]
    categories = fJson["intents"]
    for i in categories:
        if i["tag"] == tag:
            if i["tag"] == "goodbye":
                result = random.choice(i["responses"])
                print(result)
                exit()
            elif i["tag"] == "table":
                result = random.choice(i["responses"])
                print(result)
                partySize = input("You: ")
                if partySize.lower() == 'no':
                    break
                print("Bot: Sounds good, can I get your name?")
                name = input("")
                print("Bot: " + name + " party of " + partySize + " was added to the list.")
                print("Bot: What else can I do for you?")
                reservation = "" + name + "-" + partySize
                waitingList.append(reservation)
                return None
            elif i["tag"] == "waiting":
                result = random.choice(i["responses"])
                print(result)
                if waitingList == []:
                    print("Bot: The waiting list is currently empty.")
                else:
                    print("Waiting List\n----------")
                    for i in waitingList:
                        print(i)
                print("Bot: What else can I do for you?")
                return None
            else:
                result = random.choice(i["responses"])
                break
    return result


def chatCreation(input, vocab, labels):
    wordBank = wordBag(input, vocab)
    ourResult = organizedData.predict(num.array([wordBank]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList


# creates a while loop to keep the chat going
# there is a special case for when someone wants to make a reservation
# which would return none in the bot response hence the if statement
def startChat():
    print("\n\n\n\n\n\nHello, I am Carl the Computer Science Cafe Bot.\nYou can make add your name to the reservation "
          "list, get our address, \nfind out when we are opened, \nlearn how to order food or get the menu!")
    while True:
        userMessage = input("You: ")
        intents = chatCreation(userMessage, newWords, tagClasses)
        answer = botResponse(intents, botData)
        if answer is not None:
            print(answer)


startChat()
