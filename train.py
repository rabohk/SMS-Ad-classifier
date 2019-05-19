# transcation(notSpam) classifier
import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

def getDict():
    spamFile = "dataset/spam.txt"
    notSpamFile = "dataset/notSpam.txt"

    words = []

    f = open(spamFile)
    blob = f.read().lower()
    words += blob.split(" ")
    f = open(notSpamFile)
    blob = f.read().lower()
    words += blob.split(" ")

    for i in range(len(words)):
        if(not words[i].isalpha() or words[i] is "#"):
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]

    # print(words)
    # print(len(words))

    # print(dictionary.most_common(1000))
    # print(len(dictionary.most_common(1000)))

    return dictionary.most_common(1000)

def makeDataSet(dictionary):

    spamFile = "dataset/spam.txt"
    notSpamFile = "dataset/notSpam.txt"

    X = []
    y = []

    f = open(spamFile)
    blob = f.read().lower()
    sms = blob.split("#")
    for msg in sms:
        vec = []
        words = msg.split(" ")
        for entry in dictionary:
            vec.append(words.count(entry[0]))
        X.append(vec)
        y.append(0)
    
    f = open(notSpamFile)
    blob = f.read().lower()
    sms = blob.split("#")
    for msg in sms:
        vec = []
        words = msg.split(" ")
        for entry in dictionary:
            vec.append(words.count(entry[0]))
        X.append(vec)
        y.append(1)
    
    return X,y

d = getDict()
X, y = makeDataSet(d)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clas = RandomForestClassifier(n_estimators=10,criterion='entropy')
clas.fit(X_train, y_train)

y_pred=clas.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("acc = "+str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[1][0]+cm[0][1])*100))
print(cm)

while(True):
    msg = input("enter a message: ")
    if "exit" in msg:
        break
    vec = []
    words = msg.split(" ")
    for entry in d:
        vec.append(words.count(entry[0]))

    print(clas.predict([vec]))