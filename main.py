#coding:utf-8
import os
import jieba
import time
from numpy import *

page = 0

#分词
def wordSegmentation(folderName):
    global page
    page += 1
    start = time.time()
    path = folderName + "/"
    wordsList = []
    for i in range(0, 8000):
        if i % 500 == 0:
            print "截至" +  str(page)  + "---" + str(i) + "耗时:"
            print time.time() - start
        filePath = path + str(i) + ".txt"
        filePath = os.path.normcase(filePath)
        f = open(filePath)
        text = f.read()
        words = list(jieba.cut(text))
        for word in words:
            wordsList.append(word)
        f.close()
    return wordsList

#词列表
def loadDataSet():
    postingList = [
        wordSegmentation("C000007"),
        wordSegmentation("C000008"),
        wordSegmentation("C000010"),
        wordSegmentation("C000013"),
        wordSegmentation("C000014"),
        wordSegmentation("C000016"),
        wordSegmentation("C000020"),
        wordSegmentation("C000022"),
        wordSegmentation("C000023"),
        wordSegmentation("C000024")
    ]
    classVec = [
        "汽车",
        "财经",
        "IT",
        "健康",
        "体育",
        "旅游",
        "教育",
        "招聘",
        "文化",
        "军事"
    ]
    return postingList, classVec
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print "the word: %s is not in my Vocabulary!" %word
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p2Num = ones(numWords)
    p3Num = ones(numWords)
    p4Num = ones(numWords)
    p5Num = ones(numWords)
    p6Num = ones(numWords)
    p7Num = ones(numWords)
    p8Num = ones(numWords)
    p9Num = ones(numWords)

    p0Denom = 2.0
    p1Denom = 2.0
    p2Denom = 2.0
    p3Denom = 2.0
    p4Denom = 2.0
    p5Denom = 2.0
    p6Denom = 2.0
    p7Denom = 2.0
    p8Denom = 2.0
    p9Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == "汽车":
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "财经":
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "IT":
            p2Num += trainMatrix[i]
            p2Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "健康":
            p3Num += trainMatrix[i]
            p3Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "体育":
            p4Num += trainMatrix[i]
            p4Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "旅游":
            p5Num += trainMatrix[i]
            p5Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "教育":
            p6Num += trainMatrix[i]
            p6Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "招聘":
            p7Num += trainMatrix[i]
            p7Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "文化":
            p8Num += trainMatrix[i]
            p8Denom += sum(trainMatrix[i])
        elif trainCategory[i] == "军事":
            p9Num += trainMatrix[i]
            p9Denom += sum(trainMatrix[i])
    p0Vect = log(p0Num/p0Denom)
    p1Vect = log(p1Num/p1Denom)
    p2Vect = log(p2Num/p2Denom)
    p3Vect = log(p3Num/p3Denom)
    p4Vect = log(p4Num/p4Denom)
    p5Vect = log(p5Num/p5Denom)
    p6Vect = log(p6Num/p6Denom)
    p7Vect = log(p7Num/p7Denom)
    p8Vect = log(p8Num/p8Denom)
    p9Vect = log(p9Num/p9Denom)
    return p0Vect,p1Vect,p2Vect,p3Vect,p4Vect,p5Vect,p6Vect,p7Vect,p8Vect,p9Vect

def classifyNB(vec2Classify,p0Vec,p1Vec,p2Vec,p3Vec,p4Vec,p5Vec,p6Vec,p7Vec,p8Vec,p9Vec):
    p0 = sum(vec2Classify * p0Vec)
    p1 = sum(vec2Classify * p1Vec)
    p2 = sum(vec2Classify * p2Vec)
    p3 = sum(vec2Classify * p3Vec)
    p4 = sum(vec2Classify * p4Vec)
    p5 = sum(vec2Classify * p5Vec)
    p6 = sum(vec2Classify * p6Vec)
    p7 = sum(vec2Classify * p7Vec)
    p8 = sum(vec2Classify * p8Vec)
    p9 = sum(vec2Classify * p9Vec)
    p = [p0,p1,p2,p3,p4,p5,p6,p7,p8,p9]
    a = sorted(p, reverse=True)
    i = p.index(a[0])
    print i
    return i
def testingNB():
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    writeToTxt(myVocabList, "myVocabList.txt")
    trainMat = []
    for postInDoc in listOfPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, postInDoc))
    writeToTxt(trainMat, "trainMat.txt")

def writeToTxt(list_name,file_path):
    try:
        fp = open(file_path,"w+")
        for item in list_name:
            fp.write(str(item)+"\n")
        fp.close()
    except IOError:
        print("fail to open file")


testingNB()