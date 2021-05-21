import numpy as np
import nltk
import string
import random

file = open('data.txt','r',errors ='ignore')

test_doc = file.read()
test_doc = test_doc.lower()

nltk.download('punkt')
nltk.download('wordnet')

sent_tokens = nltk.sent_tokenize(test_doc)
word_tokens = nltk.word_tokenize(test_doc)


result = sent_tokens[:4]
stem = word_tokens[:5]

# print(result)
# print(stem)


# TEXT-PROCESSING

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
	return(lemmer.lemmatize(token) for token in tokens)

remove_puct = dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_puct)))

# Input of greeting by the bot and to the bot

inp_greet = ["hi","hello","hey","wassup","hey Bruce"]
greet_resp= ["Hi","Hey there","How are you?","Good to see you"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in inp_greet:
            return random.choice(greet_resp)


# RESPONSE GENEREATOR

from sklearn.feature_extraction.text import TfidfVectorizer #Basically this define how many times a word repeated and how many times a unique word repeats

from sklearn.metrics.pairwise import cosine_similarity #This mainly converts the vectorized data into machine-level language

def response(user_resp):
    robo_resp =''
    Tfidvec = TfidfVectorizer(tokenizer = LemNormalize , stop_words='english')
    tfidfreq = Tfidvec.fit_transform(sent_tokens)

    val = cosine_similarity(tfidfreq[-1],tfidfreq)
    matr = val.argsort()[0][-2]
    flat = val.flatten()
    flat.sort()

    req_tfreq = flat[-2]

    if(req_tfreq==0):
        robo_resp = robo_resp+"I am sorry didn't get you"
        return robo_resp
    else:
        robo_resp = robo_resp + sent_tokens[matr]
        return robo_resp

#MAIN-BODY OF BOT 

flag = True

print("Bot: Hello , myself Bruce nice to meet you! :)")
while(flag):
    user_resp = input("ME:")
    user_resp = user_resp.lower()
    if(user_resp!='bye'):
        if(user_resp=='thanks <3' or user_resp == 'Thank you'):
            flag = False
            print("Bot : MOst Welcome!!<3")

        else:
            if(greet(user_resp)!=None):
                print("Bot: "+greet(user_resp))
            else:
                sent_tokens.append(user_resp)
                word_tokens = word_tokens + nltk.word_tokenize(user_resp)
                final_list = list(set(word_tokens))
                print("Bot: ",end="")
                print(response(user_resp))
                sent_tokens.remove(user_resp)
    else:
        flag = False
        print("Bot : Gooday! Take Care")