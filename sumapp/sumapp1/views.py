#for comparison-ROUGE
from __future__ import division
from itertools import chain
#from pythonrouge.pythonrouge import Pythonrouge

from django.shortcuts import render, HttpResponse,redirect
from django.http import HttpResponse

from django.contrib import messages
from django.contrib.auth.models import User
from . models import TextInfo
from django.contrib.auth import authenticate,login,logout

import nltk

from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


import math

def generatesum(request):
    if request.method == 'POST':
        original = request.POST['original']
        maxnum=request.POST['maxnum']
        top_n=int(maxnum)
        stop_words = stopwords.words('english')
        summarize_text = []

        ##TF-IDF
        # 1 Sentence Tokenize
        sentences = sent_tokenize(original)
        total_documents = len(sentences)
        #print(sentences)

        # 2 Create the Frequency matrix of the words in each sentence.
        freq_matrix = _create_frequency_matrix(sentences)
        #print(freq_matrix)

        '''
        Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
        '''
        # 3 Calculate TermFrequency and generate a matrix
        tf_matrix = _create_tf_matrix(freq_matrix)
        #print(tf_matrix)

        # 4 creating table for documents per words
        count_doc_per_words = _create_documents_per_words(freq_matrix)
        #print(count_doc_per_words)

        '''
        Inverse document frequency (IDF) is how unique or rare a word is.
        '''
        # 5 Calculate IDF and generate a matrix
        idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
        #print(idf_matrix)

        # 6 Calculate TF-IDF and generate a matrix
        tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
        #print(tf_idf_matrix)

        # 7 Important Algorithm: score the sentences
        sentence_scores = _score_sentences(tf_idf_matrix)
        #print(sentence_scores)

        # 8 Find the threshold
        threshold = _find_average_score(sentence_scores)
        #print(threshold)

        # 9 Important Algorithm: Generate the summary
        TFSummary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)


        #TextRank
        # Step 1 - Read text and split it
        sentences = read_article(original)

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

        rankSummary = ". ".join(summarize_text)

        u=request.user
        tuser = TextInfo.objects.create()
        tuser.otext = original
        tuser.stextrank = rankSummary
        tuser.stfidf = TFSummary
        tuser.uid=u.email
        tuser.save()
        
        return render(request,'summarypage.html',{'rankSummary':rankSummary, 'TFIDFSummary':TFSummary})

##TF-DF
#function for step1
def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


#function for step1
def read_article(file_name):
    article = file_name.split(".")
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    return sentences

#function for step2
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

#subfunction for step2
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)



#human generated summary
def humansum(request):
    return render(request,'humansum.html')







##ROUGE
def get_unigram_count(tokens):
    count_dict = dict()
    for t in tokens:
        if t in count_dict:
            count_dict[t] += 1
        else:
            count_dict[t] = 1

    return count_dict


class Rouge:
    beta = 1

    @staticmethod
    def my_lcs_grid(x, y):
        n = len(x)
        m = len(y)

        table = [[0 for i in range(m + 1)] for j in range(n + 1)]

        for j in range(m + 1):
            for i in range(n + 1):
                if i == 0 or j == 0:
                    cell = (0, 'e')
                elif x[i - 1] == y[j - 1]:
                    cell = (table[i - 1][j - 1][0] + 1, '\\')
                else:
                    over = table[i - 1][j][0]
                    left = table[i][j - 1][0]

                    if left < over:
                        cell = (over, '^')
                    else:
                        cell = (left, '<')

                table[i][j] = cell

        return table

    @staticmethod
    def my_lcs(x, y, mask_x):
        table = Rouge.my_lcs_grid(x, y)
        i = len(x)
        j = len(y)

        while i > 0 and j > 0:
            move = table[i][j][1]
            if move == '\\':
                mask_x[i - 1] = 1
                i -= 1
                j -= 1
            elif move == '^':
                i -= 1
            elif move == '<':
                j -= 1

        return mask_x

    @staticmethod
    def rouge_l(cand_sents, ref_sents):
        lcs_scores = 0.0
        cand_unigrams = get_unigram_count(chain(*cand_sents))
        ref_unigrams = get_unigram_count(chain(*ref_sents))
        for cand_sent in cand_sents:
            cand_token_mask = [0 for t in cand_sent]
            cand_len = len(cand_sent)
            for ref_sent in ref_sents:
                # aligns = []
                # Rouge.lcs(ref_sent, cand_sent, aligns)
                Rouge.my_lcs(cand_sent, ref_sent, cand_token_mask)

                # for i in aligns:
                #     ref_token_mask[i] = 1
            # lcs = []
            cur_lcs_score = 0.0
            for i in range(cand_len):
                if cand_token_mask[i]:
                    token = cand_sent[i]
                    if cand_unigrams[token] > 0 and ref_unigrams[token] > 0:
                        cand_unigrams[token] -= 1
                        ref_unigrams[token] -= 1
                        cur_lcs_score += 1

                        # lcs.append(token)

            # print ' '.join(lcs)

            lcs_scores += cur_lcs_score

        # print "lcs_scores: %d" % lcs_scores
        ref_words_count = sum(len(s) for s in ref_sents)
        # print "ref_words_count: %d" % ref_words_count
        cand_words_count = sum(len(s) for s in cand_sents)
        # print "cand_words_count: %d" % cand_words_count

        precision = lcs_scores / cand_words_count
        recall = lcs_scores / ref_words_count
        f_score = (1 + Rouge.beta ** 2) * precision * recall / (recall +
                                                                Rouge.beta ** 2 * precision + 1e-7) + 1e-6  # prevent underflow
        return precision, recall, f_score

def savehumansum(request):
    if request.method == 'POST':
        email = None
        if request.user.is_authenticated:
            email = request.user.email
        humansum=request.POST['human']
        r = Rouge()
        
        obj=TextInfo.objects.filter(uid=email)
        obj1=obj.order_by('uid').first()

        system_generated_summary_textrank =obj1.stextrank
        system_generated_summary_tfidf = obj1.stfidf

        
        manual_summmary = humansum
        [precision, recall, f_score] = r.rouge_l([system_generated_summary_textrank], [manual_summmary])
        [precision1, recall1, f_score1] = r.rouge_l([system_generated_summary_tfidf], [manual_summmary])

        #print("Precision is :"+str(precision)+"\nRecall is :"+str(recall)+"\nF Score is :"+str(f_score))
        return render(request,'dispcompare.html',{'precision':str(precision), 'recall':str(recall), 'f_score':str(f_score),'precision1':str(precision1), 'recall1':str(recall1), 'f_score1':str(f_score1)})





#views here.

def index(request):
    return render(request,'index.html')

def handleLog(request):
    return render(request,'log.html')
def handleSign(request):
    return render(request,'sign.html')


def handleLogin(request):
    if request.method=='POST':
        #get parameters
        loginusername=request.POST['loginusername']
        loginpass=request.POST['loginpass']
        #use built in authenticate
        user=authenticate(username=loginusername, password=loginpass)
        if user is not None:
            login(request,user)
            messages.success(request,"Successfully logged in")
            return render(request,"login.html")
        else:
            messages.error(request,"Invalid credentials...please try again")
            return redirect("index")
    else:
        return HttpResponse("404-not Allowed Login")

def handleLogout(request):
    logout(request)
    messages.success(request,"Successfully logged out")
    return redirect("index")


def handleSignup(request):
    if request.method=='POST':
        #get parameters
        username=request.POST['username']
        fname=request.POST['fname']
        lname=request.POST['lname']
        email=request.POST['email']
        pass1=request.POST['pass1']
        pass2=request.POST['pass2']
        #check for errorneous inputs
        if not username.isalnum():
            messages.error(request,"username must should only contain letters and numbers")
            return redirect('index')
        if len(username)>10:
            messages.error(request,"username must be under 10 char")
            return redirect('index')
        if pass1!=pass2:
            messages.error(request,"Passwords do not match")
            return redirect('index')
        # create user
        myuser=User.objects.create_user(username,email,pass1)
        myuser.first_name=fname
        myuser.last_name=lname
        myuser.save()
        messages.success(request,"account created")
        return redirect('index')

    else:
        return HttpResponse("404-not Allowed Signup")

def about(request):
    return render(request,'aboutpage.html')

def logsum(request):
    return render(request,'login.html')

