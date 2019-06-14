from flask import Flask, render_template, request
import os
import json
from nltk.tokenize import word_tokenize
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.corpora.dictionary import Dictionary
factory = StemmerFactory()
stemmer = factory.create_stemmer()
import pickle as pk

app = Flask(__name__)

@app.route('/')
def index():
    data={
    'active':'koleksi',
    'listdir':[item.replace('.txt','') for item in os.listdir('lirik')]
    }
    return render_template('index.html',data=data)

@app.route('/koleksi')
def koleksi():
    data={
    'active':'koleksi',
    'listdir':os.listdir('lirik')
    }
    # return render_template('koleksi.html',data=data)
    m=pk.load(open("support/model_lda_20t[100c].pkl","rb"))
    return json.dumps(m.print_topics()).replace("],", '],<br>')

@app.route('/proses', methods=['GET','POST'])
def proses():
    if request.method=='POST':
        judul = request.form['judul']
        lirik = request.form['lirik']
        type = request.form['type']
        if(type!=''):
            if(type=='pilih'):
                f_inp=open("lirik/"+judul+'.txt')
                raw_inp=f_inp.read()
                isi = raw_inp.replace("\n", '<br>')
                raw_inp=raw_inp.replace("\n",' ')
                f_inp.close()
            else:
                judul="Lirk Lagu Baru"
                isi=lirik
                raw_inp=lirik
            token = tokenisasi(raw_inp)
            stopwords_remove=stopwords(token)
            ekstraksi=ekstraksiva(stopwords_remove)
            ekstraksi=[[round(item[0],3),round(item[1],3)] for item in ekstraksi]
            vawords=bovaw(ekstraksi)
            dictionary=pk.load(open('support/dictionary.pkl','rb'))
            bow=dictionary.doc2bow(vawords)
            predict=prediction(bow)
            predict=[str(item) for item in predict]
            return json.dumps({'judul':judul,'isi':isi,'tokenisasi':token,'stopwords':stopwords_remove,'ekstraksiva':ekstraksi,'vawords':vawords,'bow':bow, 'prediction':predict})
        else:
            return json.dumps({'msg':'Invalid POST'})
    else:
        return json.dumps({'msg':'Invalid POST'})

def tokenisasi(var):
    return word_tokenize(var.lower())

def stopwords(text_tokenized):
    list_stopwords=pd.read_csv("support/stopword_bahasa_indonesia.csv",header=None)[0].tolist()
    word_non_stopwords=list()
    alphabet= re.compile('^[a-z-]+$')
    for word in text_tokenized:
        word = re.sub(r'(.)\1+', r'\1\1', word)
        if(len(word)>2):
            if word not in list_stopwords:
                if(alphabet.match(word)):
                    word_non_stopwords.append(word)
    return word_non_stopwords

def ekstraksiva(text):
    file_json=list()
    file = text
    va_table=pd.read_excel('support/VA_table_sinonim2.xlsx')
    daftar_kata=va_table.iloc[:,1].tolist()
    # print(daftar_kata)
    i=0
    while (i<len(file)):
        if(i < len(file)-1):
            bigram=file[i]+' '+file[i+1]
            if(bigram in daftar_kata):
                file_json.append([va_table[va_table["kata"]==bigram]["arousal"].tolist()[0],va_table[va_table["kata"]==bigram]["valance"].tolist()[0]])
                i+=2
            else:
                stemm_bigram=stemmer.stem(file[i])+' '+stemmer.stem(file[i+1])
                if(stemm_bigram in daftar_kata):
                    file_json.append([va_table[va_table["kata"]==stemm_bigram]["arousal"].tolist()[0],va_table[va_table["kata"]==stemm_bigram]["valance"].tolist()[0]])
                    i+=2
                else:
                    if(file[i] in daftar_kata):
                        file_json.append([va_table[va_table["kata"]==file[i]]["arousal"].tolist()[0],va_table[va_table["kata"]==file[i]]["valance"].tolist()[0]])
                    else:
                        stemmed=stemmer.stem(file[i])
                        if (stemmed in daftar_kata):
                            file_json.append([va_table[va_table["kata"]==stemmed]["arousal"].tolist()[0],va_table[va_table["kata"]==stemmed]["valance"].tolist()[0]])
                    i+=1
        else:
            if(file[i] in daftar_kata):
                file_json.append([va_table[va_table["kata"]==file[i]]["arousal"].tolist()[0],va_table[va_table["kata"]==file[i]]["valance"].tolist()[0]])
            else:
                stemmed=stemmer.stem(file[i])
                if (stemmed in daftar_kata):
                    file_json.append([va_table[va_table["kata"]==stemmed]["arousal"].tolist()[0],va_table[va_table["kata"]==stemmed]["valance"].tolist()[0]])
            i+=1
    return file_json

def bovaw(text):
    model=pk.load(open("support/model_kmeans_100c.pkl","rb"))
    rename=[]
    for i in range(100):
        rename.append("kata"+str(i))

    out=list()
    if(text=='' or len(text)==0):
        return False
    else:
        cluster = model.predict(text)
#         return text[9]
        for c in cluster:
            out.append(rename[c])
        return out

def prediction(bow):
    m=pk.load(open("support/model_lda_20t[100c].pkl","rb"))
    predict=m.get_document_topics(bow)
    label_topik=["Annoying","Glad","Glad","Happy","Annoying","Depressing",
             "Peaceful","Relax","Peaceful","Happy","Relax","Happy","Glad",
             "Sad","Relax","Angry","Happy","Sad","Glad","Glad"]
    label_emosi=["Happy","Glad","Angry","Annoying","Sad","Depressing","Relax","Peaceful"]
    data_emosi =[0 for x in range(8)]
    data=[0 for x in range(20)]
    for item in predict:
        data[item[0]]=item[1]
    data_emosi[0]=max([data[3],data[9],data[11],data[16]])
    data_emosi[1]=max([data[1],data[2],data[12],data[18],data[19]])
    data_emosi[2]=data[15]
    data_emosi[3]=max([data[0],data[4]])
    data_emosi[4]=max([data[17],data[13]])
    data_emosi[5]=data[5]
    data_emosi[6]=max([data[7],data[10],data[14]])
    data_emosi[7]=max([data[8],data[6]])
    # data_emosi[0]=data[3]+data[9]+data[11]+data[16]
    # data_emosi[1]=data[1]+data[2]+data[12]+data[18]+data[19]
    # data_emosi[2]=data[15]
    # data_emosi[3]=data[0]+data[4]
    # data_emosi[4]=data[17]+data[13]
    # data_emosi[5]=data[5]
    # data_emosi[6]=data[7]+data[10]+data[14]
    # data_emosi[7]=data[8]+data[6]
    return data_emosi

if __name__ == "__main__":
    app.run(debug=True)
