# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 22:31:28 2018

@author: Vijay Gupta
"""

import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pickle import dump,load
from keras.preprocessing import image, sequence,text
from keras.applications import inception_v3
from keras.layers import Input,Dense, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from os import listdir
from keras.preprocessing.sequence import pad_sequences
from keras.layers.merge import add
from keras.utils import to_categorical,plot_model
from keras.callbacks import ModelCheckpoint
from numpy import array
import nltk
import string

path1='Flicker8k_text/Flickr8k.token.txt'
path2='Flicker8k_text/Flickr_8k.trainImages.txt'
path3='Flicker8k_text/Flickr_8k.devImages.txt'
path4='Flicker8k_text/Flickr_8k.testImages.txt'
caption=open(path1,'r').read().split("\n")
train_img=open(path2,'r').read().split("\n")
val_img=open(path3,'r').read().split("\n")
test_img=open(path4,'r').read().split("\n")
def extract_features(directory):
    model=InceptionV3()
    model.layers.pop()
    model=Model(inputs=model.inputs,outputs=model.layers[-1].output)
    features={}
    for img_name in listdir(directory):
        filename=directory+'/'+img_name
        img=image.load_img(filename,target_size=(299,299))
        img=image.img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=inception_v3.preprocess_input(img)
        feature=model.predict(img,verbose=0)
        img_id=img_name.split('.')[0]
        features[img_id]=feature
    return features
directory='Flicker8k_Dataset'
features=extract_features(directory)
dump(features,open('features.pkl','wb'))

def load_descriptions(caption):
    mapping=dict()
    for line in caption:
        token=line.split()
        if len(line)<2:
            continue
        img_id, img_desc=token[0], token[1:]
        img_id=img_id.split('.')[0]
        img_desc=' '.join(img_desc)
        if img_id not in mapping:
            mapping[img_id]=list()
        mapping[img_id].append(img_desc)
    return mapping
descriptions=load_descriptions(caption)


def cleaned_desc(descriptions):
    table=str.maketrans('','',string.punctuation)
    for key,desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc=desc_list[i]
            desc=desc.split()
            desc=[word.lower() for word in desc]
            desc=[w.translate(table) for w in desc]
            desc=[word for word in desc if len(word)>1]
            desc=[word for word in desc if word.isalpha()]
            desc_list[i]=' '.join(desc)
cleaned_desc(descriptions)

def to_vocabulary(descriptions):
    all_vocab=set()
    for key in descriptions.keys():
        [all_vocab.update(d.split()) for d in descriptions[key]]
    return all_vocab
vocabulary=to_vocabulary(descriptions)

def dict_to_list_text(descriptions):
    lines=list()
    for key,desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key+' '+desc)
    data='\n'.join(lines)
    return data
data=dict_to_list_text(descriptions)

def load_trainset(trainlist):
    dataset=list()
    for line in trainlist:
        if len(line)<1:
            continue
        identifier=line.split('.')[0]
        dataset.append(identifier)
    return dataset
train=load_trainset(train_img)
validation=load_trainset(val_img)
test=load_trainset(test_img)

def load_clean_desc(traindesc,dataset):
    descriptions=dict()
    for line in traindesc.split('\n'):
        token=line.split()
        img_id,img_desc=token[0],token[1:]
        if img_id in dataset:
            if img_id not in descriptions:
                descriptions[img_id]=list()
            desc='startseq '+' '.join(img_desc)+' endseq'
            descriptions[img_id].append(desc)
    return descriptions
train_descriptions=load_clean_desc(data,train)
validation_descriptions=load_clean_desc(data,validation)
test_descriptions=load_clean_desc(data,test)

def load_photo_features(file,dataset):
    pic_features=load(open(file,'rb'))
    features={k: pic_features[k] for k in dataset}
    return features
train_features=load_photo_features('features.pkl',train)
validation_features=load_photo_features('features.pkl',validation)
test_features=load_photo_features('features.pkl',test)
def to_lines(descriptions):
     desc_lines=list()
     for key in descriptions.keys():
         [desc_lines.append(d) for d in descriptions[key]]
     return desc_lines
 
def tokenization(descriptions):
    lines=to_lines(descriptions)
    tokenizer=text.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
tokenizer=tokenization(train_descriptions)
vocabulary_size=len(tokenizer.word_index)+1

def create_sequences(tokenizer,max_length,desc_list,photo):
    i1,i2,op=list(),list(),list()
    for desc in desc_list:
            seq=tokenizer.texts_to_sequences([desc])[0]
            for i in range(1,len(seq)):
                input_seq,output_seq=seq[:i],seq[i]
                input_seq=pad_sequences([input_seq],maxlen=max_length)[0]
                output_seq=to_categorical([output_seq],num_classes=vocabulary_size)[0]
                i1.append(photo)
                i2.append(input_seq)
                op.append(output_seq)
    return array(i1),array(i2),array(op)

def to_find_maxlen(descriptions):
    lines=to_lines(descriptions)
    return max(len(d.split()) for d in lines)
max_length=to_find_maxlen(train_descriptions)

def define_model(vocabulary_size,max_length):
    inputs1=Input(shape=(2048,))
    feature_ex1=Dropout(0.5)(inputs1)
    feature_ex2=Dense(256,activation='relu')(feature_ex1)
    inputs2=Input(shape=(max_length,))
    sequence_mo1=Embedding(vocabulary_size,256,mask_zero=True)(inputs2)
    sequence_mo2=Dropout(0.5)(sequence_mo1)
    sequence_mo3=LSTM(256)(sequence_mo2)
    decoder1=add([feature_ex2,sequence_mo3])
    decoder2=Dense(256,activation='relu')(decoder1)
    outputs=Dense(vocabulary_size,activation='softmax')(decoder2)
    model=Model(inputs=[inputs1,inputs2],outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    print(model.summary())
     #plot_model(model,to_file='model.png',show_shapes=True)
    return model

def data_generator(descriptions,photos,tokenizer,max_length):
    while 1:
        for key,desc_list in descriptions.items():
            photo=photos[key][0]
            in_img,in_seq,out_seq=create_sequences(tokenizer,max_length,desc_list,photo)
            yield [[in_img,in_seq],out_seq]
    


model=define_model(vocabulary_size,max_length)
epochs=20
steps=len(train_descriptions)
steps1=len(validation_descriptions)
for i in range(epochs):
    generator=data_generator(train_descriptions,train_features,tokenizer,max_length)
    generator1=data_generator(validation_descriptions,validation_features,tokenizer,max_length)
    model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=2,validation_data=generator1,validation_steps=steps1)
    model.save('model_'+str(i)+'.h5')
from keras.models import load_model
filename='model_19.h5'
model=load_model(filename)

def word_for_int_id(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None
from numpy import argmax,argsort

def generate_desc(model,tokenizer,photo,max_length):
    in_text='startseq'
    for i in range(max_length):
        seq=tokenizer.texts_to_sequences([in_text])[0]
        seq=pad_sequences([seq],maxlen=max_length)
        pred=model.predict([photo,seq],verbose=1)
        pred=argmax(pred)
        word=word_for_int_id(pred,tokenizer)
        if word is None:
            break
        in_text+=' ' + word
        if word=='endseq':
            break
    return in_text
from nltk.translate.bleu_score import corpus_bleu

def evaluate_model(model,descriptions,photos,tokenizer,max_length):
    actual,predicted=list(),list()
    for key,desc_list in descriptions.items():
        pred=generate_desc(model,tokenizer,photos[key],max_length)
        ref_desc=[desc.split() for desc in desc_list]
        actual.append(ref_desc)
        predicted.append(pred.split())
    print(corpus_bleu(actual,predicted,weights=(1.0,0,0,0)))
    print(corpus_bleu(actual,predicted,weights=(0.5,0.5,0,0)))
    print(corpus_bleu(actual,predicted,weights=(0.3,0.3,0.3,0)))
    print(corpus_bleu(actual,predicted,weights=(0.25,0.25,0.25,0.25)))
evaluate_model(model,test_descriptions,test_features,tokenizer,max_length)

def beam_search_predictions(model,tokenizer,photo,max_length,beam_index):
    in_text='startseq'
    start=tokenizer.texts_to_sequences([in_text])[0]
    start_word=[[start,0.0]]
    while len(start_word[0][0])<max_length:
        temp=[]
        for s in start_word:
            par_caps=pad_sequences([s[0]],maxlen=max_length)
            pred=model.predict([photo,par_caps],verbose=1)
            word_pred=argsort(pred[0])[-beam_index:]
            for w in word_pred:
                next_cap,prob=s[0][:],s[1]
                next_cap.append(w)
                prob+=pred[0][w]
                temp.append([next_cap,prob])
        start_word=temp
        start_word=sorted(start_word,reverse=False,key=lambda l: l[1])
        start_word=start_word[-beam_index:]
    start_word=start_word[-1][0]
    intermediate_caption=[word_for_int_id(i,tokenizer) for i in start_word]
    final_caption=[]
    for i in intermediate_caption:
        if i!='endseq':
            final_caption.append(i)
        else:
            break
    final_caption=' '.join(final_caption[1:])
    return final_caption

#Testing on 1 image            
def extract_features1(directory):
    model=InceptionV3()
    model.layers.pop()
    model=Model(inputs=model.inputs,outputs=model.layers[-1].output)
    filename=directory+'/'+'3695064885_a6922f06b2.jpg'
    img=image.load_img(filename,target_size=(299,299))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=inception_v3.preprocess_input(img)
    feature=model.predict(img,verbose=0)
    return feature
    
photo=extract_features1(directory)

des1=generate_desc(model,tokenizer,photo,max_length)
print(des1)
des2=beam_search_predictions(model,tokenizer,photo,max_length,3)
print(des2)
des3=beam_search_predictions(model,tokenizer,photo,max_length,5)
print(des3)
des4=beam_search_predictions(model,tokenizer,photo,max_length,7)
print(des4)
    
        
        
    