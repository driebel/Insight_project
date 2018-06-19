#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:13:52 2018
#keep up the good work!
@author: driebel
"""
import numpy as np
import matplotlib as plt
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas.io.sql as psql
import re

def token_found(response, token):
    if response.find(token) >= 0:
        return True
    else:
        return False



username = 'postgres'
password = 'SQLpassword'     # change this
host     = 'localhost'
port     = '5432'            # default port that postgres listens on
db_name  = 'full_pulse_db'

engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name) )
# print(engine.url)

if not database_exists(engine.url):
    create_database(engine.url)
# print(database_exists(engine.url))


con = psycopg2.connect(database = db_name,
                       user = username,
                       host = host,
                       password = password)

answer_df = psql.read_sql('SELECT * FROM answer_db', con)
question_df = psql.read_sql('SELECT * FROM question_db',con)
file_name = '/home/driebel/Dropbox/Insight/insight_project/data/full_data_table.csv'
full_data = pd.read_csv(file_name,sep='*',low_memory=False)
full_data.set_index('user_id', inplace=True)
number_of_students_who_answered = full_data.count(axis=0)
unanswered_questions = np.where(number_of_students_who_answered == 0)
cols_to_drop = full_data.columns[unanswered_questions]
colab_filter_df = full_data.copy(deep=True)
colab_filter_df = colab_filter_df.drop(columns=cols_to_drop)

# multiple_choice_question_list = answer_df[answer_df.type=='mc'].question_id
# Eventually, right now work with a tiny tiny subset of questions:
multiple_choice_question_list = ['5b2165535fc890000376de3e'] # hard coded for now


for q in multiple_choice_question_list:
    all_used_answers = full_data.loc[:,q].unique() # Gather all combinatorics of answers
    bag_of_tokens = []
    for each_answer in all_used_answers:
        if type(each_answer)==str:
            for each_choice in each_answer.split(','):
                bag_of_tokens.append(each_choice) 
    # bag of tokens is every token used, every time it's used
    unique_tokens = list(set(bag_of_tokens))
    #find the tokens only used once, toss them
    empty_answer = r'^[ \t\r\n]*$'
    for solo_token in unique_tokens:
        if ((bag_of_tokens.count(solo_token) <= 1) | 
                re.match(empty_answer, solo_token)):
            bag_of_tokens.remove(solo_token)
    unique_tokens = list(set(bag_of_tokens))
    number_of_tokens = len(unique_tokens)
    new_question_ids = []
    for i, each_token in enumerate(number_of_tokens):
        colab_filter_df[q+'_a'+str(i)] = pd.Series([0 for x in 
                        range(len(colab_filter_df.index))],
                        index=colab_filter_df.index)
        
        
        valid_answers = full_data[full_data.loc[:,q].notnull()].loc[:,q]
        
        for answer in valid_answers:
            if token_found(answer,each_token):
                
        




n, bins, patches = plt.hist(number_of_students_who_answered, facecolor='blue', alpha=0.5)
plt.show() 

number_of_questions_answered = full_data.count(axis=1)

all_multiple_choice_entries=answer_df[answer_df.type=='mc'].response.unique()


# Maybe, loop over everey question, loop over all unique answers, split them based on commas, 
# take the set of the splits, which should be all uniue choices,
# turn these into new columns



