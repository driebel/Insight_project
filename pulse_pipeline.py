#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:42:19 2018

@author: driebel
"""

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import re
import pandas.io.sql as psql
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD, evaluate
from surprise import accuracy
from surprise.model_selection import train_test_split


def will_it_float(s):
# =============================================================================
# Fucntion to determine of a string can be converted to a float
# The truly pythonic name for this would be heavier_than_a_duck()
# =============================================================================
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_demographics(s):
# =============================================================================
# Function to parse the demographics from the user table,
# Return a dictionary of the demographics and their entry.
# It sorts through a demographic entry and finds all of the demographic keys
# in the string.  It then identifies all characters immediately after the 
# key that are contained between a : and a ].  This is extracted as the value
# of the key.
# =============================================================================
    entries=[]
    for d in demo_dict_keys:
        if d == 'complete':  # Treat complete different, because true is never in brackets
            pattern = r'true|false'
        else:
            pattern = r':(.*?)\]' # find anything enclosed between : and ]
        search_key = '"'+d+'"'
        m = re.split(search_key, s)
        if (len(m) > 1):
            n = re.search(pattern, m[1])
            if (n == None):
                entries.append('Null')
            elif ((n.group() == '["none"]') | 
                  (n.group() == ':[]') |
                  (n.group() == ':["."]')):
                entries.append('Null')
            else:
                entries.append(n.group().replace(':["','').replace('"]',''))          
        else:
            entries.append('Null')
    return dict(zip(demo_dict_keys,entries))


def token_found(response, token):
# =============================================================================
# Function to determine if a given user response contains a particular "token"
# =============================================================================
    if response.find(token) >= 0:
        return True
    else:
        return False

    
def princomp(A):
# =============================================================================
#     performs principal components analysis
#     (PCA) on the n-by-p data matrix A
#     Rows of A correspond to observations, columns to variables.
# 
#     Returns :
#     trans_matrix:
#     is a p-by-p matrix, each column containing coefficients
#     for one principal component (the eigenvectors of the covariance matrix)
#     data_project :
#     The representation of A in the principal component space. Rows of
#     data_project correspond to observations, columns to components.
#     eig_val :
#     a vector containing the eigenvalues of the covariance matrix of A.
# =============================================================================
    # computing eigenvalues and eigenvectors of covariance matrix
    [eig_val, trans_matrix] = np.linalg.eig(np.cov(A, rowvar=False))
    correct_order = np.flip(np.argsort(eig_val), axis=0)  # indicies of sorted eigenvalues
    trans_matrix = trans_matrix[:, correct_order]
    eig_val = eig_val[correct_order]
    # projection of the data in the new space
    data_project = (np.real(np.dot(np.linalg.pinv(trans_matrix), A.T))).T
    explained = 100.*np.real(eig_val/sum(eig_val))  # percent explained by each PC
    return trans_matrix, data_project, eig_val, explained


def update_token_df_index(token_df, new_addition):
# =============================================================================
# The token_df dataframe contains the information needed to recover textual tokens
# from numeric encoding and predictions.  The question ID is the index of this
# dataframe, and updating it properly involves a little shuffling.  This function
# handles that tedium
# =============================================================================
    old_index = token_df.index.tolist()
    token_df = token_df.append(token_series,ignore_index=True)
    old_index.append(new_addition.name)
    token_df.index = old_index
    return token_df


def all_types_same(answer_list):
# =============================================================================
# This function quickly verifies that every answer to a question is the same type.
# This was a bit of quality control to enure that there were no questions with numeric
# answers by some users and strings by others.
# =============================================================================
    type_list = [type(x) for x in answer_list]
    if type_list.count(type_list[0]) == len(type_list):
        return True
    else:
        return False
    

def get_question(q):
# =============================================================================
# Short function that prints the test of a question given its ID (q).  Saves typing.
# =============================================================================
    return list(question_df[question_df.question_id == q]['question_text'])


def get_answers(q):
# =============================================================================
# Prints all answers to a given question, to quickly verify if it is "restricted"
# or "combinatoric" given the question ID (q).
# =============================================================================
    return colab_filter_df[colab_filter_df.loc[:,q].notnull()].loc[:,q]


def is_restricted(answer_list):
# =============================================================================
# This function determines what questions restrict user answers to a set list,
# or allow combinatoric answer of "choose all that apply,"  It does this by
# splitting all answers at commas, and counting the number of unique"tokens"
# which result.  If the number of answers is strictly greater than the number 
# of unie tokens, the question is "combinatoric" in that the responses of individual 
# users were constructed from a a smaller set of tokens.  It is conceivable that
# there are "restricted" questions with unused selections.  If identified,
# These need to be recoded.
# =============================================================================
    all_used_answers = pd.unique(answer_list)
    bag_of_tokens = []
    for each_answer in all_used_answers:
        for each_choice in each_answer.split(','):
            bag_of_tokens.append(each_choice)
    # bag of tokens is every token used, every time it's used
    unique_tokens = list(set(bag_of_tokens))
    if (len(all_used_answers) > len(unique_tokens)):
        return False
    else:
        return True

    
def question_answer(u,q):
# =============================================================================
# Function to display the question text and response from a give user for a 
# specific question.  Inputs are the user_ID (u) and question_ID (q)
# =============================================================================
    question = list(question_df[question_df.question_id == q]['question_text'])
    answer = answer_df[((answer_df.user_id==u) & (answer_df.question_id==q))].response
    print(question)
    print(answer)
    

# =============================================================================
# Initial read of CSV files
# =============================================================================


orig_data_dir = '/home/driebel/Dropbox/Insight/insight_project/data/'  #Edit, obviously

questions_csv = orig_data_dir+'questions.csv'
answers_csv = orig_data_dir+'answers.csv'
users_csv = orig_data_dir+'users.csv'

question_df = pd.read_csv(questions_csv, sep='*')

# =============================================================================
# question_df now has 2 columns, 'id' and 'desriptions'.
# ID has the question ID, wrapped in ObjectId().
# description is the text of the question, looks good to go.
# =============================================================================

answer_df = pd.read_csv(answers_csv, sep=',')

# =============================================================================
# answer_df has 4 columns: user, question, type, selections.
# user is the user ID, with no wrapping or decoration
# question is the question ID, with no wrapping or decoration
# type is a code indicating the type of response:
#     mc - mulitple choice (yay!)
#     numeric - a number (double yay)
#     text - free response 
# selections is the actual textual response, wrapped in [" "], even numeric
# =============================================================================

user_df = pd.read_csv(users_csv, sep=',')

# =============================================================================
# user_df has two columns, _id and demographics
# id is the user ID, wrapped in ObjectID()
# demographics is going to be tricky.  It is a single string of the demographic 
# descriptors, formatted as
# {"complete":true, "CATEGORY":["VALUE"],...}
# Order of the tags is not consistent!
# =============================================================================

# Process Questions.  
# Remove ObjectId() wrapper
question_df.iloc[:,0] = [i.replace('ObjectId(','').replace(')','') for i in question_df.iloc[:,0]]

# rename question columns
question_cols = {'id':'question_id',
                 'description':'question_text'}
question_df.rename(columns = question_cols, inplace = True)


# Process Answers
# Dump the free text
answer_df = answer_df[answer_df['type'] != 'text']

answer_df.iloc[:,3] = [str(i).replace(']','').replace('[','') for i in answer_df.iloc[:,3]]

answer_df.iloc[:,3] = [str(i).replace('"','') for i in answer_df.iloc[:,3]]

b = answer_df.iloc[:,3].apply(lambda x: will_it_float(x))
# b is a boolean mask finding the elements of answer_df.selections which can be turned into numbers

# Convert all strings to floating point numbers
answer_df.loc[(b & (answer_df.type == 'numeric')), 'selections'] = answer_df.loc[(b & (answer_df.type == 'numeric')), 'selections'].apply(lambda x: float(x))
# answers still need to be coded as 1-4 for category questions

#rename answer columns
answer_cols = {'user':'user_id',
               'question':'question_id',
               'selections':'response'}
answer_df.rename(columns=answer_cols,inplace=True)

# Process users
#strip formatting from question IDs, exactly as before
user_df.iloc[:,0] = [i.replace('ObjectId(','').replace(')','') for i in user_df.iloc[:,0]]

demo_dict_keys = ['greek', 'athlete', 'financialAid', 'gender', 'geography',
                  'highschool', 'legacy', 'major', 'orientation', 'race',
                  'year','school', 'complete']

demo_data=[]
for s in user_df.iloc[:,1]:
    demo_data.append(parse_demographics(s))

demographic_df = pd.DataFrame(demo_data)

#Merge the deomgraphic data back into the user dataframe, as separate columns
user_df = user_df.merge(demographic_df,
                        how='outer',
                        left_index=True,
                        right_index=True)

# remove original demographics field, not useful
user_df = user_df.drop(['demographics'],axis=1)

# Rename columns of user DB
user_cols = {'id':'user_id'}
user_df.rename(columns=user_cols, inplace=True)

# Create SQL server
# =============================================================================
# I use SQL locally, not MongoDB.  The following long block of commented code
# Saves the modified dataframes to a local SQL server, and then reads them back in.
# If you are not using SQL, this step is unnecessary.  I include it for completeness
#
# username = 'postgres'
# password = 'SQLpassword'
# host     = 'localhost'
# port     = '5432'            # default port that postgres listens on
# db_name  = 'full_pulse_db'
# 
# engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name) )
# print(engine.url)
# 
# if not database_exists(engine.url):
#     create_database(engine.url)
# print(database_exists(engine.url))
# 
# answer_df.to_sql('answer_db', engine, if_exists='replace')
# user_df.to_sql('user_db', engine, if_exists='replace')
# question_df.to_sql('question_db',engine, if_exists='replace')
# =============================================================================

# Alternatively, you may wish to simply save these as csv files:
# =============================================================================
# answer_df.to_csv('<file_name>', sep='*')
# user_df.to_csv('<file_name>', sep='*')
# question_df.to_csv('<file_name>',sep='*')
# =============================================================================


# =============================================================================
# The next several lines read the data back out of a SQL server, using SQL queries
# to construct new tables.  The point is to construct the full_data table,
# which I have included elsewhere as a csv file.
# =============================================================================


# =============================================================================
# con = psycopg2.connect(database = db_name,
#                        user = username,
#                        host = host,
#                        password = password)
# 
# sql_query = """
# SELECT count(*) FROM answer_db WHERE type='mc';
# """
# data_from_sql = pd.read_sql_query(sql_query, con)
# data_from_sql.head()
# 
# duplicate_questions = question_df.question_text.duplicated()
# # a is a boolean list of where there are duplicated question texts
# 
# sql_query = """
# SELECT distinct question_db.question_id
# FROM question_db
# INNER JOIN answer_db ON question_db.question_ID = answer_db.question_id;
# """
# cur = con.cursor()
# cur.execute(sql_query)
# 
# answered_questions = pd.read_sql_query(sql_query, con)
# # This is a list of all the distinct answered questions
# answered_questions.to_sql('answered_questions',engine,if_exists='replace')
# 
# answer_df = psql.read_sql('SELECT * FROM answer_db', con)
# user_df = psql.read_sql('SELECT * FROM user_db', con)
# question_df = psql.read_sql('SELECT * FROM question_db',con)
# answered_questions = psql.read_sql('SELECT * FROM answered_questions',con)
# 
# sql_query = """
# SELECT a.user_id, a.question_id, a.type, a.response
# FROM answer_db a
# INNER JOIN answered_questions ON answered_questions.question_id = a.question_id;
# """
# 
# distinct = psql.read_sql(sql_query, con)
# distinct.drop_duplicates(subset=['user_id','question_id'],keep=False,inplace=True)
# a = distinct.pivot(index='user_id',columns = 'question_id', values = 'response')
# # a is now a table with USERS rows and QUESTIONS columns.  This is the format for
# # the eventual collaborative filter
# file_name = '/home/driebel/Dropbox/Insight/insight_project/data/full_data_table.csv'
# a.to_csv(file_name,sep='*')  # save it, it is too large for SQL
# =============================================================================

# =============================================================================
# I stored the data in a SQL database.  Here I read it in.
# The above code reads in the question_df and answer_df data from CSV files,
# so converting them to SQL and back is not necessary.
# username = 'postgres'
# password = 'SQLpassword'     
# host     = 'localhost'
# port     = '5432'            # default port that postgres listens on
# db_name  = 'full_pulse_db'
# 
# engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name) )
# # print(engine.url)
# 
# if not database_exists(engine.url):
#     create_database(engine.url)
# # print(database_exists(engine.url))
# 
# 
# con = psycopg2.connect(database = db_name,
#                        user = username,
#                        host = host,
#                        password = password)
# 
# answer_df = psql.read_sql('SELECT * FROM answer_db', con)
# question_df = psql.read_sql('SELECT * FROM question_db',con)
# =============================================================================

file_name = '/home/driebel/Dropbox/Insight/insight_project/data/full_data_table.csv'
full_data = pd.read_csv(file_name,sep='*',low_memory=False)
full_data.set_index('user_id', inplace=True)
colab_filter_df = full_data.copy(deep=True)


# Drop users who have only answered 2 or fewer questions ~320
number_answers = colab_filter_df.count(axis=1)
non_participant = np.where(number_answers <= 2)
rows_to_drop = colab_filter_df.index[non_participant]
colab_filter_df.drop(labels=rows_to_drop, inplace=True)

# Drop questions whch haven't been answered by more than 2 students ~300
number_of_students_who_answered = colab_filter_df.count(axis=0)
unanswered_questions = np.where(number_of_students_who_answered < 10)
cols_to_drop = colab_filter_df.columns[unanswered_questions]
colab_filter_df.drop(columns=cols_to_drop, inplace=True)

# =============================================================================
# colab_filter_df is a dataframe with each unique user as a row, with the response
# to each question stored as a column.  This is what will eventually be fed into a
# collaborative filter algorithm.
# =============================================================================


# =============================================================================
# Begin question processing.  First, determine what kind of question we are dealing with:
# a "combinatoric" question allowed users to select "all that apply," or even add their own
# These responses must be "tokenized" to find the choices allowed.
# Then create new "question" columns, representing each of the various possible tokens
# each user will be assigned a binary value for each token, whether they used it in their response or not
# These binary values will constitute an N dimensional vector for each user
# PCA those vectors, keep only the 1st PCA
# This PCA will be the target of our collaborative filter algorithm.
# 
# A "restricted string" question allowed users to only select from a well-defined spectrum of choices.
# These must be turned into numbers 1-4 (or so), and can be left alone beyond that
# 
# A "numeric" question, where users input a number.  These can also be left entirely alone
# 
# There are two questions in the database with Boolean True/False answers.
# Since 0 is missing data, convert True to 2, and False to 1.
# =============================================================================

token_df = pd.DataFrame() #token_df will store info about the "tokens" used in
# the combinatoric questions

for q in colab_filter_df.columns:
    valid_answers = colab_filter_df[colab_filter_df.loc[:,q].notnull()].loc[:,q]
    # valid_answers.index is user_ids of all people who have answered the current question
    
    if (type(valid_answers[0]) == bool):
        true_mask = valid_answers[valid_answers == True].index
        false_mask = valid_answers[valid_answers == False].index
        colab_filter_df.loc[true_mask, q] = 2
        colab_filter_df.loc[false_mask, q] = 1
        token_series = pd.Series([False, True],name=q,index=[1,2])
        token_df = update_token_df_index(token_df,token_series)
        
    if (type(valid_answers[0]) == str):
        if is_restricted(valid_answers):
            token_list = pd.unique(valid_answers)
            token_series = pd.Series(token_list,
                                     index = range(1,len(token_list)+1),
                                     name = q)
            for each_token in token_series:
                token_mask = valid_answers[valid_answers == each_token].index
                colab_filter_df.loc[token_mask, q] = token_series[token_series==each_token].index[0]
            token_df = update_token_df_index(token_df,token_series)
            
        else:
            all_used_answers = pd.unique(valid_answers)
            # This is the full combinatorics of all possible answers
            bag_of_tokens = []
            for each_answer in all_used_answers:
                if type(each_answer)==str:
                    for each_choice in each_answer.split(','):
                        bag_of_tokens.append(each_choice)
                else:
                    bag_of_tokens.append(each_answer)
            # bag of tokens is every token used, every time it's used
            unique_tokens = list(set(bag_of_tokens))
            if (len(all_used_answers) > len(unique_tokens)):
                # if this is true, we have a"combintoric" question, and it must be 
                # tokenized, eigenvlaued, etc.  If not, it just needs to be numeralized and
                # kept as is.
                empty_answer = r'^[ \t\r\n]*$'
                for solo_token in unique_tokens:
                    if ((bag_of_tokens.count(solo_token) <= 1) | 
                        bool(re.match(empty_answer, solo_token))):
                        #find the tokens only used once, or emtpy: toss them
                        bag_of_tokens.remove(solo_token)
                        unique_tokens.remove(solo_token)
                number_of_tokens = len(unique_tokens)
                token_series = pd.Series(data=unique_tokens, name=q, index=range(1,number_of_tokens+1))
                token_df = update_token_df_index(token_df, token_series)
                new_question_ids = []
                for i, each_token in enumerate(unique_tokens):
                    new_question_ids.append(q+'_a'+str(i))
                    #new_question_ids is a list of the new columns just created
                    #thus, valid_answers.index by new question_ids is the matrix to be PCA'ed
                    colab_filter_df[q+'_a'+str(i)] = pd.Series(index=valid_answers.index, 
                                    data=[1 if token_found(x,each_token) else 0 for x in valid_answers])
                # colab_filter_df now has N new columns, one for each possible token for question q.
                # these are binary columns.
                # Extract subset of data to be PCA'ed:
                data_to_be_pca = colab_filter_df.loc[valid_answers.index, new_question_ids]
                trans_matrix, data_project, eig_val, explained = princomp(data_to_be_pca)
                # now extract PCA_1 for each user and put it in the "main" array
                colab_filter_df.loc[valid_answers.index, q] = data_project[:,1]
         
    if (type(valid_answers[0]) == np.float64):
        pass
    
# After processing, colab_filter_df has many extra columns, all the "token" columns
# created above.  Remove them.  They will all be of the form <characters>_a[number]
        
token_column_pattern = r'(\w+)_a[0-9]+'
cols_to_drop = [x for x in colab_filter_df.columns[1:] if bool(re.search(token_column_pattern,x))]
colab_filter_df.drop(columns = cols_to_drop, inplace=True)

colab_filter_file = '/home/driebel/Dropbox/Insight/insight_project/data/colab_filter_df.csv'    
colab_filter_df.to_csv(colab_filter_file,sep='*')

token_file = '/home/driebel/Dropbox/Insight/insight_project/data/token_df.csv'
token_df.to_csv(token_file,sep='*')

# Center the dataset, and convert everything to Z-score
colab_filter_df = (colab_filter_df-colab_filter_df.mean())/colab_filter_df.std()
colab_filter_df.reset_index(inplace=True)
colab_filter_df.fillna(0,inplace=True)


# The index is just an integer list, and user_id is just an ordinary column.
# =============================================================================
# RAM limitations mean I cannot handle the entire dataset.  So let's pull out a 
# random 1/2 and go with that
# =============================================================================
subsample = colab_filter_df.iloc[np.random.choice(colab_filter_df.index,
                                                  size=round(colab_filter_df.shape[0]/2)),:]


# =============================================================================
# The Surprise collaborative filter package for python requires data be in three columns:
# user, item, rating.
# In our case, that means user_id, question_id, response.
# The pandas melt method accomplishes this.
# Because of the encoding of ordinal vs categorical text responses, predictions are not
# currently meaningful.  The below code runs without crashing, and does return
# numerical evaluations.  However, the underlying data quality does not justify
# any actual prediction, and thus there is NO fine tuning of the underlying model. 
# =============================================================================

df_for_surprise = pd.melt(subsample,id_vars='user_id')
df_for_surprise.rename(columns={'variable':'question_id'},inplace=True)

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(df_for_surprise, reader)

trainset = data.build_full_trainset()

algo = SVD()

algo.fit(trainset)

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    
    
