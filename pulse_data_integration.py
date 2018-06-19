#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:04:28 2018

@author: driebel
"""


from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import re
import pandas.io.sql as psql

# Fucntion to determine of a string can be converted to a float
def will_it_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
    
# Function to parse the demographics from the user table,
# Return a dictionary of the demographics and their entry.
def parse_demographics(s):
    entries=[]
    for d in demo_dict_keys:
        if d == 'complete':  # Treat complete different, because true is never in brackets
            pattern = r'true|false'
        else:
            pattern = r':(.*?)\]'
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
        

orig_data_dir = '/home/driebel/Dropbox/Insight/insight_project/data/'

questions_csv = orig_data_dir+'questions.csv'
answers_csv = orig_data_dir+'answers.csv'
users_csv = orig_data_dir+'users.csv'

question_df = pd.read_csv(questions_csv, sep='*')

# =============================================================================
# q now has 2 columns, 'id' and 'desriptions'.
# ID has the question ID, wrapped in ObjectId().
# description is the text of the question, looks good to go.
# =============================================================================

answer_df = pd.read_csv(answers_csv, sep=',')

# =============================================================================
# a has 4 columns: user, question, type, selections.
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
# u has two columns, _id and demographics
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

# This is the best thing.
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
username = 'postgres'
password = 'SQLpassword'     # change this
host     = 'localhost'
port     = '5432'            # default port that postgres listens on
db_name  = 'full_pulse_db'

engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name) )
print(engine.url)

if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))

answer_df.to_sql('answer_db', engine, if_exists='replace')
user_df.to_sql('user_db', engine, if_exists='replace')
question_df.to_sql('question_db',engine, if_exists='replace')

con = psycopg2.connect(database = db_name,
                       user = username,
                       host = host,
                       password = password)

sql_query = """
SELECT count(*) FROM answer_db WHERE type='mc';
"""
data_from_sql = pd.read_sql_query(sql_query, con)
data_from_sql.head()

duplicate_questions = question_df.question_text.duplicated()
# a is a boolean list of where there are duplicated question texts

sql_query = """
SELECT distinct question_db.question_id
FROM question_db
INNER JOIN answer_db ON question_db.question_ID = answer_db.question_id;
"""
cur = con.cursor()
cur.execute(sql_query)

answered_questions = pd.read_sql_query(sql_query, con)
# This is a list of all the distinct answered questions
answered_questions.to_sql('answered_questions',engine,if_exists='replace')

answer_df = psql.read_sql('SELECT * FROM answer_db', con)
user_df = psql.read_sql('SELECT * FROM user_db', con)
question_df = psql.read_sql('SELECT * FROM question_db',con)
answered_questions = psql.read_sql('SELECT * FROM answered_questions',con)

sql_query = """
SELECT a.user_id, a.question_id, a.type, a.response
FROM answer_db a
INNER JOIN answered_questions ON answered_questions.question_id = a.question_id;
"""

distinct = psql.read_sql(sql_query, con)
distinct.drop_duplicates(subset=['user_id','question_id'],keep=False,inplace=True)
a = distinct.pivot(index='user_id',columns = 'question_id', values = 'response')
# a is now a table with USERS rows and QUESTIONS columns.  This is the format for
# the eventual collaborative filter
file_name = '/home/driebel/Dropbox/Insight/insight_project/data/full_data_table.csv'
a.to_csv(file_name,sep='*')  # save it, it is too large for SQL


# The below comments were attempts to remove duplicates from the dataset.
# =============================================================================
# 
# users_with_dup_quest =[]
# for i,u in enumerate(all_users):
#     if i % 1000 == 0:
#         print(i)
#     if((len(distinct.loc[u, 'question_id']) != len(set(distinct.loc[u, 'question_id'])))
#     & (type(distinct.loc[u, 'question_id']) != str)):
#         users_with_dup_quest.append(u)
#   
# Loop to find duplicate entries, takes over an hour to run  
# =============================================================================
# 
# 
# file_name = '/home/driebel/Desktop/dummies.txt'
# with open(file_name,'w') as f:
#     for i in users_with_dup_quest:
#         out = str(i)+'\n'
#         f.write(out)
#         
# 
# =============================================================================


# =============================================================================
# file_name = '/home/driebel/Desktop/dummies.txt'
# import collections
# with open(file_name) as f:
#     content = f.readlines()
#     for u in content:
#         a = distinct.loc[u.rstrip()].question_id
#         b = [item for item, count in collections.Counter(a).items() if count > 1]
#         distinct[distinct.question_id==b[0]].loc[u.rstrip()].drop_duplicates(inplace=True)
# =============================================================================





 



