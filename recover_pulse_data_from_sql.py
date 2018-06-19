#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:53:04 2018

@author: driebel
"""

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import re
import pandas.io.sql as psql
import collections

# Create SQL server
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

#answer_df = psql.read_sql('SELECT * FROM answer_db', con)
#user_df = psql.read_sql('SELECT * FROM user_db', con)
question_df = psql.read_sql('SELECT * FROM question_db',con)
#answered_questions = psql.read_sql('SELECT * FROM answered_questions',con)

file_name = '/home/driebel/Dropbox/Insight/insight_project/data/full_data_table.csv'
full_data = pd.read_csv(file_name,sep='*',low_memory=False)
full_data.set_index('user_id', inplace=True)


sql_query = """
SELECT a.user_id, a.question_id, a.type, a.response
FROM answer_db a
INNER JOIN answered_questions ON answered_questions.question_id = a.question_id;
"""

distinct = psql.read_sql(sql_query, con)

all_users = list(set(distinct.user_id))
all_questions = list(set(distinct.question_id))
distinct = distinct.set_index('user_id')


