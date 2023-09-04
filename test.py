import pandas as pd
import openpyxl
import langchainSQL
from langchain.chat_models import ChatOpenAI
from importlib import reload
import sqlite3
from sklearn.datasets import load_breast_cancer

reload(langchainSQL)

data = load_breast_cancer()
dataset = pd.DataFrame(data.data, columns=data.feature_names)
# dataset['Target'] = data.target
# connection = sqlite3.connect("test.db")
# cursor = connection.cursor()
# dataset.to_sql(name="Breast Cancer", con=connection)
# connection.close()

description = "The name of the dataset is 'Breast Cancer'." + data.DESCR + "The name of the dataset is 'Breast Cancer'." 
schema = list(data.feature_names) + ["Target"]

analyzer_chain = langchainSQL.BatchSQLInstructionsChain(llm=ChatOpenAI(),
                                                        output_parser=langchainSQL.NewlineSeparatedListOutputParser())

result = analyzer_chain.run(description=description,
                            schema=schema)

converter_chain = langchainSQL.SQLiteConverterChain(llm=ChatOpenAI(model="gpt-4"))
validator_chain = langchainSQL.SQLliteValidatorChain(llm=ChatOpenAI())
queries: dict[str, str] = {}
for step, query in result.items():
    result_sql = converter_chain.run(description=description,
                                     schema=schema,
                                     query=query)
    if result_sql:
        queries[step] = result_sql
    print(query)
    print(result_sql)
    print("\n")

connection = sqlite3.connect("./test.db")
cursor = connection.cursor()
results = []
sql_query = queries[2]
# sql_query = queries[3][:-1]+" limit 100;;"
for sql_query in queries.values():
    try:
        step_result = cursor.execute(sql_query.replace("breast_cancer_dataset", "'Breast Cancer'")).fetchall()
        results.append(step_result)
    except Exception as e:
        print(e)
        continue
