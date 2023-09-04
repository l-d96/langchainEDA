import asyncio
import aiohttp
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate)
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from functools import partial
import openai
import os
import re


openai.api_key = os.getenv('OPENAI_API_KEY')


class BaseSQLChain:
    def __init__(self, llm, output_parser=StrOutputParser(), **kwargs):
        self.llm_chain = LLMChain(llm=llm,
                                  output_parser=output_parser,
                                  prompt=self.chat_prompt,
                                  **kwargs)

    def run(self, **kwargs):
        return self.llm_chain.run(**kwargs)


class NewlineSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a newline-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call"""
        steps = text.strip().split("\n")
        partial_findall = partial(re.findall, r'\d{1,3}\.\s')
        bullet_point = map(lambda x: partial_findall(x), steps)
        cleaned_bullet_point_steps = filter(lambda x: x[0],
                                            zip(bullet_point, steps))

        cleaned_steps = map(lambda arg: arg[1].removeprefix(arg[0][0]),
                            cleaned_bullet_point_steps)

        final_instructions = {index+1: step for index, step
                              in enumerate(cleaned_steps)}

        return final_instructions


class SQLValidationOutputParser(BaseOutputParser):
    """
    Parse the SQL instruction output of an LLM,
    and validate it using an LLM
    """

    def parse(self, text: str):
        validated_query = ""
        try:
            validation = int(SQLliteValidatorChain(llm=ChatOpenAI()).run(text))
            if validation:
                validated_query = text
        except:
            pass

        return validated_query


class BatchSQLInstructionsChain(BaseSQLChain):
    template = ("You are a powerful data analyzer specialized in generating "
                "exploratory data analysis steps. When prompted with the "
                "schema of" " a dataset, you ONLY return a newline separated "
                "list of steps to " "perform to extract useful insights from "
                "the dataset, and nothing " "more. The steps must be in a "
                "format that's easy to translate in "
                "SQL instructions. However do not return the SQL instructions "
                "directly. The steps must be in the following format "
                " and in no " "other format whatsoever: `step_no`. `step`")
    human_template = "Description of dataset: {description}\n\nSchema: {schema}"

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    human_message_prompt])


class SQLiteConverterChain(BaseSQLChain):

    template = ("You are an expert data analyst well versed in converting "
                "natural language queries into SQL statements. Your particular"
                " field of expertise is the SQLite dialect. You are going to "
                "be prompted a natural language query and the schema of a "
                "dataset, along with its description. Your ONLY output must be"
                " a SQLite query that executes the natural language query you "
                "were prompted on the dataset with the particular schema you "
                "were provided")
    human_template = ("This is the description of the dataset: {description}"
                      "\n\nThis is the schema of the dataset. You allowed to "
                      "use only these fields in your SQLite queries and "
                      "nothing else: {schema}\n\nThis is the query that you "
                      "have to translate into SQLite: {query}")

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    human_message_prompt])


class SQLliteValidatorChain(BaseSQLChain):
    template = ("You are a powerful SQLite validator engine. You are provided "
                "a SQLite query, the description of the dataset that is being query and its schema."
                " If the query is syntactically correct, your"
                " ONLY output is 1. If the Query is syntactically wrong, "
                "your ONLY output is 0.")
    human_template = ("This is the description of the dataset: {description}\n\n"
                      "This is the schema of the dataset that is being queried: {schema}\n\n"
                      "This is the query that you need to validate: {query}")

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    human_message_prompt])
