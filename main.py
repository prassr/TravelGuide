import logging
import time
from langchain.chains import LLMChain, SequentialChain
from langchain_community.utilities import GoogleSerperAPIWrapper
from groq import Groq
from langchain_groq import ChatGroq
import os
import requests
import json
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
import clipboard

import streamlit as st
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

st.set_page_config(page_title="Travel Guide - Ankur AI")

with st.sidebar:
    st.title("Travel Guide")
    st.subheader('Assistant')

    description = """
        Get exciting itinerary with **Travel Guide**.
        Know about places you want to visit with place names.
    """
    st.markdown(description)

@tool
def scrape(city):
    """Search by city name"""

    url = "https://scrape.serper.dev"

    payload = json.dumps({
        "url": "https://en.wikivoyage.org/wiki/Goa"
    })
    headers = {
        'X-API-KEY': st.secrets["SERPER_API_KEY"],
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text

def search_images(city):
    url = "https://google.serper.dev/images"

    payload = json.dumps({
        "q": f"best images for {city} where image size is less than or equal to 600 x 600"
    })
    headers = {
        'X-API-KEY': st.secrets["SERPER_API_KEY"],
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


def google_search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': st.secrets["SERPER_API_KEY"],
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text

logging.basicConfig(level=logging.INFO)

class Validation(BaseModel):
    query_is_valid: str = Field(
        description="This field is 'yes' if the query is feasible, 'no' otherwise"
    )
    type: str = Field("This field is 'itinerary' if the user asks about travel plan, 'guide' if the user asks information about a places, villages, states countries")
    updated_request: str = Field(description="Your update to the query.")
    cities: list = Field(description="A list of cities mentioned in the query, the destination city should be at the end.")



def load_secets():
    return {
        "GROQ_API_KEY":st.secrets["GROQ_API_KEY"]
    }

class ValidationTemplate(object):
    def __init__(self):
        self.system_template = """
      You are a Travel Query Validator who helps validate user queries related to all aspects tourism.

      The user's request will be denoted by four hashtags. Determine if the user's
      request is reasonable and achievable within the constraints they set.

      A valid request should contain the following:
      - A start and end location
      - A trip duration that is reasonable given the start and end location
      - Some other details, like the user's interests and/or preferred mode of transport
      - It asks about a location like hospitals, places, villages, tourism, states, countries.
      - A city name is a valid query. Here the user is asking about the city. In this case provide information about the city.
      Any request that contains potentially harmful activities is not valid, regardless of what
      other details are provided.

      If the request is not valid, set
      query_is_valid = 0 and use your travel expertise to update the request to make it valid,
      keeping your revised request shorter than 100 words.

      If the request seems reasonable, then set query_is_valid = 1 and
      don't revise the request.

      {format_instructions}
    """

        self.human_template = """
      ####{query}####
    """

        self.parser = PydanticOutputParser(pydantic_object=Validation)

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )


class ItineraryTemplate(object):
    def __init__(self):
        self.system_template = """
      You are a Travel Agent who helps users make exciting travel plans.

      The user's request will be denoted by four hashtags. Convert the
      user's request into a detailed itinerary describing the places
      they should visit and the things they should do.

      Try to include the specific address of each location.

      Remember to take the user's preferences and timeframe into account,
      and give them an itinerary that would be fun and doable given their constraints.

      Return the itinerary as a bulleted list with clear start and end locations.
      Be sure to mention the type of transit for the trip.
      If specific start and end locations are not given, choose ones that you think are suitable and give specific addresses.
      Your output must be the list and nothing else.
    """

        self.human_template = """
            ####{query}####
        """

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template, input_variables=["context"]
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )


class GuideTemplate(object):
    def __init__(self):
        self.system_template = """
        As a local expert on this city you must compile an 
        in-depth guide for someone traveling there and wanting 
        to have THE BEST trip ever!
        Gather information about  key attractions, local customs,
        special events, and daily activity recommendations.
        Find the best spots to go to, the kind of place only a
        local would know.
        This guide should provide a thorough overview of what 
        the city has to offer, including hidden gems, cultural
        hotspots, must-visit landmarks, weather forecasts, and
        high level costs.
        
        The final answer must be a comprehensive city guide, 
        rich in cultural insights and practical tips, 
        tailored to enhance the travel experience.
        """

        self.human_template = """
            ####{query}####
        """
        
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )

class SerperTemplate(object):
    def __init__(self):
        self.system_template = """
        You are the best markdown formatter given json data. 
        This is for your information only. 
        No need to tell to the user that you are receiving JSON data.
        $$${context}$$$
        Your task is to get the tourism related links from above links and put them in markdown format in following categories.

        Start answer with:
        
        Here are some useful links:

        Government Websites:

        Tourism-related Links:

        Other Relevant Links:
        
        """

        self.human_template = """
            ####{query}####
        """
        
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template, input_variables=["context"]

        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )


class ImageTemplate(object):
    def __init__(self):
        self.system_template = """
            You are the best Image manager out there.
            Given a set of images in JSON, your task is to put them in markdown format.
            Take the image url and put it as markdown. Also specify the source of the image. 
            Pick only the first 2 images for the display.
            Here are the links:
                $$${context}$$$
            Your response starts with:
            Images\n\n:
                
        """

        self.human_template = """
            ####{query}####
        """
        
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template, input_variables=["context"]

        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )


class Agent(object):
    def __init__(
        self,
        groq_api_key,
        model = "llama3-70b-8192",
        temperature=0,
        debug=True,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._groq_key = groq_api_key

        self.chat_model = ChatGroq(model=model, temperature=temperature, groq_api_key=self._groq_key)
        
        self.validation_prompt = ValidationTemplate()
        self.itinerary_prompt = ItineraryTemplate()
        self.guide_prompt = GuideTemplate()
        self.serper_prompt = SerperTemplate()
        self.image_prompt = ImageTemplate()
        self.validation_chain = self._set_up_validation_chain(debug)
        self.agent_chain = self._set_up_agent_chain(debug)
        self.guide_chain = self._set_up_guide_chain(debug)
        self.serper_chain = self._set_up_serper_chain(debug)
        self.image_chain = self._set_up_image_chain(debug)

    def _set_up_validation_chain(self, debug=True):
      
        # make validation agent chain
        validation_agent = LLMChain(
            llm=self.chat_model,
            prompt=self.validation_prompt.chat_prompt,
            output_parser=self.validation_prompt.parser,
            output_key="validation_output",
            verbose=debug,
        )
        
        # add to sequential chain 
        overall_chain = SequentialChain(
            chains=[validation_agent],
            input_variables=["query", "format_instructions"],
            output_variables=["validation_output"],
            verbose=debug,
        )
        return overall_chain

    def validate_travel(self, query):
        self.logger.info("Validating query")
        t1 = time.time()
        self.logger.info(
            "Calling validation (model is {}) on user input".format(
                self.chat_model.model_name
            )
        )
        validation_result = self.validation_chain(
            {
                "query": query,
                "format_instructions": self.validation_prompt.parser.get_format_instructions(),
            }
        )

        validation_test = validation_result["validation_output"].dict()
        t2 = time.time()
        self.logger.info("Time to validate request: {}".format(round(t2 - t1, 2)))
        return validation_test

    def _set_up_agent_chain(self, debug=True):
  
        # set up LLMChain to get the itinerary as a string
        travel_agent = LLMChain(
                llm=self.chat_model,
                prompt=self.itinerary_prompt.chat_prompt,
                verbose=debug,
                output_key="agent_suggestion",
            )

        overall_chain = SequentialChain(
                chains=[travel_agent],
                input_variables=["query"],
                output_variables=["agent_suggestion"],
                verbose=debug,
            )

        return overall_chain

    def _set_up_guide_chain(self, debug=True):
        guide = LLMChain(
                llm=self.chat_model,
                prompt=self.guide_prompt.chat_prompt,
                verbose=debug,
                output_key="agent_suggestion"
                )
        
        overall_chain = SequentialChain(
                chains=[guide],
                input_variables=["query"],
                output_variables=["agent_suggestion"],
                verbose=debug,
            )

        return overall_chain

    def _set_up_serper_chain(self, debug=True):
        serper = LLMChain(
                llm=self.chat_model,
                prompt=self.serper_prompt.chat_prompt,
                verbose=debug,
                output_key="agent_suggestion"
                )
        
        overall_chain = SequentialChain(
                chains=[serper],
                input_variables=['context', "query"],
                output_variables=["agent_suggestion"],
                verbose=debug,
            )

        return overall_chain

    def _set_up_image_chain(self, debug=True):
        serper = LLMChain(
                llm=self.chat_model,
                prompt=self.image_prompt.chat_prompt,
                verbose=debug,
                output_key="agent_suggestion"
                )

        overall_chain = SequentialChain(
                chains=[serper],
                input_variables=['context', "query"],
                output_variables=["agent_suggestion"],
                verbose=debug,
            )

        return overall_chain


    def suggest_travel(self, query):
        self.logger.info("Validating query")
        t1 = time.time()
        self.logger.info(
            "Calling validation (model is {}) on user input".format(
                self.chat_model.model_name
            )
        )
        validation_result = self.validation_chain(
            {
                "query": query,
                "format_instructions": self.validation_prompt.parser.get_format_instructions(),
            }
        )

        validation_test = validation_result["validation_output"].dict()
        t2 = time.time()
        self.logger.info("Time to validate request: {}".format(round(t2 - t1, 2)))

        if validation_test["query_is_valid"].lower() == "no":
            self.logger.warning("User request was not valid!")
            print("\n######\n Travel plan is not valid \n######\n")
            print(validation_test["updated_request"])

            if validation_test["type"].lower() == "itinerary":

                agent_result = self.agent_chain(
                    {
                        # "context": search_city(validation_test["cities"][-1]),
                        "query": validation_test["updated_request"],
                    # "format_instructions": self.mapping_prompt.parser.get_format_instructions(),
                    }
                )
            if validation_test["type"].lower() == "guide":
                agent_result = self.guide_chain({
                    "query": validation_test["updated_request"],
                })

            
            external_resource = self.serper_chain({
                        "context": google_search(query),
                        "query": validation_test["updated_request"],
                })

        if validation_test["query_is_valid"].lower()=="yes":
            # plan is valid
            self.logger.info("Query is valid")
            self.logger.info("Getting travel suggestions")
            t1 = time.time()

            self.logger.info(
                "User request is valid, calling agent (model is {})".format(
                    self.chat_model.model_name
                )
            )

            if validation_test["type"].lower() == "itinerary":

                agent_result = self.agent_chain(
                    {
                        # "context": search_city(validation_test["cities"][-1]),
                        "query": query,
                    # "format_instructions": self.mapping_prompt.parser.get_format_instructions(),
                    }
                )
            if validation_test["type"].lower() == "guide":
                agent_result = self.guide_chain({
                    "query": query
                })

            trip_suggestion = agent_result["agent_suggestion"]
            images = self.image_chain({
                "context": search_images(validation_test["cities"][-1]),
                "query" : ""
            })["agent_suggestion"]
            external_resource = self.serper_chain({
                        "context": google_search(query),
                        "query": validation_test["updated_request"],
                })


            # list_of_places = agent_result["mapping_list"].dict()
            t2 = time.time()
            self.logger.info("Time to get suggestions: {}".format(round(t2 - t1, 2)))

            return trip_suggestion, validation_test, images,external_resource["agent_suggestion"]






secrets = load_secets()
travel_agent = Agent(groq_api_key=secrets["GROQ_API_KEY"],debug=True)

query = """
        What are the beautiful places in Goa?
        """

input = st.chat_input("I want to plan for a trip from ...")

if prompt := input:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Getting plan for you..."):
            itinerary, validation, images, external_resource = travel_agent.suggest_travel(prompt)
            placeholder = st.empty()

            full_response = str(itinerary +"\n\n"+ images +"\n\n"+ external_resource)
            # resources = st.empty()
            # images = st.empty()
            # placeholder.markdown(itinerary)
            # placeholder.markdown(images)
            # resources.markdown(external_resource)
            placeholder.markdown(full_response)
   

# # valid = travel_agent.validate_travel(query)
# # print(valid)
# itinerary, validation = travel_agent.suggest_travel(query)
# print(itinerary)


