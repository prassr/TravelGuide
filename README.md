# Travel Guide

Generates Travel Itinerary. Gets you information about a place.

Running Instructions
* Clone the repository.
* Set Up the Environment
  ```bash
  cd TravelGuide
  python3 -m venv venv
  . venv/bin/activate
  pip install -r requirements.txt
  ```
* Create `.streamlit/secrets.toml` file, and add the following environment variables.
  ```toml
  GROQ_API_KEY = ""
  SERPER_API_KEY=""
  ```
* Run
  ```bash
  streamlit run main.py
  ```
