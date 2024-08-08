from prompts import *
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse
import logging
import google.generativeai as genai
import pickle
import json
import time
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import google.generativeai as genai
import logging
import speech_recognition as sr  # For Google API transcription

import base64
from pydantic import BaseModel
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define agent prompts (assume these are imported or defined elsewhere)
agent_prompts = {
    0: Zara_prompt,
    1: Karlah_prompt,
    2: Raphael_prompt,
    3: John_prompt,
    4: Amara_prompt
}

monitor_agent_prompts = {
    0: monitor_Zara_prompt,
    1: monitor_Karlah_prompt,
    2: monitor_Raphael_prompt,
    3: monitor_John_prompt,
    4: monitor_Amara_prompt
}

class Narration(BaseModel):
    transcription: str
    description: str

class Monitoring(BaseModel):
    assessment: str
    transcription: str
    description: str

class AnalysisResponse(BaseModel):
    narration: str
    updated_history: str  # Base64 encoded pickle data
    status: str

def format_recent_history(history, num_exchanges=2):
    # We'll get up to 5 messages (2 full exchanges plus the last user input)
    recent_history = history[-(num_exchanges*2 + 1):]
    formatted_history = ""
    for i in range(0, len(recent_history)):
        if i % 2 == 0:
            user_message = recent_history[i].parts[0].text
            formatted_history += f"User: {user_message}\n"
        else:
            agent_message = recent_history[i].parts[0].text
            formatted_history += f"Agent: {agent_message}\n\n"
    return formatted_history.strip()

@app.get("/")
async def root():
    return {"message": "Welcome to the audio analysis API"}

GOOGLE_AI_ENDPOINT = "https://generativelanguage.googleapis.com/v1/models"

class APIKeyCheck(BaseModel):
    gemini_api_key: str

@app.post("/check_api_key/")
async def check_api_key(api_key_data: APIKeyCheck):
    gemini_api_key = api_key_data.gemini_api_key
    logger.info("Received request to check API key")
    
    # Check server connectivity
    try:
        response = requests.get(GOOGLE_AI_ENDPOINT, timeout=5)
        logger.info(f"Server responded with status code: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"Failed to connect to server: {str(e)}")
        raise HTTPException(status_code=500, detail="Unable to connect to Google AI server")

    # Check API key validity
    try:
        logger.debug("Configuring genai with provided API key")
        genai.configure(api_key=gemini_api_key)
        
        logger.debug("Attempting to create a GenerativeModel")
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Attempt a simple operation to verify the key
        model.generate_content("Test")
        
        logger.info("API key validation successful")
        return {"status": "valid", "message": "API key is valid"}
    except Exception as e:
        logger.error(f"API key validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid API key: {str(e)}")


class TranscriptionResponse(BaseModel):
    transcription: str
    processing_time: float

### Method 1: Transcribe Using Google's SpeechRecognition
@app.post("/transcribe_google/", response_model=TranscriptionResponse)
async def transcribe_google(audio_file: UploadFile = File(...)):
    start_time = time.time()
    try:
        # Step 1: Convert Audio to Text using Google's API
        recognizer = sr.Recognizer()
        audio_data = sr.AudioFile(await audio_file.read())

        with audio_data as source:
            audio = recognizer.record(source)
            transcription_text = recognizer.recognize_google(audio)
        
        # Measure total processing time
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Google API transcription completed in {processing_time:.2f} seconds")

        return TranscriptionResponse(transcription=transcription_text, processing_time=processing_time)

    except Exception as e:
        logger.error(f"Error during Google API transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during Google API transcription: {str(e)}")

### Method 2: Transcribe Using Gemini API
@app.post("/transcribe_basic/", response_model=TranscriptionResponse)
async def transcribe_basic(
    gemini_api_key: str = Form(...),
    audio_file: UploadFile = File(...)
):
    start_time = time.time()
    try:
        # Configure the Gemini API with the provided API key
        genai.configure(api_key=gemini_api_key)
        logger.info("Genai configured with provided API key")

        # Read the audio file content
        audio_content = await audio_file.read()

        # Upload the audio file to Gemini
        upload_start_time = time.time()
        file_response = genai.upload_file(audio_file.filename, content=audio_content)
        logger.info(f"Audio file uploaded in {time.time() - upload_start_time:.2f} seconds")

        # Get the URI of the uploaded file
        audio_uri = file_response.uri

        # Create a generative model and request transcription
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content([audio_uri, "Transcribe this audio."])

        # Extract the transcription from the response
        transcription_text = response.text

        # Measure total processing time
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Gemini API transcription completed in {processing_time:.2f} seconds")

        return TranscriptionResponse(transcription=transcription_text, processing_time=processing_time)

    except Exception as e:
        logger.error(f"Error during Gemini API transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during Gemini API transcription: {str(e)}")

@app.post("/talk_to_agents/", response_model=AnalysisResponse)
async def analyze_audio(
    text: str=Form(...),
    agent_number: int = Form(...),
    history_file: UploadFile = File(...),
    gemini_api_key: str = Form(...)
):
    try:
        # Validate agent number
        if agent_number not in range(5):
            raise HTTPException(status_code=400, detail="Invalid agent number")

        # Configure genai with the provided API key
        genai.configure(api_key=gemini_api_key)
        logger.info("Genai configured with provided API key")


        # Load history from base64 encoded pickle
        history_base64 = await history_file.read()
        history = pickle.loads(base64.b64decode(history_base64))
        logger.info("History loaded from base64 encoded pickle data")
        logger.info(history)
  

        # Select agent and process response
        agent_prompt = agent_prompts[agent_number]
        agent = genai.GenerativeModel('gemini-1.5-flash', system_instruction=agent_prompt,
                                            generation_config=genai.GenerationConfig(
                                                response_schema=Narration,
                                                response_mime_type="application/json",
                                                temperature=0.1
                                            ))

        
        
        agent_chat = agent.start_chat(history=history[agent_number])
        logger.info(f"Agent {agent_number} chat started with history")
        
        agent_response=agent_chat.send_message(text).text
        recent_conversation = format_recent_history(agent_chat.history)        
        # This is the transcription part of the response
        agent_response_transcirption = json.loads(agent_response)['transcription']
        
        
        

    
        monitor_agent_prompt= monitor_agent_prompts[agent_number]
        recent_conversation = format_recent_history(agent_chat.history)
        #I made the agent response and history to a dict to feed it in the prompt of the monitor
        monitor_data = {
            "latest_response": agent_response,
            "conversation_history": recent_conversation
        }
        monitor_agent_prompt = custom_format_double(monitor_agent_prompt, monitor_data)                                                     
        monitor_agent=genai.GenerativeModel('gemini-1.5-flash', system_instruction=monitor_agent_prompt,
                    generation_config=genai.GenerationConfig(
                        response_schema=Monitoring,
                        response_mime_type="application/json",
                        temperature=0
                    ))
        monitor_agent_chat=monitor_agent.start_chat(history=[])
        try:
            monitor_agent_response = monitor_agent_chat.send_message("""\n Monitor_Administrator: Analyze and provide the assessment please""").text
        except Exception as e:
            print(f"An error occurred: {e}")
            monitor_agent_response = '{"assessment": "There was a problem with the answer.", "transcription": "Stop harassing me", "description": "nothing"}'

        # Now you can use monitor_agent_response safely
        
        

        monitor_agent_response_assessment= json.loads(monitor_agent_response)['assessment']
        
        if monitor_agent_response_assessment=="There was a problem with the answer.":
            logger.info("there was a security issute")
            monitor_agent_response_transcirption = json.loads(monitor_agent_response)['transcription']
            agent_response_transcirption=monitor_agent_response_transcirption
        agent_chat.history[-1].parts[0].text = agent_response_transcirption
        logger.info(f"Agent {agent_number} response generated")

        # Update history
        history[agent_number] = agent_chat.history
        logger.info(agent_chat.history)
        # Prepare response
        
        # Pickle and encode updated history
        updated_history_pickle = pickle.dumps(history)
        updated_history_base64 = base64.b64encode(updated_history_pickle).decode('utf-8')
        
        return AnalysisResponse(
            narration=agent_response_transcirption,
            updated_history=updated_history_base64,
            status="success"
        )

    except Exception as e:
        logger.error(f"Error processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)