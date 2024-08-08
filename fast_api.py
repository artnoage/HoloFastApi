import os
import tempfile
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import google.generativeai as genai
import pickle
import json
import base64
import io
import numpy as np
from pydub import AudioSegment
from datetime import datetime
from pydantic import BaseModel
import google.api_core
# Import your prompt configurations
from prompts import *

# Configuration
DEBUG_MODE = False  # Set to False in production
DEBUG_AUDIO_DIR = "./"  # Change this to your preferred debug directory
MAX_DEBUG_FILES = 100  # Maximum number of debug files to keep
SAMPLE_RATE = 22050  # Make sure this matches your client-side recording sample rate

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class APIKeyCheck(BaseModel):
    gemini_api_key: str

def format_recent_history(history, num_exchanges=2):
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

def save_debug_audio(audio_data: bytes, format: str = 'wav'):
    if not DEBUG_MODE:
        return

    os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(DEBUG_AUDIO_DIR, f"debug_audio_{timestamp}.{format}")

    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    audio_segment = AudioSegment(
        audio_np.tobytes(), 
        frame_rate=SAMPLE_RATE, 
        sample_width=2, 
        channels=1
    )
    audio_segment.export(file_path, format=format)

    logger.info(f"Debug audio saved: {file_path}")

    # Clean up old debug files if exceeding the maximum
    debug_files = sorted(
        [f for f in os.listdir(DEBUG_AUDIO_DIR) if f.startswith("debug_audio_")],
        key=lambda x: os.path.getctime(os.path.join(DEBUG_AUDIO_DIR, x))
    )
    for old_file in debug_files[:-MAX_DEBUG_FILES]:
        os.remove(os.path.join(DEBUG_AUDIO_DIR, old_file))

def process_audio(audio_data: bytes, format: str = 'wav'):
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    
    buffer = io.BytesIO()
    AudioSegment(
        audio_np.tobytes(), 
        frame_rate=SAMPLE_RATE, 
        sample_width=2, 
        channels=1
    ).export(buffer, format=format)
    
    buffer.seek(0)
    return buffer

@app.get("/")
async def root():
    return {"message": "Welcome to the audio analysis API"}

GOOGLE_AI_ENDPOINT = "https://generativelanguage.googleapis.com/v1/models"


@app.post("/check_api_key/")
async def check_api_key(api_key_data: APIKeyCheck):
    gemini_api_key = api_key_data.gemini_api_key
    logger.info("Received request to check API key")
    
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Attempt a simple operation to verify the key
        model.generate_content("Test")
        
        logger.info("API key validation successful")
        return {"status": "valid", "message": "API key is valid"}
    except google.api_core.exceptions.ResourceExhausted as e:
        logger.error(f"Quota exceeded: {str(e)}")
        raise HTTPException(status_code=429, detail="API quota exceeded. Please try again later.")
    except google.api_core.exceptions.PermissionDenied as e:
        logger.error(f"Permission denied: {str(e)}")
        raise HTTPException(status_code=403, detail="Permission denied. Please check your API key.")
    except Exception as e:
        logger.error(f"API key validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid API key: {str(e)}")

@app.post("/speak_to_agents/", response_model=AnalysisResponse)
async def speak_to_agents(
    audio_file: UploadFile = File(...),
    agent_number: int = Form(...),
    history_file: UploadFile = File(...),
    gemini_api_key: str = Form(...)
):
    try:
        genai.configure(api_key=gemini_api_key)
        logger.info("Genai configured with provided API key")

        audio_data = await audio_file.read()
        
        # Save debug audio
        if DEBUG_MODE:
            save_debug_audio(audio_data, format='wav')

        processed_audio = process_audio(audio_data, format='wav')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(processed_audio.getvalue())
            temp_file_path = temp_file.name

        try:
            file_response = genai.upload_file(temp_file_path)
            audio_uri = file_response

            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content([audio_uri, "Transcribe this audio."])
            transcribed_text = response.text
            logger.info(f"Audio transcribed: {transcribed_text[:50]}...")

            if agent_number not in range(5):
                raise HTTPException(status_code=400, detail="Invalid agent number")

            history_content = await history_file.read()
            history = pickle.loads(base64.b64decode(history_content))
            logger.info("History loaded from base64 encoded pickle data")

            agent_prompt = agent_prompts[agent_number]
            agent = genai.GenerativeModel('gemini-1.5-flash', system_instruction=agent_prompt,
                                          generation_config=genai.GenerationConfig(
                                              response_schema=Narration,
                                              response_mime_type="application/json",
                                              temperature=0.1
                                          ))

            agent_chat = agent.start_chat(history=history[agent_number])
            logger.info(f"Agent {agent_number} chat started with history")

            agent_response = agent_chat.send_message(transcribed_text).text
            recent_conversation = format_recent_history(agent_chat.history)

            agent_response_transcription = json.loads(agent_response)['transcription']

            # Monitor agent processing
            monitor_agent_prompt = monitor_agent_prompts[agent_number]
            monitor_data = {
                "latest_response": agent_response,
                "conversation_history": recent_conversation
            }
            monitor_agent_prompt = custom_format_double(monitor_agent_prompt, monitor_data)
            monitor_agent = genai.GenerativeModel('gemini-1.5-flash', system_instruction=monitor_agent_prompt,
                                                  generation_config=genai.GenerationConfig(
                                                      response_schema=Monitoring,
                                                      response_mime_type="application/json",
                                                      temperature=0
                                                  ))
            monitor_agent_chat = monitor_agent.start_chat(history=[])
            
            try:
                monitor_agent_response = monitor_agent_chat.send_message("Monitor_Administrator: Analyze and provide the assessment please").text
            except Exception as e:
                logger.error(f"Monitor agent error: {e}")
                monitor_agent_response = '{"assessment": "There was a problem with the answer.", "transcription": "Stop harassing me", "description": "nothing"}'

            monitor_agent_response_assessment = json.loads(monitor_agent_response)['assessment']

            if monitor_agent_response_assessment == "There was a problem with the answer.":
                logger.info("There was a security issue")
                monitor_agent_response_transcription = json.loads(monitor_agent_response)['transcription']
                agent_response_transcription = monitor_agent_response_transcription

            agent_chat.history[-1].parts[0].text = agent_response_transcription
            logger.info(f"Agent {agent_number} response generated")

            # Update history
            history[agent_number] = agent_chat.history
            
            # Prepare response
            updated_history_pickle = pickle.dumps(history)
            updated_history_base64 = base64.b64encode(updated_history_pickle).decode('utf-8')

            return AnalysisResponse(
                narration=agent_response_transcription,
                updated_history=updated_history_base64,
                status="success"
            )

        finally:
            os.unlink(temp_file_path)

    except Exception as e:
        logger.error(f"Error processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")

@app.post("/talk_to_agents/", response_model=AnalysisResponse)
async def analyze_audio(
    text: str = Form(...),
    agent_number: int = Form(...),
    history_file: UploadFile = File(...),
    gemini_api_key: str = Form(...)
):
    try:
        if agent_number not in range(5):
            raise HTTPException(status_code=400, detail="Invalid agent number")

        genai.configure(api_key=gemini_api_key)
        logger.info("Genai configured with provided API key")

        history_base64 = await history_file.read()
        history = pickle.loads(base64.b64decode(history_base64))
        logger.info("History loaded from base64 encoded pickle data")

        agent_prompt = agent_prompts[agent_number]
        agent = genai.GenerativeModel('gemini-1.5-flash', system_instruction=agent_prompt,
                                      generation_config=genai.GenerationConfig(
                                          response_schema=Narration,
                                          response_mime_type="application/json",
                                          temperature=0.1
                                      ))

        agent_chat = agent.start_chat(history=history[agent_number])
        logger.info(f"Agent {agent_number} chat started with history")
        
        agent_response = agent_chat.send_message(text).text
        recent_conversation = format_recent_history(agent_chat.history)        
        agent_response_transcription = json.loads(agent_response)['transcription']
        
        monitor_agent_prompt = monitor_agent_prompts[agent_number]
        monitor_data = {
            "latest_response": agent_response,
            "conversation_history": recent_conversation
        }
        monitor_agent_prompt = custom_format_double(monitor_agent_prompt, monitor_data)                                                     
        monitor_agent = genai.GenerativeModel('gemini-1.5-flash', system_instruction=monitor_agent_prompt,
                    generation_config=genai.GenerationConfig(
                        response_schema=Monitoring,
                        response_mime_type="application/json",
                        temperature=0
                    ))
        monitor_agent_chat = monitor_agent.start_chat(history=[])
        try:
            monitor_agent_response = monitor_agent_chat.send_message("Monitor_Administrator: Analyze and provide the assessment please").text
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            monitor_agent_response = '{"assessment": "There was a problem with the answer.", "transcription": "Stop harassing me", "description": "nothing"}'

        monitor_agent_response_assessment = json.loads(monitor_agent_response)['assessment']
        
        if monitor_agent_response_assessment == "There was a problem with the answer.":
            logger.info("There was a security issue")
            monitor_agent_response_transcription = json.loads(monitor_agent_response)['transcription']
            agent_response_transcription = monitor_agent_response_transcription

        agent_chat.history[-1].parts[0].text = agent_response_transcription
        logger.info(f"Agent {agent_number} response generated")

        # Update history
        history[agent_number] = agent_chat.history
        logger.info(agent_chat.history)
        
        # Pickle and encode updated history
        updated_history_pickle = pickle.dumps(history)
        updated_history_base64 = base64.b64encode(updated_history_pickle).decode('utf-8')
        
        return AnalysisResponse(
            narration=agent_response_transcription,
            updated_history=updated_history_base64,
            status="success"
        )

    except Exception as e:
        logger.error(f"Error processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)