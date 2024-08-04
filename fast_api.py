from prompts import *
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import logging
import google.generativeai as genai
import pickle
import json
import base64
from pydantic import BaseModel
import tempfile
import os

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

class Narration(BaseModel):
    transcription: str
    description: str

class AnalysisResponse(BaseModel):
    narration: str
    updated_history: str  # Base64 encoded pickle data
    status: str
@app.get("/")
async def root():
    return {"message": "Welcome to the audio analysis API"}

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
        agent_model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=agent_prompt,
                                            generation_config=genai.GenerationConfig(
                                                response_schema=Narration,
                                                response_mime_type="application/json",
                                                temperature=0.1
                                            ))
        
        agent_chat = agent_model.start_chat(history=history[agent_number])
        logger.info(f"Agent {agent_number} chat started with history")
        agent_response = json.loads(agent_chat.send_message(text).text)['transcription']
        logger.info(agent_response)
        agent_chat.history[-1].parts[0].text = agent_response
        logger.info(f"Agent {agent_number} response generated")

        # Update history
        history[agent_number] = agent_chat.history
        logger.info(agent_chat.history)
        # Prepare response
        
        # Pickle and encode updated history
        updated_history_pickle = pickle.dumps(history)
        updated_history_base64 = base64.b64encode(updated_history_pickle).decode('utf-8')
        
        return AnalysisResponse(
            narration=agent_response,
            updated_history=updated_history_base64,
            status="success"
        )

    except Exception as e:
        logger.error(f"Error processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)