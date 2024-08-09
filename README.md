# HoloBar BackEnd

## Overview

This FastAPI application provides an interface for interacting with AI agents representing characters in a bar setting. Users can communicate with these agents through both text and voice inputs. The system includes a unique monitoring feature to ensure agent responses adhere to their character scripts.

## Features

- Multiple AI agents with distinct personalities
- Voice and text-based interactions
- Real-time audio transcription
- Conversation history management
- Response monitoring and safety checks
- Google AI (Gemini) integration

## Endpoints

1. **API Key Validation**
   - `POST /check_api_key/`
   - Validates the provided Google AI (Gemini) API key

2. **Voice Interaction**
   - `POST /speak_to_agents/`
   - Accepts audio input, transcribes it, and processes it through the chosen AI agent

3. **Text Interaction**
   - `POST /talk_to_agents/`
   - Accepts text input and processes it through the chosen AI agent

## How It Works

1. **User Input**: The user sends either text or audio to interact with a chosen AI agent (bar character).

2. **Agent Processing**: The selected agent processes the input and generates a response based on their unique personality and conversation history.

3. **Monitoring**: A separate monitoring system checks the agent's response to ensure it aligns with the character's script and personality.

4. **Response Delivery**: The API returns the agent's response, which may be modified if the monitor detects any issues.

5. **History Update**: The conversation history is updated and returned for maintaining context in future interactions.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (including Google AI API key)
4. Run the server: `uvicorn main:app --reload`

## Usage

To interact with an agent:

1. Validate your API key using the `/check_api_key/` endpoint
2. Send a POST request to either `/speak_to_agents/` (for audio) or `/talk_to_agents/` (for text)
3. Include the agent number, input (audio file or text), conversation history, and API key in your request
4. Receive the agent's response and updated conversation history

## Security

- API key validation for each request
- Response monitoring to ensure appropriate content
- Error handling and logging for troubleshooting

## Development

- Debug mode available for testing and development
- Logging implemented for monitoring application behavior

## Note

This API is designed for a specific use case involving AI characters in a bar setting. Ensure all content and interactions are appropriate for your intended audience.