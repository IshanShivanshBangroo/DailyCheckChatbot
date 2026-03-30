# Daily Check-in Chatbot

## Overview
This project implements a Streamlit-based daily check-in chatbot. The app collects daily depression level from 1 to 10, gives deterministic range-based feedback, asks for an optional note, generates a bounded supportive follow-up response with an OpenAI model and then supports multi-turn empathic chat.

## Key features
- Strict level validation for supported inputs such as `6`, `log 6`, `level 7`, `lvl7`, `7/10`, and `7 out of 10`
- Deterministic Python feedback for the 1-3, 4-6, 7-8, and 9-10 ranges
- Optional note support with `skip` path
- Supportive LLM response that uses the stored level and note
- Multi-turn chat with recent history preserved in `st.session_state`
- Crisis-language detection which bypasses the LLM and returns fixed safety-oriented message
- Export chat history as JSON
- Clear chat history and restart the flow
- No OpenAI API key stored in the source file

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Paste a valid OpenAI API key into the sidebar.
4. Click **Change API Key**.
5. Use the chat interface to enter the level, optional note, and follow-up messages.

## Safety and risk-reduction choices
- Step 2 uses deterministic Python logic instead of the LLM.
- The parser uses strict full-string regular expressions instead of scanning for digits anywhere in the input.
- The model prompt is bounded and uses only a fixed list of low-risk coping suggestions.
- Crisis-language checks run before every model call.
- The app is framed as a classroom prototype, not a clinician or crisis service.

## Known limitations
- Crisis detection is keyword-based and may miss subtle or indirect language.
- Later chat responses still depend on model availability and behavior.
- The app does not currently provide location-aware crisis resources.
- The system is designed for coursework, not real-world mental health deployment.
