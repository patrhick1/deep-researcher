#app.py

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os, asyncio, uvicorn
from dotenv import load_dotenv
from agent import (
    initialize_agent,
    ResearchChatbot,
    ask_for_clarification,
    update_clarification,
    handle_confirmation
)

load_dotenv()

app = FastAPI()

# Mount the static folder (with our custom CSS and JS files)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize the agent and chatbot once at startup.
reporter_agent = asyncio.run(initialize_agent())
chatbot = ResearchChatbot(reporter_agent)

# Global in-memory conversation state for demonstration.
conversation_state = {"topic": "", "filename": "", "config": {}}

# New configuration model
class ReportConfig(BaseModel):
    research_type: str
    target_audience: str
    structure: str
    section_word_limit: int
    writing_style: str

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    # Render "index.html" from templates folder
    return templates.TemplateResponse("index.html", {"request": request})

# New endpoint to accept configuration
@app.post("/config")
async def config_endpoint(config: ReportConfig):
    global conversation_state
    conversation_state["config"] = config.model_dump()
    print(f"Configuration received: {conversation_state['config']}")
    return JSONResponse({"message": "Configuration saved"})

# Reset endpoint to clear conversation state
@app.post("/reset")
async def reset_conversation():
    """Reset the conversation state to start a new report."""
    global conversation_state
    # Reset to empty initial state
    conversation_state = {"topic": "", "filename": "", "config": {}}
    return JSONResponse({"message": "Conversation reset successfully"})

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    global conversation_state
    
    # 1. Handle confirmation responses
    if conversation_state.get("awaiting_confirmation"):
        updated_state = await handle_confirmation(conversation_state, user_message)
        conversation_state.update(updated_state)
        
        if conversation_state.get("awaiting_confirmation"):
            return JSONResponse({
                "reply": conversation_state["confirmation_summary"],
                "type": "confirmation"
            })
        
        # If confirmation was accepted, proceed to report generation
        return await proceed_to_report_generation(conversation_state)
    
    # 2. Handle feedback responses
    if conversation_state.get("awaiting_feedback_response"):
        # Store user's response as clarification
        conversation_state["clarifications"].append(user_message)
        conversation_state.pop("awaiting_feedback_response", None)
        
        # Re-evaluate with updated clarifications
        updated_state = await update_clarification(conversation_state, user_message)
        conversation_state.update(updated_state)
        
        if "last_feedback" in conversation_state:
            return JSONResponse({
                "reply": conversation_state["last_feedback"],
                "type": "feedback"
            })
    
    # 3. Initial topic setup
    if not conversation_state.get("topic"):
        conversation_state["topic"] = user_message
        conversation_state["clarifications"] = []
        updated_state = await ask_for_clarification(conversation_state)
        conversation_state.update(updated_state)
        return JSONResponse({
            "reply": conversation_state["clarifying_question"],
            "type": "clarification"
        })
    
    # 4. Handle ongoing clarification process
    if conversation_state.get("awaiting_clarification"):
        updated_state = await update_clarification(conversation_state, user_message)
        conversation_state.update(updated_state)
        
        if "last_feedback" in updated_state:
            return JSONResponse({
                "reply": updated_state["last_feedback"],
                "type": "feedback"
            })
            
        if "confirmation_summary" in updated_state:
            return JSONResponse({
                "reply": updated_state["confirmation_summary"],
                "type": "confirmation"
            })
    
    # 5. Proceed to report generation if all checks passed
    return await proceed_to_report_generation(conversation_state)

async def proceed_to_report_generation(state):
    """Handle the final report generation process"""
    try:
        final_report_response = await chatbot.handle_input(state)
        
        if "final_report" in final_report_response:
            state["filename"] = final_report_response.get("filename")
            return JSONResponse({
                "reply": final_report_response["final_report"],
                "type": "final_report"
            })
            
        return JSONResponse({
            "reply": "Starting report generation process...",
            "type": "status"
        })
        
    except Exception as e:
        return JSONResponse({
            "reply": f"Error generating report: {str(e)}",
            "type": "error"
        })

@app.get("/download_report")
async def download_report(background_tasks: BackgroundTasks):
    filename = conversation_state.get("filename")
    print(f"Looking for file: {filename}")
    if not filename:
        return JSONResponse(content={"error": "Report not generated yet."}, status_code=404)
    
    file_path = os.path.join(os.getcwd(), filename)
    print(f"Looking for file at: {file_path}")
    
    # Schedule deletion of the file after the response is sent
    background_tasks.add_task(os.remove, file_path)
    
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            filename=filename, 
            media_type="text/markdown",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            background=background_tasks
        )
    
    return JSONResponse(content={"error": "Report not found."}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)