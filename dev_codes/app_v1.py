# app.py
import os
import asyncio
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
from agent import (
    initialize_agent,  # a function that compiles your StateGraph
    ResearchChatbot,
    call_planner_agent  # if needed
)

load_dotenv()

app = FastAPI()

# Initialize your agent once at startup
# (Ensure initialize_agent() returns your compiled reporter_agent)
reporter_agent = asyncio.run(initialize_agent())
chatbot = ResearchChatbot(reporter_agent)

@app.get("/", response_class=HTMLResponse)
async def get_form():
    html_content = """
    <html>
      <head>
         <title>Research Report Generator</title>
      </head>
      <body>
         <h1>Generate Your Report</h1>
         <form action="/generate" method="post">
           <label for="topic">Research Topic:</label>
           <input type="text" id="topic" name="topic" required><br><br>
           <label for="filename">Desired File Name (e.g., report.md):</label>
           <input type="text" id="filename" name="filename" required><br><br>
           <input type="submit" value="Generate Report">
         </form>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/generate")
async def generate_report(topic: str = Form(...), filename: str = Form(...)):
    # Run the chatbot interface which includes the clarification step
    final_report = await chatbot.handle_input(topic)
    
    # Save the generated Markdown report to a temporary file
    file_path = os.path.join("/tmp", filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    
    # Return the file as a download
    return FileResponse(file_path, filename=filename, media_type="text/markdown")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
