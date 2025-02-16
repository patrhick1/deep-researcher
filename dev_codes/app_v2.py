#app.py
import os
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from agent import (
    initialize_agent,       # Function to compile and return the reporter_agent (a CompiledStateGraph)
    ResearchChatbot,
    ask_for_clarification,    # Clarification function
    update_clarification    # Function to update state with user's clarification
)
load_dotenv()

app = FastAPI()

# Mount the static folder (with our custom CSS and JS files)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the agent and chatbot once at startup.
reporter_agent = asyncio.run(initialize_agent())
chatbot = ResearchChatbot(reporter_agent)

# Global in-memory conversation state for demonstration.
# In practice, you might store session-specific state.
conversation_state = {"topic": "", "filename": ""}

@app.get("/", response_class=HTMLResponse)
async def get_chat():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Research Report Chatbot</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <h1>Research Report Chatbot</h1>
        <div id="chatbox"></div>
        <div id="input-area">
            <input type="text" id="user-input" placeholder="Enter your message here...">
            <button id="send-btn">Send</button>
        </div>
        <br>
        <button id="download-btn">Download Report</button>
        
        <script src="/static/script.js"></script>
        <script>
            const chatbox = document.getElementById("chatbox");
            const userInput = document.getElementById("user-input");
            const sendBtn = document.getElementById("send-btn");
            const downloadBtn = document.getElementById("download-btn");

            function addMessage(sender, message) {
                const div = document.createElement("div");
                div.className = "message " + (sender === "User" ? "user" : "agent");
                div.innerHTML = "<strong>" + sender + ":</strong> " + message;
                chatbox.appendChild(div);
                chatbox.scrollTop = chatbox.scrollHeight;
            }

            async function sendMessage() {
                const text = userInput.value;
                if (!text) return;
                addMessage("User", text);
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({message: text})
                });
                const data = await response.json();
                addMessage("Agent", data.reply);
                userInput.value = "";
            }

            sendBtn.addEventListener("click", sendMessage);
            userInput.addEventListener("keydown", function(event) {
                if (event.key === "Enter") sendMessage();
            });

            
            async function downloadReport() {
                const res = await fetch("/download_report?ts=" + Date.now());
                if (res.ok) {
                    // 1. Extract text content (the file data)
                    const content = await res.text();

                    // 2. Parse filename from Content-Disposition if it exists
                    let filename = "report.md";  // default
                    const disposition = res.headers.get("Content-Disposition");
                    if (disposition) {
                        // Typically looks like: attachment; filename="your_report_name.md"
                        const match = /filename="([^"]+)"/.exec(disposition);
                        if (match && match[1]) {
                            filename = match[1];
                        }
                    }

                    // 3. Try the File System Access API first
                    if (window.showSaveFilePicker) {
                        try {
                            const options = {
                                suggestedName: filename,
                                types: [{
                                    description: 'Markdown File',
                                    accept: {'text/markdown': ['.md']},
                                }],
                            };
                            const handle = await window.showSaveFilePicker(options);
                            const writable = await handle.createWritable();
                            await writable.write(content);
                            await writable.close();
                            alert("Report saved successfully!");
                        } catch (err) {
                            console.error("Error saving file:", err);
                        }
                    } else {
                        // 4. Fallback: create a download link with a dynamic filename
                        const blob = new Blob([content], {type: "text/markdown"});
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = filename; // ← Use parsed filename here
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                        window.URL.revokeObjectURL(url);
                    }
                } else {
                    alert("Report not found.");
                }
            }

            downloadBtn.addEventListener("click", downloadReport);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Updated call_planner_agent to return the full event dictionary.
async def call_planner_agent(agent, prompt, config={"recursion_limit": 50}):
    from rich.console import Console
    from rich.markdown import Markdown as RichMarkdown
    console = Console()
    events = agent.astream({'topic': prompt}, config, stream_mode="values")
    async for event in events:
        if 'final_report' in event:
            md = RichMarkdown(event['final_report'])
            console.print(md)
            return event  # Return the entire event dictionary

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_message = data.get("message", "")

    # For demonstration, we keep everything in a conversation_state dict
    # that matches the structure of ReportState
    # e.g. conversation_state = {"topic": "", "clarifications": [], ...}

    # 1. If we have never started, set the base topic to user_message
    if not conversation_state.get("topic"):
        conversation_state["topic"] = user_message
        conversation_state["clarifications"] = []
        conversation_state["clarification_attempts"] = 0
        updated_state = await ask_for_clarification(conversation_state)
        return JSONResponse({"reply": updated_state["clarifying_question"]})

    # 2. If we're still awaiting a clarification
    elif conversation_state.get("awaiting_clarification"):
        updated_state = await update_clarification(conversation_state, user_message)
        if updated_state.get("awaiting_clarification"):
            # Still not enough detail
            return JSONResponse({"reply": updated_state["clarifying_question"]})
        # else, done clarifying => we can proceed to final step

    # 3. If no clarifications needed, or we've just finished clarifications,
    # pass everything to the chatbot handle_input
    final_report_response = await chatbot.handle_input(conversation_state)

    if "reply" in final_report_response:
        # Possibly handle the partial or final response
        return JSONResponse({"reply": final_report_response["reply"]})

    # If we got the final report, store the filename, etc.
    conversation_state["filename"] = final_report_response.get("filename")
    return JSONResponse({"reply": final_report_response.get("final_report")})


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
        return FileResponse(file_path, filename=filename, media_type="text/markdown",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
                            background=background_tasks)
    return JSONResponse(content={"error": "Report not found."}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
