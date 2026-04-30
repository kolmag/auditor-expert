"""
app.py — Auditor Expert Gradio interface
"""

import os
import sys
import asyncio

import gradio as gr
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from answer import answer_stream

langfuse = Langfuse(
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

SYSTEM_DESCRIPTION = """**Auditor Expert** — ISO 9001 / IATF 16949 / AS9100 audit knowledge base

Ask questions about:
- NCR grading (observation / minor / major / critical)
- Clause requirements and audit evidence
- Corrective action and closure requirements
- Supplier audit methodology
- IATF 16949 automotive-specific requirements
- AS9100D aerospace additions
- Semiconductor manufacturing quality context

*Answers are grounded in the knowledge base. Sources cited where applicable.*"""

PLACEHOLDER = "e.g. What evidence do I need to close a major NCR? / Is missing a reaction plan a major or minor? / What does IATF 16949 require for process audits?"


async def respond(message: str, history: list):
    """Gradio streaming response handler (Async)."""
    try:
        langfuse.create_event(
            name="chat_turn",
            input={"question": message},
            metadata={"history_length": len(history)}
        )
    except Exception:
        pass

    # Await the async generator appropriately
    async for partial in answer_stream(message, history=history):
        yield partial

    try:
        langfuse.flush()
    except Exception:
        pass


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Auditor Expert", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🔍 Auditor Expert")
    gr.Markdown(SYSTEM_DESCRIPTION)

    with gr.Row():
        # Left column for the chat (takes up 2/3 of the space)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Audit Knowledge Base",
                height=650,
                type="messages"
            )

        # Right column for inputs, controls, and examples (takes up 1/3 of the space)
        with gr.Column(scale=1):
            msg = gr.Textbox(
                placeholder=PLACEHOLDER,
                label="Your question",
                lines=3,
            )
            
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.ClearButton([msg, chatbot], value="Clear")
                
            gr.Markdown(
                "*Powered by text-embedding-3-small + BGE reranker + GPT-OSS-120B via Groq*"
            )

            gr.Examples(
                examples=[
                    "What are the four required elements of a well-written NCR?",
                    "When does a minor NCR become a major? What is the escalation threshold?",
                    "What evidence do I need to close a major NCR from a supplier audit?",
                    "What does IATF 16949 clause 9.2.2 require for internal audit programme?",
                    "What is the difference between a process audit and a system audit?",
                    "How do I handle a supplier who refuses access to a process area during audit?",
                    "What GR&R percentage makes a measurement system unacceptable?",
                    "What AS9100D additions are most commonly found as major NCRs at certification?",
                    "Is SPC without a reaction plan a minor or major NCR under IATF 16949?",
                    "What are the required elements of a corrective action plan for a major NCR?",
                ],
                inputs=msg,
                label="Example questions"
            )

    async def submit(message, history):
        # Guard against empty submissions
        if not message or not message.strip():
            yield history, ""
            return
            
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        yield history, ""
        
        # Iterate streaming answer using async for
        async for partial in respond(message, history[:-2]):
            history[-1]["content"] = partial
            yield history, ""

    msg.submit(submit, [msg, chatbot], [chatbot, msg])
    submit_btn.click(submit, [msg, chatbot], [chatbot, msg])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )