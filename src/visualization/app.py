"""
src/visualization/app.py
========================
RAGApp -- Production-grade Gradio dashboard for the RAG pipeline.

A custom 3-panel layout designed to mimic ChatGPT/Claude modern chat interfaces.
Includes a conversation history sidebar (Left), a central chat panel, 
and a dynamic RAG context inspector sidebar (Right).
"""

import os
import sys
import json
import time
import dataclasses
from typing import Generator, List, Dict, Any

import gradio as gr

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import config
from logger import get_logger
from models.predict_model import Generator as BackendGenerator
from preprocessing_data.pre_processing import Chunker
from models.train_model import LangChainVectorStore
from models.retriever import HybridRetriever

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────── #
# Custom CSS -- Advanced 3-Panel GPT-style UI
# ─────────────────────────────────────────────────────────────────────────── #
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --body-background-fill: #f9f9f7;
    --block-background-fill: #ffffff;
    --border-color-primary: rgba(0,0,0,0.12);
    --color-accent: #534AB7;
    --button-primary-background-fill: #534AB7;
    --button-primary-text-color: #ffffff;

    --panel-bg: #f4f3ef;
    --chat-user-bg: #EEEDFE;
    --chat-user-border: #AFA9EC;
    --chat-user-text: #26215C;
    
    --chat-bot-bg: #ffffff;
    --chat-bot-border: #e5e5e3;
    --font-family: 'Inter', sans-serif;
}

body, .gradio-container {
    font-family: var(--font-family) !important;
    background-color: var(--body-background-fill) !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* Strip main app padding */
.gradio-container > .main { padding: 0 !important; max-width: 100% !important; margin: 0 !important; height: 100vh !important;}

.main-wrap {
    display: flex !important;
    height: 100vh !important;
    flex-wrap: nowrap !important;
    gap: 0 !important;
    margin: 0 !important;
}

/* Layout Columns */
#left-panel { min-width: 220px !important; max-width: 220px !important; background: var(--panel-bg); height: 100vh; overflow-y: auto; padding: 15px; border-right: 1px solid var(--border-color-primary); flex-shrink: 0; }
#center-panel { flex: 1 !important; height: 100vh; display: flex; flex-direction: column; background: var(--block-background-fill); overflow-y: hidden; min-width: 0; position: relative;}
#right-panel { min-width: 300px !important; max-width: 300px !important; background: var(--panel-bg); height: 100vh; overflow-y: auto; padding: 15px; border-left: 1px solid var(--border-color-primary); flex-shrink: 0; }

/* Left panel text tweaks */
.sidebar-title { font-size: 16px; font-weight: bold; margin-bottom: 20px; color: #111;}
.sidebar-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: #888; margin-top: 15px; margin-bottom: 5px; }
.history-item { display: block; font-size: 12px; padding: 8px; border-radius: 8px; cursor: pointer; margin-bottom: 5px; background: transparent; border: none; text-align: left; width: 100%; transition: background 0.2s; color: #333;}
.history-item:hover { background: rgba(0,0,0,0.05); }
.history-item.active { background: #e0e0e0; font-weight: bold; }
.history-time { display: block; font-size: 10px; color: #999; margin-top: 2px; font-weight: normal;}
.kb-status { font-size: 11px; margin-top: auto; padding-top: 20px; color: #555; display: flex; align-items: center; gap: 6px;}
.dot { height: 8px; width: 8px; background-color: #639922; border-radius: 50%; display: inline-block; }

/* Center panel chat area */
#chat-header { border-bottom: 1px solid var(--border-color-primary); padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; background: white;}
#chat-title { font-weight: 500; font-size: 14px; color: #333;}
.badge { background: #E6F1FB; color: #0C447C; border: 1px solid #85B7EB; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 500;}
.badge.green { background: #eaf5e1; color: #2d5a0d; border-color: #639922; }

#chat-scroller { flex: 1 !important; overflow-y: auto !important; padding: 20px !important; }
.gradio-chatbot { height: 100% !important; border: none !important; background: transparent !important;}

/* Fix chatbot bubbles */
.message.user { background-color: var(--chat-user-bg) !important; border-color: var(--chat-user-border) !important; color: var(--chat-user-text) !important; border-radius: 10px !important; }
.message.bot { background-color: var(--chat-bot-bg) !important; border-color: var(--chat-bot-border) !important; border-radius: 10px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;}

/* Input area */
#input-wrapper { background: var(--block-background-fill); border-top: 1px solid var(--border-color-primary); margin: 0; display: flex; flex-direction: column;}
.chip-row { padding: 10px 20px 0 20px; display: flex; gap: 8px; flex-wrap: wrap;}
.chip { background: transparent; border: 1px solid var(--border-color-primary); color: #555; padding: 6px 12px; border-radius: 16px; font-size: 12px; cursor: pointer; transition: all 0.2s; white-space: nowrap;}
.chip:hover { background: #f0f0f0; border-color: #ccc;}

/* Right panel cards */
.chunk-card { background: #ffffff; border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px; margin-bottom: 12px; font-size: 11px; color: #444; box-shadow: 0 1px 2px rgba(0,0,0,0.02);}
.chunk-header { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 8px; color: #222;}
.score-green { color: #639922; }
.score-amber { color: #EF9F27; }
.score-red { color: #E24B4A; }
.progress-bar-bg { width: 100%; height: 5px; background: #eee; border-radius: 3px; margin-bottom: 10px; overflow: hidden;}
.progress-bar-fill { height: 100%; }
.bg-green { background: #639922; }
.bg-amber { background: #EF9F27; }
.bg-red { background: #E24B4A; }
.pills-row { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.pill { background: #f0f0f0; border: 1px solid #e0e0e0; padding: 2px 6px; border-radius: 4px; font-size: 9px; color: #666;}
.preview-text { font-style: italic; color: #555; border-left: 2px solid var(--color-accent); padding-left: 8px; margin-top: 5px; line-height: 1.4;}

/* Citations inside bot message */
.citation-pill { display: inline-block; background: #E6F1FB; color: #0C447C; border: 1px solid #85B7EB; border-radius: 10px; padding: 2px 8px; font-size: 10px; margin-right: 5px; margin-top: 10px; cursor: pointer; font-weight: 500;}

/* Right Panel Meta table */
.meta-card { background: #fff; border-radius: 10px; padding: 12px; border: 1px solid var(--border-color-primary); margin-bottom: 15px;}
.metadata-table { font-size: 10px; font-family: monospace; width: 100%; border-collapse: collapse;}
.metadata-table td { padding: 6px 4px; border-bottom: 1px solid #f0f0f0; }
.metadata-table tr:last-child td { border-bottom: none; }
.metadata-label { color: #888; }
.metadata-value { text-align: right; color: #222; font-weight: 600;}

footer { display: none !important; }
"""

# ─────────────────────────────────────────────────────────────────────────── #
# UI DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────── #

@dataclasses.dataclass
class ChunkResult:
    chunk_id: str
    source_title: str
    source_type: str        
    publish_date: str       
    chunk_index: int        
    total_chunks: int       
    similarity_score: float 
    similarity_metric: str  
    content_preview: str    
    full_content: str       


@dataclasses.dataclass
class RAGResult:
    answer: str
    chunks: list[ChunkResult]
    model_name: str
    tokens_used: int
    latency_seconds: float
    confidence_score: float 
    top_k: int
    score_threshold: float


# ─────────────────────────────────────────────────────────────────────────── #
# HTML TEMPLATES FOR DYNAMIC CARDS
# ─────────────────────────────────────────────────────────────────────────── #

def get_color_class(score: float) -> str:
    if score >= 0.85: return "green"
    elif score >= 0.70: return "amber"
    else: return "red"

def build_right_panel_html(result: RAGResult) -> str:
    """Creates the HTML for the Right Sidebar metadata and chunks."""
    if not result:
        return "<div style='color: #888; font-size: 12px; text-align: center; margin-top: 50px;'>No context retrieved yet.<br>Ask a question to see RAG internals here.</div>"
    
    overall_color = get_color_class(result.confidence_score)
    
    # Section A: Meta Board
    html = f"""
    <div class="sidebar-title">RAG Context Inspector</div>
    <div class="meta-card">
        <table class="metadata-table">
            <tr><td class="metadata-label">Confidence</td><td class="metadata-value score-{overall_color}">{result.confidence_score:.2f}</td></tr>
            <tr><td class="metadata-label">LLM Model</td><td class="metadata-value">{result.model_name}</td></tr>
            <tr><td class="metadata-label">Tokens (est)</td><td class="metadata-value">{result.tokens_used}</td></tr>
            <tr><td class="metadata-label">Latency</td><td class="metadata-value">{result.latency_seconds:.2f}s</td></tr>
            <tr><td class="metadata-label">top_k</td><td class="metadata-value">{result.top_k}</td></tr>
            <tr><td class="metadata-label">threshold</td><td class="metadata-value">{result.score_threshold}</td></tr>
        </table>
    </div>
    <div class="sidebar-label">RETRIEVED CHUNKS ({len(result.chunks)})</div>
    """

    # Section B: Chunk Cards
    for c in result.chunks:
        clr = get_color_class(c.similarity_score)
        html += f"""
        <div class="chunk-card" id="chunk_{c.chunk_id}">
            <div class="chunk-header">
                <span title="{c.source_title}">{c.source_title[:35]}{"..." if len(c.source_title)>35 else ""}</span>
                <span class="score-{clr}">{c.similarity_score:.2f}</span>
            </div>
            <div class="progress-bar-bg"><div class="progress-bar-fill bg-{clr}" style="width: {c.similarity_score * 100}%;"></div></div>
            <div class="pills-row">
                <span class="pill">{c.source_type}</span>
                <span class="pill">Chunk {c.chunk_index}/{c.total_chunks}</span>
                <span class="pill">{c.similarity_metric}</span>
            </div>
            <div class="preview-text">{c.content_preview}...</div>
        </div>
        """
    return html

def build_left_panel_html() -> str:
    return """
    <div class="sidebar-title">RAG QnA Bot</div>
    <div class="sidebar-label">Today</div>
    <div class="history-item active">
        What is the current state...
        <span class="history-time">Active now</span>
    </div>
    
    <div style="flex-grow: 1;"></div>
    
    <div class="kb-status">
        <span class="dot"></span> Pinecone Vector Store<br>
        (HotpotQA) connected
    </div>
    """

# ─────────────────────────────────────────────────────────────────────────── #
# GRADIO APPLICATION
# ─────────────────────────────────────────────────────────────────────────── #

class RAGApp:
    """Production-grade layout implementing a 3-column GPT-style Chat UI."""

    def __init__(self) -> None:
        logger.info("Initialising standalone UI with real pipeline connection...")
        self._retriever = self._build_retriever()
        self._generator = BackendGenerator()
        
        self.last_rag_result = None
        self._demo = self._build_ui()

    def _build_retriever(self) -> HybridRetriever:
        chunker = Chunker()
        chunks = chunker.load_chunks_from_disk()
        if not chunks:
            logger.warning("No chunks found -- run ingest.")
            return None
        documents = Chunker.to_langchain_documents(chunks)
        store = LangChainVectorStore()
        store.connect_existing()
        return HybridRetriever(documents=documents, langchain_store=store)

    def _prepare_rag_response(self, user_message: str, top_k: int, threshold: float, model: str) -> RAGResult:
        """Calls actual pipeline and builds RAGResult dataclass."""
        start_time = time.time()
        
        # 1. Retrieval
        if not self._retriever:
            return RAGResult("No data ingested. Run `python src/main.py --ingest`.", [], model, 0, 0, 0, top_k, threshold)
            
        raw_chunks = self._retriever.retrieve(user_message)[:top_k]
        
        # Filter by mocked threshold for UI demonstration
        chunk_results = []
        for i, rc in enumerate(raw_chunks):
            score = rc.get("score", 0.0)
            if score < threshold:
                continue
            
            chunk_results.append(ChunkResult(
                chunk_id=str(i),
                source_title=rc.get("title", rc.get("source", "Unknown Source")),
                source_type="Wiki / Web",
                publish_date="N/A",
                chunk_index=1,
                total_chunks=1,
                similarity_score=score,
                similarity_metric="cosine",
                content_preview=rc.get("text", "")[:150],
                full_content=rc.get("text", "")
            ))
            
        # 2. Generation using real LLM setup
        gen_result = self._generator.generate(query=user_message, chunks=raw_chunks)
        answer = gen_result.get("answer", "Failed to generate.")
        
        latency = time.time() - start_time
        
        conf = 0.0
        if chunk_results:
            conf = sum([c.similarity_score for c in chunk_results[:3]]) / min(len(chunk_results), 3)
            
        return RAGResult(
            answer=answer,
            chunks=chunk_results,
            model_name=model,
            tokens_used=len(answer)//4 + sum(len(c.full_content)//4 for c in chunk_results),
            latency_seconds=latency,
            confidence_score=conf,
            top_k=top_k,
            score_threshold=threshold
        )

    def _respond(self, user_message: str, chat_history: list, top_k: int, threshold: float, model: str, temp: float):
        """Streaming generator function for UI feedback."""
        if not user_message.strip():
            yield chat_history, "", "", "0 chunks retrieved", build_right_panel_html(None)
            return

        chat_history = chat_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": ""}]
        yield chat_history, "", "Generating...", "Thinking...", ""

        # Fetch answer sync from backend
        self.last_rag_result = self._prepare_rag_response(user_message, top_k, threshold, model)
        
        if not self.last_rag_result.chunks:
            chat_history[-1]["content"] = self.last_rag_result.answer
            yield chat_history, "", "Active session", "0 chunks retrieved", build_right_panel_html(self.last_rag_result)
            return

        # Build answer with inline citations
        ans = self.last_rag_result.answer
        
        # Stream UI effect
        current_ans = ""
        words = ans.split(" ")
        for i, word in enumerate(words):
            current_ans += word + " "
            chat_history[-1]["content"] = current_ans + "▌"
            # Optional UI sleep for typewriter effect
            time.sleep(0.01)
            yield chat_history, "", "Active session", f"{len(self.last_rag_result.chunks)} chunks retrieved", build_right_panel_html(self.last_rag_result)
        
        # Remove pipe, add pill citations natively supported by grad CSS
        final_answer = current_ans.strip() + "\n\n"
        for i, c in enumerate(self.last_rag_result.chunks):
            final_answer += f'<span class="citation-pill" onclick="document.getElementById(\'chunk_{c.chunk_id}\').scrollIntoView();">[{i+1}] {c.source_title[:20]}</span>'
        
        chat_history[-1]["content"] = final_answer
        yield chat_history, "", "Active session", f"Badge: {len(self.last_rag_result.chunks)} chunks retrieved (Conf: {self.last_rag_result.confidence_score:.2f})", build_right_panel_html(self.last_rag_result)

    def _export_json(self):
        """Dumps current result chunk data to JSON file."""
        if not self.last_rag_result:
            return None
        file_path = "rag_context.json"
        
        dump_data = {
            "answer": self.last_rag_result.answer,
            "chunks": [dataclasses.asdict(c) for c in self.last_rag_result.chunks],
            "meta": {
                "latency": self.last_rag_result.latency_seconds,
                "confidence": self.last_rag_result.confidence_score
            }
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=4)
        return file_path

    def _build_ui(self) -> gr.Blocks:
        with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
            gr.Markdown("<style>footer {visibility: hidden}</style>") # Fallback footer hide

            with gr.Row(elem_classes=["main-wrap"]):
                
                # ── PANEL 1: LEFT SIDEBAR (History) ────────────────── #
                with gr.Column(elem_id="left-panel"):
                    btn_new = gr.Button("+ New Chat", variant="primary", size="sm")
                    left_html = gr.HTML(build_left_panel_html())

                # ── PANEL 2: CENTER CHAT ───────────────────────────── #
                with gr.Column(elem_id="center-panel"):
                    with gr.Row(elem_id="chat-header"):
                        chat_title = gr.HTML("<div id='chat-title'>New Conversation</div>")
                        chat_badge = gr.HTML("<div class='badge green' id='chat-badge'>Ready</div>")

                    with gr.Column(elem_id="chat-scroller"):
                        chatbot = gr.Chatbot(
                            value=[],
                            elem_classes=["gradio-chatbot"],
                            show_label=False,
                            render_markdown=True,
                            layout="bubble"
                        )
                    
                    with gr.Column(elem_id="input-wrapper"):
                        # Starter chips
                        with gr.Row(elem_classes=["chip-row"]) as chip_row:
                            gr.HTML("<span class='chip' onclick='document.querySelector(\"textarea\").value=\"What is the architecture of RAG?\";'>What is the architecture of RAG?</span>")
                            gr.HTML("<span class='chip' onclick='document.querySelector(\"textarea\").value=\"Summarize latest trends.\";'>Summarize latest trends.</span>")

                        with gr.Row():
                            msg_input = gr.Textbox(
                                show_label=False,
                                placeholder="Ask a question about the knowledge base...",
                                lines=2,
                                max_lines=6,
                                scale=8
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                            
                        with gr.Accordion("Retrieval settings", open=False):
                            with gr.Row():
                                top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Top-k chunks")
                                threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.0, label="Min similarity threshold")
                            with gr.Row():
                                model_dd = gr.Dropdown(choices=["llama-3.3-70b-versatile"], value="llama-3.3-70b-versatile", label="LLM model")
                                temp_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.2, label="Temperature")

                # ── PANEL 3: RIGHT RAG CONTEXT ─────────────────────── #
                with gr.Column(elem_id="right-panel"):
                    right_html = gr.HTML(build_right_panel_html(None))
                    
                    with gr.Row():
                        copy_btn = gr.Button("Copy context", size="sm")
                        down_btn = gr.Button("Download JSON", size="sm")
                        file_out = gr.File(visible=False, label="Download")

            # ── EVENTS ─────────────────────────────────────────── #
            
            # Hide chips on first send
            def hide_chips():
                return gr.update(visible=False)

            submit_inputs = [msg_input, chatbot, top_k_slider, threshold_slider, model_dd, temp_slider]
            submit_outputs = [chatbot, msg_input, chat_title, chat_badge, right_html]

            # Chat events
            msg_input.submit(hide_chips, [], [chip_row]).then(
                fn=self._respond,
                inputs=submit_inputs,
                outputs=submit_outputs
            )
            
            send_btn.click(hide_chips, [], [chip_row]).then(
                fn=self._respond,
                inputs=submit_inputs,
                outputs=submit_outputs
            )

            # Utils
            btn_new.click(
                fn=lambda: ([], "New Conversation", "<div class='badge'>Ready</div>", build_right_panel_html(None), gr.update(visible=True)),
                outputs=[chatbot, chat_title, chat_badge, right_html, chip_row]
            )

            down_btn.click(fn=self._export_json, outputs=[file_out]).then(
                fn=lambda: gr.update(visible=True), outputs=[file_out]
            )
            
            # Simple JS Copy Context mechanism
            copy_btn.click(
                None, [], [], js="""
                () => {
                    let texts = [];
                    document.querySelectorAll('.chunk-card .preview-text').forEach(e => texts.push(e.innerText));
                    navigator.clipboard.writeText(texts.join('\\n\\n'));
                    alert("Context copied to clipboard!");
                }
                """
            )

            demo.load(None, None, None)

        return demo

    def launch(self, **kwargs) -> None:
        self._demo.launch(server_name="0.0.0.0", server_port=config.APP_PORT, **kwargs)

if __name__ == "__main__":
    app = RAGApp()
    app.launch()
