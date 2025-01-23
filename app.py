import os
import datetime
import json
import gradio as gr
from openai import OpenAI
import torch
import torch.nn.functional as F
from transformers import AutoModel
from nltk.tokenize import word_tokenize
import chromadb

# Set your batch size here
BATCH_SIZE = 64

ENDPOINT_URL = "http://0.0.0.0:8999/v1"

todays_date_string = datetime.date.today().strftime("%d %B %Y")

RECREATE_DB = not os.path.exists("db/")
VERBOSE_SHELL = True

SYSTEM_PROMPT = """Cutting Knowledge Date: December 2023
Today Date: """ + todays_date_string + """

You are a helpful assistant with tool calling capabilities.

You specialize in Enron email documentation retrieval and consolidation. You have access to a documentation search function to help answer queries accurately.
The Enron Email Dataset is a collection of emails exchanged by Enron employees, made public during investigations into the company's collapse. The dataset provides insight into corporate communication, organizational structure, and decision-making processes during a pivotal moment in business history. It has become a valuable resource for researchers in fields like data science, sociology, and linguistics.

Key features of the dataset include:
- Over 600,000 emails from approximately 158 employees, primarily senior management.
- Messages detailing internal communication, policy discussions, and personal exchanges.
- A structure that preserves metadata such as timestamps, recipients, and subject lines.

The dataset addresses challenges in understanding corporate dynamics and offers a unique opportunity to study real-world organizational behavior. Researchers have used it to analyze social networks, detect anomalies, and develop natural language processing tools.

If you choose to use one of the following functions, respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. Do not use variables.

{functions}

After receiving the results back from a function (formatted as {{"name": function name, "return": returned data after running function}}) formulate your response to the user. Do not mention that you are using retrieved/returned data to answer the question. If the information needed is not found in the returned data, either attempt a new function call, or inform the user that you cannot answer based on your available knowledge. The user cannot see the function results. They are only aware that you have access to a documentation search tool. So don't say "Based on the search results..." or anything similar. Just answer the question directly."""

FUNCTIONS_LIST = [
    {
        "type": "function",
        "function": {
            "name": "enron_email_search",
            "description": "Search for a query string in the Enron email documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query string to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum quantity of results to return",
                        "default": 3,
                    }
                },
                "required": ["query"]
            }
        }
    },
]


###############################
# ChromaDB Indexing Functions #
###############################
import os
import chromadb
from nltk.tokenize import word_tokenize
from openai import OpenAI
from transformers import AutoTokenizer
import os
import chromadb
from datasets import load_dataset

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8998/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

local_tokenizer = None

def depr_chunk_text_by_tokens(text: str, max_tokens: int = 512,overlap=0.2,min_tokens=50):
    """
    Split a text into chunks of a maximum number of tokens. 
    """
    tokens = word_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_len = 0
    for token in tokens:
        current_chunk.append(token)
        current_chunk_len += 1
        if current_chunk_len >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_chunk_len = 0
        elif current_chunk_len >= min_tokens and token in [".", "?", "!"]:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_chunk_len = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

from transformers import AutoTokenizer
import semchunk

local_embed_tokenizer = None

def chunk_text_by_tokens(
    text: str,
    max_tokens: int = 512,
    min_overlap: float = 0.2,
    model_name: str = "intfloat/multilingual-e5-large",
):
    global local_embed_tokenizer
    if local_embed_tokenizer is None:
        local_embed_tokenizer = AutoTokenizer.from_pretrained(model_name,device="cuda")

    # Determine the chunk size and overlap
    chunk_size = max_tokens  # semchunk uses a fixed chunk size
    overlap = min_overlap  # semchunk uses a fixed overlap

    # Create the chunker
    chunker = semchunk.chunkerify(local_embed_tokenizer, chunk_size)

    # Split the text into chunks
    chunks = chunker(text, overlap=overlap)

    return chunks


def create_search_index(db_directory: str = "db/",
                        collection_name: str = "text_chunks",
                        max_tokens_per_chunk: int = 512):
    """
    Creates a searchable index from the Enron email dataset by token-based chunking
    and storing embeddings in ChromaDB.

    Args:
        db_directory: Where to store the ChromaDB embeddings.
        collection_name: Name for the ChromaDB collection.
        max_tokens_per_chunk: Maximum tokens per chunk to avoid exceeding model limits.
    """
    # 1. Load the Enron dataset
    print("Loading the Enron dataset...")
    ds = load_dataset("snoop2head/enron_aeslc_emails", split="train")

    # 2. Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=db_directory)
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        print(f"Creating collection '{collection_name}' in ChromaDB...")
        collection = chroma_client.create_collection(name=collection_name)

    # 3. Process each email and store in chunks
    total_emails = len(ds)
    print(f"Processing {total_emails} Enron emails...")

    batch_data = []

    def process_batch(batch_data):
        """
        Helper function to embed a batch and store in ChromaDB.
        """
        texts = [item["text"] for item in batch_data]
        snippet_ids = [item["snippet_id"] for item in batch_data]
        metadata_list = [item["metadata"] for item in batch_data]

        # Use the vLLM / OpenAI client to create embeddings
        responses = client.embeddings.create(
            input=texts,
            model=client.models.list().data[0].id  # or hardcode your chosen model
        )

        # Each entry in responses.data corresponds to the embedding of the input with the same index
        embeddings = [d.embedding for d in responses.data]

        # Add to ChromaDB
        for i, snippet_id in enumerate(snippet_ids):
            collection.add(
                ids=[snippet_id],
                embeddings=[embeddings[i]],
                documents=[texts[i]],
                metadatas=[metadata_list[i]]
            )
    email_idx_start = 0
    if os.path.exists("db/.progress"):
        with open("db/.progress", "r") as f:
            email_idx_start = int(f.read().strip())
    for email_idx, row in enumerate(ds):
        if email_idx < email_idx_start:
            continue
        text = row["text"]
        if not text:
            continue

        # Split into token-based chunks (each <= max_tokens_per_chunk)
        chunks = chunk_text_by_tokens(text, max_tokens=max_tokens_per_chunk)

        # Prepare chunks for embedding
        for c_idx, chunk in enumerate(chunks):
            snippet_id = f"email_{email_idx}_chunk_{c_idx}"

            # Skip if already indexed
            found = collection.get(ids=[snippet_id])
            if len(found["ids"]) > 0:
                continue

            batch_data.append({
                "text": chunk,
                "snippet_id": snippet_id,
                "metadata": {
                    "email_idx": email_idx,
                    "chunk_index": c_idx
                }
            })

            # If the batch is full, process it
            if len(batch_data) >= BATCH_SIZE:
                process_batch(batch_data)
                batch_data = []

        if (email_idx + 1) % 1000 == 0:
            print(f"Processed {email_idx + 1}/{total_emails} emails...")
            with open("db/.progress", "w") as f:
                f.write(str(email_idx ))

    # Process any remaining items in the batch
    if batch_data:
        process_batch(batch_data)

    print("Done embedding Enron emails into ChromaDB.")

# -------------------------
# 3) Simple search function
# -------------------------
def search_text(query: str,
                db_directory: str = "db/",
                collection_name: str = "text_chunks",
                top_n: int = 5) -> list:
    """
    Search the indexed Enron dataset using a query string.

    Args:
        query:        Search query string
        db_directory: ChromaDB directory
        collection_name: Name of the collection to search
        top_n:        Number of results to return

    Returns:
        List of relevant text chunks
    """
    # Load DB
    chroma_client = chromadb.PersistentClient(path=db_directory)
    collection = chroma_client.get_collection(name=collection_name)
    
    # Get the embedding model
    embed_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8998/v1")
    embed_models = embed_client.models.list()
    embed_model = embed_models.data[0].id
    
    # Create query embedding
    response = embed_client.embeddings.create(
        input=[query],
        model=embed_model
    )
    query_vector = response.data[0].embedding
    
    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_n
    )
    
    if "documents" in results and len(results["documents"]) > 0:
        return results["documents"][0]
    return []

# (Optional) Initialize the DB if needed
if not os.path.exists("db/.completed"):
    create_search_index()
    with open("db/.completed", "w") as f:
        f.write("")


FUNCTIONS_DICT = {f["function"]["name"]: f for f in FUNCTIONS_LIST}
FUNCTION_BACKENDS = {
    "enron_email_search": search_text,
}

EOT_STRING = "<|eot_id|>"
FUNCTION_EOT_STRING = "<|eom_id|>"
ROLE_HEADER = "<|start_header_id|>{role}<|end_header_id|>"
INIT_STRING = "<|begin_of_text|>"

class LLM:
    def __init__(self, max_model_len: int = 4096):
        self.api_key = "dummykey"
        self.max_model_len = max_model_len
        self.client = OpenAI(base_url=ENDPOINT_URL, api_key=self.api_key)
        models_list = self.client.models.list()
        self.model_name = models_list.data[0].id

    def generate(self, prompt: str, sampling_params: dict) -> dict:
        completion_params = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": sampling_params.get("max_tokens", 2048),
            "temperature": sampling_params.get("temperature", 0.8),
            "top_p": sampling_params.get("top_p", 0.95),
            "n": sampling_params.get("n", 1),
            "stream": False,
        }
        
        if "stop" in sampling_params:
            completion_params["stop"] = sampling_params["stop"]
        if "presence_penalty" in sampling_params:
            completion_params["presence_penalty"] = sampling_params["presence_penalty"]
        if "frequency_penalty" in sampling_params:
            completion_params["frequency_penalty"] = sampling_params["frequency_penalty"]
            
        return self.client.completions.create(**completion_params)


def form_chat_prompt(message_history, functions=FUNCTIONS_DICT.keys()):
    """Builds the custom text prompt for your vLLM."""
    functions_string = "\n\n".join([json.dumps(FUNCTIONS_DICT[f], indent=4) for f in functions])
    full_prompt = (
        INIT_STRING
        + ROLE_HEADER.format(role="system")
        + "\n\n"
        + SYSTEM_PROMPT.format(functions=functions_string)
        + EOT_STRING
    )
    for message in message_history:
        full_prompt += (
            ROLE_HEADER.format(role=message["role"])
            + "\n\n"
            + message["content"]
            + EOT_STRING
        )
    full_prompt += ROLE_HEADER.format(role="assistant")
    return full_prompt


def check_assistant_response_for_tool_calls(response):
    """If the LLM output is JSON with function call data, parse it."""
    response = response.replace(EOT_STRING, "").replace(FUNCTION_EOT_STRING, "").strip()
    for tool_name in FUNCTIONS_DICT.keys():
        if f"\"{tool_name}\"" in response and response.startswith("{"):
            # Try to find the last '}' character of this json and remove the rest of the string
            for _ in range(10):
                response = response.rsplit('}', 1)[0] + '}'
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    print("Failed to parse JSON, trying again")
                    continue
    return None

def process_tool_request(tool_request_data):
    """Call the relevant backend tool with the parameters from the LLM."""
    tool_name = tool_request_data["name"]
    tool_parameters = tool_request_data["parameters"]
    print("Tool name:", tool_name)
    print("Tool parameters:", tool_parameters)

    if tool_name == "enron_email_search":
        query = tool_parameters["query"]
        top_n = min(7, max(int(tool_parameters.get("max_results", 3)), 2))
        search_results = FUNCTION_BACKENDS[tool_name](query, top_n=top_n)
        print("Search results:", search_results)
        return {"name": "enron_email_search", "results": search_results}

    return None

def restore_message_history(full_history):
    """Restore tool interactions into the message history."""
    restored = []
    for message in full_history:
        if message["role"] == "assistant" and "metadata" in message:
            tool_interactions = message["metadata"].get("tool_interactions", [])
            if tool_interactions:
                for tool_msg in tool_interactions:
                    restored.append(tool_msg)
                final_msg = message.copy()
                del final_msg["metadata"]["tool_interactions"]
                restored.append(final_msg)
            else:
                restored.append(message)
        else:
            restored.append(message)
    return restored

def iterate_chat(llm, sampling_params, full_history):
    """Handle multiple turns of conversation with tool calling."""
    tool_interactions = []

    for _ in range(10):
        prompt = form_chat_prompt(restore_message_history(full_history) + tool_interactions)
        if VERBOSE_SHELL:
            print("Prompt: " + prompt + "\n-----------------------------------")
        output = llm.generate(prompt, sampling_params)
        
        if not output or not output.choices:
            raise ValueError("Invalid completion response")
            
        assistant_response = output.choices[0].text.strip()
        if VERBOSE_SHELL:
            print("Assistant response: " + assistant_response + "\n===================================")
        assistant_response = assistant_response.replace(EOT_STRING, "").replace(FUNCTION_EOT_STRING, "")

        tool_request_data = check_assistant_response_for_tool_calls(assistant_response)
        if not tool_request_data:
            final_message = {
                "role": "assistant",
                "content": assistant_response,
                "metadata": {
                    "tool_interactions": tool_interactions
                }
            }
            full_history.append(final_message)
            return full_history
        else:
            assistant_message = {
                "role": "assistant",
                "content": json.dumps(tool_request_data),
            }
            tool_interactions.append(assistant_message)
            tool_return_data = process_tool_request(tool_request_data)

            ipython_message = {
                "role": "ipython",
                "content": json.dumps(tool_return_data)
            }
            tool_interactions.append(ipython_message)

    return full_history

def user_conversation(user_message, chat_history, full_history):
    """Handle user input and maintain conversation state."""
    if full_history is None:
        full_history = []

    full_history.append({"role": "user", "content": user_message})
    updated_history = iterate_chat(llm, sampling_params, full_history)
    assistant_answer = updated_history[-1]["content"]
    chat_history.append((user_message, assistant_answer))

    return "", chat_history, updated_history

# Updated sampling parameters
sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 2048,
}

# Initialize LLM
llm = LLM(
    max_model_len=8096,
)

def build_demo():
    """Build the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("<h2>Historical Enron Emails Dataset Exploration RAG ChatBot</h2>")
        chat_state = gr.State([])
        chatbot = gr.Chatbot(label="Chat with your LLM")
        user_input = gr.Textbox(
            lines=1, 
            placeholder="Ask something about the Enron email dataset...",
        )

        user_input.submit(
            fn=user_conversation,
            inputs=[user_input, chatbot, chat_state],
            outputs=[user_input, chatbot, chat_state],
            queue=False
        )

        send_button = gr.Button("Send")
        send_button.click(
            fn=user_conversation,
            inputs=[user_input, chatbot, chat_state],
            outputs=[user_input, chatbot, chat_state],
            queue=False
        )

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True, server_port=7860)
