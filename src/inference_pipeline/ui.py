import sys
from pathlib import Path

# To mimic using multiple Python modules, such as 'core' and 'feature_pipeline',
# we will add the './src' directory to the PYTHONPATH. This is not intended for
# production use cases but for development and educational purposes.
ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from core.config import settings
from praxis_llm import PraxisLLM

settings.patch_localhost()


import gradio as gr
from inference_pipeline.praxis_llm import PraxisLLM

llm_praxis = PraxisLLM(mock=False)


def predict(message: str, history: list[list[str]], author: str) -> str:
    """
    Generates a response using the LLM Praxis, simulating a conversation with your digital praxis.

    Args:
        message (str): The user's input message or question.
        history (List[List[str]]): Previous conversation history between user and praxis.
        about_me (str): Personal context about the user to help personalize responses.

    Returns:
        str: The LLM Praxis's generated response.
    """

    query = f"I am {author}. Write about: {message}"
    response = llm_praxis.generate(
        query=query, enable_rag=True, sample_for_evaluation=False
    )

    return response["answer"]


demo = gr.ChatInterface(
    predict,
    textbox=gr.Textbox(
        placeholder="Chat with your LLM Praxis",
        label="Message",
        container=False,
        scale=7,
    ),
    additional_inputs=[
        gr.Textbox(
            "Dev Jadhav",
            label="Who are you?",
        )
    ],
    title="Your LLM Praxis",
    description="""
    Chat with your personalized LLM Praxis! This AI assistant will help you write content incorporating your style and voice.
    """,
    theme="soft",
    examples=[
        [
            "Draft a post about RAG systems.",
            "Dev Jadhav",
        ],
        [
            "Draft an article paragraph about vector databases.",
            "Dev Jadhav",
        ],
        [
            "Draft a post about LLM chatbots.",
            "Dev Jadhav",
        ],
    ],
    cache_examples=False,
    chatbot=gr.Chatbot(type="messages")
)


if __name__ == "__main__":
    # Get port from environment or use default 8050
    import os
    port = int(os.environ.get("PORT", 8050))
    
    # Determine if we're in a Docker container
    in_docker = os.environ.get("DOCKER_CONTAINER", "false").lower() == "true"
    
    # Launch with appropriate parameters
    # Don't share when in Docker, it's not needed and can cause issues
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=False)
