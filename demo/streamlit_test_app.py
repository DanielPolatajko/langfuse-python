import streamlit as st
import openai
from langfuse import observe
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from cot_monitor.monitor import CotMonitor

client = Anthropic()

load_dotenv()

from langfuse import get_client

langfuse = get_client()

st.set_page_config(page_title="Langfuse Test Chat", page_icon="💬")

cot_monitor = CotMonitor()


def call_gpt_with_monitoring(messages):
    """Call OpenAI ChatGPT API with Langfuse instrumentation."""
    with langfuse.start_as_current_span(name="call_gpt_with_monitoring") as span:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            thinking={"type": "enabled", "budget_tokens": 2000},
            messages=messages,
        )

        output = response.content[1].text
        cot = response.content[0].thinking

        action_score = cot_monitor.monitor_action(output)
        cot_score = cot_monitor.monitor_cot(cot, output)
        hybrid_score = cot_monitor.monitor_hybrid(action_score, cot_score)

        with langfuse.start_as_current_span(name="cot-monitoring") as span:
            # Score the current span
            span.score(
                name="action-score",
                value=action_score,
                data_type="NUMERIC",
                comment="Action score",
            )
            span.score(
                name="cot-score",
                value=cot_score,
                data_type="NUMERIC",
                comment="Cot score",
            )
            span.score(
                name="hybrid-score",
                value=hybrid_score,
                data_type="NUMERIC",
                comment="Hybrid score",
            )

            # Score the trace
            span.score_trace(
                name="overall-score",
                value=hybrid_score,
                data_type="NUMERIC",
                comment="Overall score",
            )

        return response.choices[0].message.content


def main():
    st.title("💬 Langfuse Test Chat")
    st.caption("A minimal ChatGPT wrapper with Langfuse instrumentation")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to chat about?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call ChatGPT with Langfuse instrumentation
                    response = call_gpt_with_monitoring(st.session_state.messages)
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Sidebar with info
    with st.sidebar:
        st.header("Configuration")
        st.info(
            "Make sure to set your environment variables:\n\n"
            "- OPENAI_API_KEY\n"
            "- LANGFUSE_PUBLIC_KEY\n"
            "- LANGFUSE_SECRET_KEY\n"
            "- LANGFUSE_HOST (optional)"
        )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
