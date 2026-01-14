"""
Streamlit dashboard for LLM Guardrail Studio
"""

import streamlit as st
from guardrails import GuardrailPipeline

st.set_page_config(
    page_title="LLM Guardrail Studio",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ LLM Guardrail Studio")
st.markdown("**Modular Trust Layer for Local LLMs**")

# Sidebar configuration
st.sidebar.header("Configuration")

model_name = st.sidebar.selectbox(
    "Model",
    ["mistralai/Mistral-7B-v0.1", "HuggingFaceH4/zephyr-7b-beta", "Custom"]
)

enable_toxicity = st.sidebar.checkbox("Toxicity Detection", value=True)
enable_hallucination = st.sidebar.checkbox("Hallucination Detection", value=True)
enable_alignment = st.sidebar.checkbox("Alignment Check", value=True)

toxicity_threshold = st.sidebar.slider("Toxicity Threshold", 0.0, 1.0, 0.7)
alignment_threshold = st.sidebar.slider("Alignment Threshold", 0.0, 1.0, 0.5)

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    return GuardrailPipeline(
        model_name=model_name,
        enable_toxicity=enable_toxicity,
        enable_hallucination=enable_hallucination,
        enable_alignment=enable_alignment,
        toxicity_threshold=toxicity_threshold,
        alignment_threshold=alignment_threshold
    )

pipeline = load_pipeline()

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    prompt = st.text_area("Prompt", height=150, placeholder="Enter your prompt here...")
    response = st.text_area("Response", height=150, placeholder="Enter model response here...")
    
    evaluate_btn = st.button("🔍 Evaluate", type="primary")

with col2:
    st.subheader("Results")
    
    if evaluate_btn and prompt and response:
        with st.spinner("Evaluating..."):
            result = pipeline.evaluate(prompt, response)
            
            # Display overall status
            if result.passed:
                st.success("✅ All checks passed")
            else:
                st.error("⚠️ Issues detected")
            
            # Display scores
            st.markdown("### Scores")
            for metric, score in result.scores.items():
                st.metric(metric.replace('_', ' ').title(), f"{score:.3f}")
            
            # Display flags
            if result.flags:
                st.markdown("### Flags")
                for flag in result.flags:
                    st.warning(flag)
    elif evaluate_btn:
        st.info("Please enter both prompt and response")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Transformers, SentenceTransformers, and Streamlit")
