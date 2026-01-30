"""
Enhanced Streamlit dashboard for LLM Guardrail Studio
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json

from guardrails import GuardrailPipeline
from dashboard.components import MetricsDisplay, FlagDisplay, ModelSelector, HistoryDisplay
from dashboard.utils import display_metric_explanation, create_download_report

st.set_page_config(
    page_title="LLM Guardrail Studio",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []

st.title("🛡️ LLM Guardrail Studio")
st.markdown("**Modular Trust Layer for Local LLMs**")

# Sidebar configuration
model_config = ModelSelector.render_model_config()

st.sidebar.header("🔧 Guardrail Settings")
enable_toxicity = st.sidebar.checkbox("Toxicity Detection", value=True)
enable_hallucination = st.sidebar.checkbox("Hallucination Detection", value=True)
enable_alignment = st.sidebar.checkbox("Alignment Check", value=True)

toxicity_threshold = st.sidebar.slider("Toxicity Threshold", 0.0, 1.0, 0.7, 0.1)
alignment_threshold = st.sidebar.slider("Alignment Threshold", 0.0, 1.0, 0.5, 0.1)

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    hallucination_threshold = st.slider("Hallucination Threshold", 0.0, 1.0, 0.6, 0.1)
    batch_mode = st.checkbox("Batch Evaluation Mode")

# Initialize pipeline
@st.cache_resource
def load_pipeline(_model_config, _enable_toxicity, _enable_hallucination, _enable_alignment, _toxicity_threshold, _alignment_threshold):
    return GuardrailPipeline(
        model_name=_model_config["model_name"],
        enable_toxicity=_enable_toxicity,
        enable_hallucination=_enable_hallucination,
        enable_alignment=_enable_alignment,
        toxicity_threshold=_toxicity_threshold,
        alignment_threshold=_alignment_threshold
    )

pipeline = load_pipeline(
    model_config, enable_toxicity, enable_hallucination, 
    enable_alignment, toxicity_threshold, alignment_threshold
)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Evaluate", "📊 Analytics", "📋 History", "ℹ️ Help"])

with tab1:
    # Main evaluation interface
    if not batch_mode:
        # Single evaluation mode
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input")
            prompt = st.text_area("Prompt", height=150, placeholder="Enter your prompt here...")
            response = st.text_area("Response", height=150, placeholder="Enter model response here...")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                evaluate_btn = st.button("🔍 Evaluate", type="primary")
            with col_btn2:
                clear_btn = st.button("🗑️ Clear")
            
            if clear_btn:
                st.rerun()
        
        with col2:
            st.subheader("Results")
            
            if evaluate_btn and prompt and response:
                with st.spinner("Evaluating..."):
                    result = pipeline.evaluate(prompt, response)
                    
                    # Store in history
                    evaluation_data = {
                        "timestamp": datetime.now(),
                        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                        "response": response[:100] + "..." if len(response) > 100 else response,
                        "passed": result.passed,
                        "scores": result.scores,
                        "flags": result.flags
                    }
                    st.session_state.evaluation_history.append(evaluation_data)
                    
                    # Display results
                    FlagDisplay.render_flags(result.flags)
                    
                    if result.scores:
                        st.markdown("### 📊 Scores")
                        MetricsDisplay.render_score_cards(result.scores)
                        
                        # Radar chart
                        MetricsDisplay.render_radar_chart(result.scores)
                        
                        # Download report
                        report = create_download_report({
                            "prompt": prompt,
                            "response": response,
                            "passed": result.passed,
                            "scores": result.scores,
                            "flags": result.flags
                        })
                        
                        st.download_button(
                            label="📄 Download Report",
                            data=report,
                            file_name=f"guardrail_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                        
            elif evaluate_btn:
                st.info("Please enter both prompt and response")
    
    else:
        # Batch evaluation mode
        st.subheader("📦 Batch Evaluation")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with 'prompt' and 'response' columns",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'prompt' not in df.columns or 'response' not in df.columns:
                    st.error("CSV must contain 'prompt' and 'response' columns")
                else:
                    st.success(f"Loaded {len(df)} rows")
                    st.dataframe(df.head())
                    
                    if st.button("🚀 Run Batch Evaluation"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, row in df.iterrows():
                            result = pipeline.evaluate(row['prompt'], row['response'])
                            results.append({
                                'id': i,
                                'passed': result.passed,
                                'toxicity': result.scores.get('toxicity', 0),
                                'alignment': result.scores.get('alignment', 0),
                                'hallucination_risk': result.scores.get('hallucination_risk', 0),
                                'flags_count': len(result.flags)
                            })
                            progress_bar.progress((i + 1) / len(df))
                        
                        # Display batch results
                        results_df = pd.DataFrame(results)
                        st.subheader("Batch Results")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", len(results_df))
                        with col2:
                            passed = results_df['passed'].sum()
                            st.metric("Passed", passed)
                        with col3:
                            failed = len(results_df) - passed
                            st.metric("Failed", failed)
                        
                        st.dataframe(results_df)
                        
                        # Download batch results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📊 Download Results CSV",
                            csv,
                            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")

with tab2:
    st.subheader("📊 Analytics Dashboard")
    
    if st.session_state.evaluation_history:
        # Summary metrics
        history_df = pd.DataFrame(st.session_state.evaluation_history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Evaluations", len(history_df))
        with col2:
            passed_rate = history_df['passed'].mean() * 100
            st.metric("Pass Rate", f"{passed_rate:.1f}%")
        with col3:
            avg_toxicity = history_df['scores'].apply(lambda x: x.get('toxicity', 0)).mean()
            st.metric("Avg Toxicity", f"{avg_toxicity:.3f}")
        with col4:
            avg_alignment = history_df['scores'].apply(lambda x: x.get('alignment', 0)).mean()
            st.metric("Avg Alignment", f"{avg_alignment:.3f}")
        
        # Trend charts
        if len(history_df) > 1:
            HistoryDisplay.render_trend_chart(st.session_state.evaluation_history)
        
        # Distribution charts
        st.subheader("Score Distributions")
        
        toxicity_scores = [scores.get('toxicity', 0) for scores in history_df['scores']]
        alignment_scores = [scores.get('alignment', 0) for scores in history_df['scores']]
        
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(data=[go.Histogram(x=toxicity_scores, name="Toxicity")])
            fig.update_layout(title="Toxicity Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[go.Histogram(x=alignment_scores, name="Alignment")])
            fig.update_layout(title="Alignment Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No evaluation data available. Run some evaluations first!")

with tab3:
    st.subheader("📋 Evaluation History")
    
    if st.session_state.evaluation_history:
        # Controls
        col1, col2 = st.columns([3, 1])
        with col1:
            show_details = st.checkbox("Show detailed view")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.evaluation_history = []
                st.rerun()
        
        # Display history
        if show_details:
            for i, entry in enumerate(reversed(st.session_state.evaluation_history)):
                with st.expander(f"Evaluation {len(st.session_state.evaluation_history) - i} - {'✅' if entry['passed'] else '❌'}"):
                    st.write(f"**Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Prompt:** {entry['prompt']}")
                    st.write(f"**Response:** {entry['response']}")
                    st.write(f"**Scores:** {entry['scores']}")
                    if entry['flags']:
                        st.write(f"**Flags:** {', '.join(entry['flags'])}")
        else:
            HistoryDisplay.render_history_table(st.session_state.evaluation_history)
        
        # Export history
        if st.button("📤 Export History"):
            history_json = json.dumps(st.session_state.evaluation_history, default=str, indent=2)
            st.download_button(
                "Download History JSON",
                history_json,
                f"evaluation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    else:
        st.info("No evaluation history available.")

with tab4:
    st.subheader("ℹ️ Help & Documentation")
    
    display_metric_explanation()
    
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        1. **Configure Settings**: Use the sidebar to select your model and adjust thresholds
        2. **Enter Content**: Input a prompt and response in the Evaluate tab
        3. **Run Evaluation**: Click the Evaluate button to analyze the content
        4. **Review Results**: Check scores and flags in the results panel
        5. **View Analytics**: Use the Analytics tab to see trends and distributions
        """)
    
    with st.expander("🔧 API Usage"):
        st.code("""
from guardrails import GuardrailPipeline

# Initialize pipeline
pipeline = GuardrailPipeline(
    enable_toxicity=True,
    enable_hallucination=True,
    enable_alignment=True
)

# Evaluate content
result = pipeline.evaluate(
    prompt="Your prompt here",
    response="Model response here"
)

print(f"Passed: {result.passed}")
print(f"Scores: {result.scores}")
print(f"Flags: {result.flags}")
        """, language="python")
    
    with st.expander("📊 Batch Processing"):
        st.markdown("""
        For batch processing:
        1. Enable "Batch Evaluation Mode" in the sidebar
        2. Upload a CSV file with 'prompt' and 'response' columns
        3. Click "Run Batch Evaluation"
        4. Download results as CSV
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<div style='text-align: center'>Built with ❤️ using Transformers, SentenceTransformers, and Streamlit</div>", 
        unsafe_allow_html=True
    )
