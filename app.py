"""
Enhanced Streamlit dashboard for LLM Guardrail Studio
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import requests
from typing import Optional

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
if 'api_mode' not in st.session_state:
    st.session_state.api_mode = False
if 'custom_rules' not in st.session_state:
    st.session_state.custom_rules = []

st.title("🛡️ LLM Guardrail Studio")
st.markdown("**Production-Ready Safety Pipeline for Local LLMs**")

# Sidebar configuration
with st.sidebar:
    st.header("🔌 Connection Mode")
    connection_mode = st.radio(
        "Select mode:",
        ["Local Pipeline", "API Server"],
        help="Use Local Pipeline for direct evaluation or API Server to connect to a running instance"
    )
    
    if connection_mode == "API Server":
        st.session_state.api_mode = True
        api_url = st.text_input("API URL", value="http://localhost:8000")
        api_key = st.text_input("API Key (optional)", type="password")
    else:
        st.session_state.api_mode = False

# Model configuration (only for local mode)
if not st.session_state.api_mode:
    model_config = ModelSelector.render_model_config()

st.sidebar.header("🔧 Guardrail Settings")

# Core evaluators
st.sidebar.subheader("Core Evaluators")
enable_toxicity = st.sidebar.checkbox("Toxicity Detection", value=True, help="Detect toxic, harmful content")
enable_hallucination = st.sidebar.checkbox("Hallucination Detection", value=True, help="Detect fabricated information")
enable_alignment = st.sidebar.checkbox("Alignment Check", value=True, help="Check response alignment with prompt")

# Security evaluators
st.sidebar.subheader("Security Evaluators")
enable_pii = st.sidebar.checkbox("PII Detection", value=True, help="Detect personal identifiable information")
enable_injection = st.sidebar.checkbox("Prompt Injection Detection", value=True, help="Detect prompt injection attacks")
enable_custom_rules = st.sidebar.checkbox("Custom Rules", value=False, help="Apply custom content rules")

# Thresholds
with st.sidebar.expander("⚙️ Thresholds"):
    toxicity_threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.7, 0.05)
    alignment_threshold = st.slider("Alignment Threshold", 0.0, 1.0, 0.5, 0.05)
    hallucination_threshold = st.slider("Hallucination Threshold", 0.0, 1.0, 0.6, 0.05)
    pii_threshold = st.slider("PII Risk Threshold", 0.0, 1.0, 0.1, 0.05)
    injection_threshold = st.slider("Injection Risk Threshold", 0.0, 1.0, 0.5, 0.05)

# Advanced settings
with st.sidebar.expander("🚀 Advanced Settings"):
    batch_mode = st.checkbox("Batch Evaluation Mode")
    show_detailed_results = st.checkbox("Show Detailed Results", value=True)
    enable_caching = st.checkbox("Enable Caching", value=True)

# Initialize pipeline (local mode only)
@st.cache_resource
def load_pipeline(
    _model_name: str,
    _enable_toxicity: bool,
    _enable_hallucination: bool, 
    _enable_alignment: bool,
    _enable_pii: bool,
    _enable_injection: bool,
    _enable_custom_rules: bool,
    _toxicity_threshold: float,
    _alignment_threshold: float,
    _hallucination_threshold: float,
    _pii_threshold: float,
    _injection_threshold: float,
    _enable_caching: bool
):
    return GuardrailPipeline(
        model_name=_model_name,
        enable_toxicity=_enable_toxicity,
        enable_hallucination=_enable_hallucination,
        enable_alignment=_enable_alignment,
        enable_pii=_enable_pii,
        enable_injection=_enable_injection,
        enable_custom_rules=_enable_custom_rules,
        toxicity_threshold=_toxicity_threshold,
        alignment_threshold=_alignment_threshold,
        hallucination_threshold=_hallucination_threshold,
        pii_threshold=_pii_threshold,
        injection_threshold=_injection_threshold,
        enable_cache=_enable_caching
    )

def evaluate_via_api(prompt: str, response: str, api_url: str, api_key: Optional[str] = None):
    """Evaluate content via API server"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        resp = requests.post(
            f"{api_url}/evaluate",
            json={"prompt": prompt, "response": response},
            headers=headers,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# Load pipeline if in local mode
pipeline = None
if not st.session_state.api_mode:
    pipeline = load_pipeline(
        model_config["model_name"],
        enable_toxicity,
        enable_hallucination, 
        enable_alignment,
        enable_pii,
        enable_injection,
        enable_custom_rules,
        toxicity_threshold,
        alignment_threshold,
        hallucination_threshold,
        pii_threshold,
        injection_threshold,
        enable_caching
    )

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Evaluate", "📊 Analytics", "📋 History", "⚙️ Rules", "ℹ️ Help"])

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
                evaluate_btn = st.button("🔍 Evaluate", type="primary", use_container_width=True)
            with col_btn2:
                clear_btn = st.button("🗑️ Clear", use_container_width=True)
            
            if clear_btn:
                st.rerun()
        
        with col2:
            st.subheader("Results")
            
            if evaluate_btn and prompt and response:
                with st.spinner("Evaluating..."):
                    # Evaluate using API or local pipeline
                    if st.session_state.api_mode:
                        api_result = evaluate_via_api(prompt, response, api_url, api_key if api_key else None)
                        if api_result:
                            result_passed = api_result.get("passed", False)
                            result_scores = api_result.get("scores", {})
                            result_flags = api_result.get("flags", [])
                            result_detailed = api_result.get("detailed_results", {})
                        else:
                            st.error("Failed to get evaluation results from API")
                            st.stop()
                    else:
                        result = pipeline.evaluate(prompt, response)
                        result_passed = result.passed
                        result_scores = result.scores
                        result_flags = result.flags
                        result_detailed = result.detailed_results
                    
                    # Store in history
                    evaluation_data = {
                        "timestamp": datetime.now(),
                        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                        "response": response[:100] + "..." if len(response) > 100 else response,
                        "passed": result_passed,
                        "scores": result_scores,
                        "flags": result_flags
                    }
                    st.session_state.evaluation_history.append(evaluation_data)
                    
                    # Overall status
                    if result_passed:
                        st.success("✅ Evaluation PASSED - Content appears safe")
                    else:
                        st.error("❌ Evaluation FAILED - Issues detected")
                    
                    # Display flags
                    FlagDisplay.render_flags(result_flags)
                    
                    if result_scores:
                        st.markdown("### 📊 Scores")
                        MetricsDisplay.render_score_cards(result_scores)
                        
                        # Radar chart
                        if len(result_scores) >= 3:
                            MetricsDisplay.render_radar_chart(result_scores)
                        
                        # Detailed results
                        if show_detailed_results and result_detailed:
                            st.markdown("### 🔬 Detailed Analysis")
                            MetricsDisplay.render_detailed_results(result_detailed)
                        
                        # Download report
                        report = create_download_report({
                            "prompt": prompt,
                            "response": response,
                            "passed": result_passed,
                            "scores": result_scores,
                            "flags": result_flags,
                            "detailed_results": result_detailed if show_detailed_results else {}
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
        
        st.info("Upload a CSV file with 'prompt' and 'response' columns to evaluate multiple items at once.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV must contain 'prompt' and 'response' columns"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'prompt' not in df.columns or 'response' not in df.columns:
                    st.error("CSV must contain 'prompt' and 'response' columns")
                else:
                    st.success(f"✅ Loaded {len(df)} rows")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("🚀 Run Batch Evaluation", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results = []
                        
                        for i, row in df.iterrows():
                            status_text.text(f"Processing {i+1}/{len(df)}...")
                            
                            if st.session_state.api_mode:
                                api_result = evaluate_via_api(row['prompt'], row['response'], api_url, api_key if api_key else None)
                                if api_result:
                                    results.append({
                                        'id': i,
                                        'passed': api_result.get('passed', False),
                                        'toxicity': api_result.get('scores', {}).get('toxicity', 0),
                                        'alignment': api_result.get('scores', {}).get('alignment', 0),
                                        'hallucination_risk': api_result.get('scores', {}).get('hallucination_risk', 0),
                                        'pii_risk': api_result.get('scores', {}).get('pii_risk', 0),
                                        'injection_risk': api_result.get('scores', {}).get('injection_risk', 0),
                                        'flags_count': len(api_result.get('flags', []))
                                    })
                            else:
                                result = pipeline.evaluate(row['prompt'], row['response'])
                                results.append({
                                    'id': i,
                                    'passed': result.passed,
                                    'toxicity': result.scores.get('toxicity', 0),
                                    'alignment': result.scores.get('alignment', 0),
                                    'hallucination_risk': result.scores.get('hallucination_risk', 0),
                                    'pii_risk': result.scores.get('pii_risk', 0),
                                    'injection_risk': result.scores.get('injection_risk', 0),
                                    'flags_count': len(result.flags)
                                })
                            
                            progress_bar.progress((i + 1) / len(df))
                        
                        status_text.empty()
                        
                        # Display batch results
                        results_df = pd.DataFrame(results)
                        st.subheader("📊 Batch Results")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total", len(results_df))
                        with col2:
                            passed = results_df['passed'].sum()
                            st.metric("Passed", passed, delta=f"{passed/len(results_df)*100:.1f}%")
                        with col3:
                            failed = len(results_df) - passed
                            st.metric("Failed", failed)
                        with col4:
                            avg_flags = results_df['flags_count'].mean()
                            st.metric("Avg Flags", f"{avg_flags:.1f}")
                        
                        # Results table
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Score distributions
                        st.subheader("Score Distributions")
                        score_cols = ['toxicity', 'alignment', 'hallucination_risk', 'pii_risk', 'injection_risk']
                        available_scores = [c for c in score_cols if c in results_df.columns and results_df[c].sum() > 0]
                        
                        if available_scores:
                            fig = go.Figure()
                            for score in available_scores:
                                fig.add_trace(go.Box(y=results_df[score], name=score.replace('_', ' ').title()))
                            fig.update_layout(title="Score Distributions", yaxis_title="Score")
                            st.plotly_chart(fig, use_container_width=True)
                        
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
        history_df = pd.DataFrame(st.session_state.evaluation_history)
        
        # Summary metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
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
        with col5:
            total_flags = sum(len(f) for f in history_df['flags'])
            st.metric("Total Flags", total_flags)
        
        # Charts section
        st.markdown("---")
        
        # Pass/Fail pie chart and trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Pass/Fail Distribution")
            pass_counts = history_df['passed'].value_counts()
            fig = px.pie(
                values=pass_counts.values, 
                names=['Passed' if v else 'Failed' for v in pass_counts.index],
                color_discrete_map={'Passed': 'green', 'Failed': 'red'}
            )
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Evaluation Timeline")
            if len(history_df) > 1:
                timeline_df = history_df.copy()
                timeline_df['color'] = timeline_df['passed'].map({True: 'Passed', False: 'Failed'})
                timeline_df['index'] = range(len(timeline_df))
                
                fig = px.scatter(
                    timeline_df, x='index', y='passed',
                    color='color',
                    color_discrete_map={'Passed': 'green', 'Failed': 'red'},
                    labels={'index': 'Evaluation #', 'passed': 'Result'}
                )
                fig.update_layout(showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
        
        # Score distributions
        st.markdown("#### Score Distributions")
        
        # Extract all scores
        all_scores = {}
        for scores in history_df['scores']:
            for key, value in scores.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(value)
        
        if all_scores:
            num_metrics = len(all_scores)
            cols = st.columns(min(num_metrics, 3))
            
            for i, (metric, values) in enumerate(all_scores.items()):
                with cols[i % 3]:
                    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=20)])
                    fig.update_layout(
                        title=metric.replace('_', ' ').title(),
                        xaxis_title="Score",
                        yaxis_title="Count",
                        height=250,
                        margin=dict(l=40, r=20, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Flag analysis
        if total_flags > 0:
            st.markdown("#### Flag Analysis")
            all_flags = []
            for flags in history_df['flags']:
                all_flags.extend(flags)
            
            flag_counts = pd.Series(all_flags).value_counts()
            fig = px.bar(
                x=flag_counts.index,
                y=flag_counts.values,
                labels={'x': 'Flag Type', 'y': 'Count'},
                color=flag_counts.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📊 No evaluation data available. Run some evaluations first!")

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
    st.subheader("⚙️ Custom Rules")
    
    st.markdown("""
    Define custom content rules to enforce specific policies. Rules can use:
    - **Keywords**: Simple word matching
    - **Regex**: Pattern matching for complex rules
    """)
    
    # Add new rule form
    with st.form("add_rule_form"):
        st.markdown("#### Add New Rule")
        
        col1, col2 = st.columns(2)
        with col1:
            rule_name = st.text_input("Rule Name", placeholder="e.g., no_competitor_mentions")
            rule_type = st.selectbox("Rule Type", ["keyword", "regex"])
        
        with col2:
            rule_pattern = st.text_input("Pattern", placeholder="e.g., competitor|rival")
            rule_action = st.selectbox("Action", ["flag", "block"])
        
        rule_description = st.text_area("Description", placeholder="Describe what this rule detects...")
        
        submitted = st.form_submit_button("Add Rule", type="primary")
        
        if submitted and rule_name and rule_pattern:
            new_rule = {
                "name": rule_name,
                "type": rule_type,
                "pattern": rule_pattern,
                "action": rule_action,
                "description": rule_description
            }
            st.session_state.custom_rules.append(new_rule)
            st.success(f"✅ Rule '{rule_name}' added successfully!")
            st.rerun()
    
    # Display existing rules
    st.markdown("---")
    st.markdown("#### Active Rules")
    
    if st.session_state.custom_rules:
        for i, rule in enumerate(st.session_state.custom_rules):
            with st.expander(f"📋 {rule['name']}", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**Type:** {rule['type']}")
                    st.markdown(f"**Pattern:** `{rule['pattern']}`")
                with col2:
                    st.markdown(f"**Action:** {rule['action']}")
                    st.markdown(f"**Description:** {rule.get('description', 'N/A')}")
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_rule_{i}"):
                        st.session_state.custom_rules.pop(i)
                        st.rerun()
    else:
        st.info("No custom rules defined. Add rules above to enforce content policies.")
    
    # Import/Export rules
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Export Rules")
        if st.session_state.custom_rules:
            rules_json = json.dumps(st.session_state.custom_rules, indent=2)
            st.download_button(
                "📤 Export Rules JSON",
                rules_json,
                "custom_rules.json",
                "application/json"
            )
    
    with col2:
        st.markdown("#### Import Rules")
        uploaded_rules = st.file_uploader("Upload rules JSON", type=['json'], key="rules_upload")
        if uploaded_rules:
            try:
                imported_rules = json.load(uploaded_rules)
                if st.button("Import"):
                    st.session_state.custom_rules.extend(imported_rules)
                    st.success(f"✅ Imported {len(imported_rules)} rules")
                    st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON file")

with tab5:
    st.subheader("ℹ️ Help & Documentation")
    
    display_metric_explanation()
    
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### Getting Started
        
        1. **Configure Settings**: Use the sidebar to select evaluators and adjust thresholds
        2. **Choose Mode**: Select Local Pipeline or connect to an API Server
        3. **Enter Content**: Input a prompt and response in the Evaluate tab
        4. **Run Evaluation**: Click Evaluate to analyze the content
        5. **Review Results**: Check scores, flags, and detailed analysis
        6. **View Analytics**: Use the Analytics tab to see trends and distributions
        
        ### What Each Evaluator Does
        
        | Evaluator | Purpose | Good Score |
        |-----------|---------|------------|
        | Toxicity | Detects harmful, offensive content | < 0.3 |
        | Alignment | Measures response-prompt relevance | > 0.7 |
        | Hallucination | Detects fabricated information | < 0.3 |
        | PII Detection | Finds personal data (emails, SSN, etc.) | 0 |
        | Prompt Injection | Detects manipulation attempts | < 0.2 |
        | Custom Rules | Your defined content policies | Pass |
        """)
    
    with st.expander("🔧 Python API Usage"):
        st.code("""
from guardrails import GuardrailPipeline

# Initialize with all evaluators
pipeline = GuardrailPipeline(
    enable_toxicity=True,
    enable_hallucination=True,
    enable_alignment=True,
    enable_pii=True,
    enable_injection=True,
    enable_custom_rules=True,
    toxicity_threshold=0.7,
    alignment_threshold=0.5
)

# Evaluate content
result = pipeline.evaluate(
    prompt="Your prompt here",
    response="Model response here"
)

print(f"Passed: {result.passed}")
print(f"Scores: {result.scores}")
print(f"Flags: {result.flags}")

# Access detailed results
for evaluator, details in result.detailed_results.items():
    print(f"{evaluator}: {details}")
        """, language="python")
    
    with st.expander("🌐 REST API Usage"):
        st.code("""
# Using curl
curl -X POST http://localhost:8000/evaluate \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello", "response": "Hi there!"}'

# Using Python requests
import requests

response = requests.post(
    "http://localhost:8000/evaluate",
    json={
        "prompt": "Explain quantum computing",
        "response": "Quantum computing uses qubits..."
    }
)

result = response.json()
print(f"Passed: {result['passed']}")
print(f"Scores: {result['scores']}")
        """, language="bash")
    
    with st.expander("📦 Batch Processing"):
        st.markdown("""
        ### CSV Batch Processing
        
        1. Enable "Batch Evaluation Mode" in the sidebar
        2. Prepare a CSV file with these columns:
           - `prompt`: The input prompt
           - `response`: The model response
        3. Upload the file and click "Run Batch Evaluation"
        4. Download results as CSV
        
        ### CLI Batch Processing
        
        ```bash
        # Evaluate from CSV file
        guardrails evaluate-file data/samples.csv --output results.csv
        
        # With JSON output
        guardrails evaluate-file data/samples.csv --format json
        ```
        """)
    
    with st.expander("🐳 Docker Deployment"):
        st.markdown("""
        ### Quick Start with Docker Compose
        
        ```bash
        # Start all services (API, Dashboard, DB, Monitoring)
        docker-compose up -d
        
        # View logs
        docker-compose logs -f guardrail-api
        
        # Stop services
        docker-compose down
        ```
        
        ### Access Points
        
        | Service | URL |
        |---------|-----|
        | API Server | http://localhost:8000 |
        | Dashboard | http://localhost:8501 |
        | API Docs | http://localhost:8000/docs |
        | Prometheus | http://localhost:9090 |
        | Grafana | http://localhost:3000 |
        """)
    
    with st.expander("🔒 Security Best Practices"):
        st.markdown("""
        ### API Security
        
        1. **API Keys**: Set `API_KEY` environment variable for authentication
        2. **HTTPS**: Use reverse proxy (nginx) with SSL in production
        3. **Rate Limiting**: Configure rate limits for public endpoints
        
        ### Content Security
        
        1. **PII Detection**: Enable PII detection to prevent data leaks
        2. **Prompt Injection**: Always enable injection detection for user inputs
        3. **Custom Rules**: Define rules for your specific compliance needs
        
        ### Audit Logging
        
        All evaluations are logged to the database with:
        - Timestamp
        - Input hashes (no PII stored)
        - Results and flags
        - User context (if provided)
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        "<div style='text-align: center'>Built with ❤️ using Transformers, SentenceTransformers, and Streamlit</div>", 
        unsafe_allow_html=True
    )
