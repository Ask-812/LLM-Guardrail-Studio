"""
Reusable Streamlit components for the dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import pandas as pd


class MetricsDisplay:
    """Component for displaying evaluation metrics"""
    
    @staticmethod
    def render_score_cards(scores: Dict[str, float]):
        """Render metric score cards"""
        if not scores:
            st.info("No scores available")
            return
        
        cols = st.columns(len(scores))
        
        for i, (metric, score) in enumerate(scores.items()):
            with cols[i]:
                # Color coding based on metric type and value
                if metric == "toxicity":
                    color = "red" if score > 0.7 else "orange" if score > 0.4 else "green"
                elif metric == "alignment":
                    color = "green" if score > 0.7 else "orange" if score > 0.4 else "red"
                elif metric == "hallucination_risk":
                    color = "red" if score > 0.6 else "orange" if score > 0.3 else "green"
                else:
                    color = "blue"
                
                st.metric(
                    label=metric.replace('_', ' ').title(),
                    value=f"{score:.3f}",
                    delta=None
                )
    
    @staticmethod
    def render_radar_chart(scores: Dict[str, float]):
        """Render radar chart for scores"""
        if not scores:
            return
        
        # Prepare data for radar chart
        metrics = list(scores.keys())
        values = list(scores.values())
        
        # Normalize values for better visualization
        normalized_values = []
        for metric, value in zip(metrics, values):
            if metric == "alignment":
                # Alignment: higher is better, scale 0-1
                normalized_values.append(value)
            elif metric == "toxicity":
                # Toxicity: lower is better, invert scale
                normalized_values.append(1 - value)
            elif metric == "hallucination_risk":
                # Hallucination: lower is better, invert scale
                normalized_values.append(1 - value)
            else:
                normalized_values.append(value)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=metrics,
            fill='toself',
            name='Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Evaluation Metrics Overview"
        )
        
        st.plotly_chart(fig, use_container_width=True)


class FlagDisplay:
    """Component for displaying evaluation flags"""
    
    @staticmethod
    def render_flags(flags: List[str]):
        """Render flag alerts"""
        if not flags:
            st.success("✅ No issues detected")
            return
        
        st.error(f"⚠️ {len(flags)} issue(s) detected:")
        
        for flag in flags:
            st.warning(f"• {flag}")


class ModelSelector:
    """Component for model selection and configuration"""
    
    @staticmethod
    def render_model_config():
        """Render model configuration sidebar"""
        st.sidebar.header("🤖 Model Configuration")
        
        model_options = {
            "Mistral 7B": "mistralai/Mistral-7B-Instruct-v0.2",
            "Zephyr 7B": "HuggingFaceH4/zephyr-7b-beta",
            "Llama 2 7B": "meta-llama/Llama-2-7b-chat-hf",
            "Custom": "custom"
        }
        
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        
        if selected_model == "Custom":
            model_name = st.sidebar.text_input(
                "Custom Model Name",
                placeholder="huggingface/model-name"
            )
        else:
            model_name = model_options[selected_model]
        
        # Generation parameters
        st.sidebar.subheader("Generation Parameters")
        max_length = st.sidebar.slider("Max Length", 50, 1024, 512)
        temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7)
        
        return {
            "model_name": model_name,
            "max_length": max_length,
            "temperature": temperature
        }


class HistoryDisplay:
    """Component for displaying evaluation history"""
    
    @staticmethod
    def render_history_table(history: List[Dict]):
        """Render evaluation history as a table"""
        if not history:
            st.info("No evaluation history available")
            return
        
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def render_trend_chart(history: List[Dict]):
        """Render trend chart for metrics over time"""
        if not history:
            return
        
        df = pd.DataFrame(history)
        
        if 'timestamp' not in df.columns:
            return
        
        # Create trend chart for each metric
        metrics = [col for col in df.columns if col not in ['timestamp', 'prompt', 'response', 'flags']]
        
        fig = go.Figure()
        
        for metric in metrics:
            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title()
                ))
        
        fig.update_layout(
            title="Metrics Trend Over Time",
            xaxis_title="Time",
            yaxis_title="Score",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)