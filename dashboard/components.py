"""
Reusable Streamlit components for the dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import pandas as pd


class MetricsDisplay:
    """Component for displaying evaluation metrics"""
    
    # Color schemes for different metric types
    METRIC_COLORS = {
        # Lower is better metrics (show red when high)
        "toxicity": {"good": 0.3, "warning": 0.6, "reverse": True},
        "hallucination_risk": {"good": 0.3, "warning": 0.5, "reverse": True},
        "pii_risk": {"good": 0.1, "warning": 0.3, "reverse": True},
        "injection_risk": {"good": 0.2, "warning": 0.5, "reverse": True},
        "custom_violations": {"good": 0, "warning": 1, "reverse": True},
        # Higher is better metrics (show green when high)
        "alignment": {"good": 0.7, "warning": 0.4, "reverse": False},
    }
    
    @staticmethod
    def get_color_for_metric(metric: str, value: float) -> str:
        """Get color based on metric type and value"""
        config = MetricsDisplay.METRIC_COLORS.get(metric, {"good": 0.5, "warning": 0.3, "reverse": False})
        
        if config["reverse"]:
            # Lower is better
            if value <= config["good"]:
                return "green"
            elif value <= config["warning"]:
                return "orange"
            else:
                return "red"
        else:
            # Higher is better
            if value >= config["good"]:
                return "green"
            elif value >= config["warning"]:
                return "orange"
            else:
                return "red"
    
    @staticmethod
    def render_score_cards(scores: Dict[str, float]):
        """Render metric score cards"""
        if not scores:
            st.info("No scores available")
            return
        
        # Use up to 6 columns, wrap if more
        num_scores = len(scores)
        cols_per_row = min(num_scores, 6)
        cols = st.columns(cols_per_row)
        
        for i, (metric, score) in enumerate(scores.items()):
            col_index = i % cols_per_row
            with cols[col_index]:
                color = MetricsDisplay.get_color_for_metric(metric, score)
                icon = "🟢" if color == "green" else "🟡" if color == "orange" else "🔴"
                
                st.metric(
                    label=f"{icon} {metric.replace('_', ' ').title()}",
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
        
        # Normalize values for better visualization (all metrics scaled to "higher is better")
        normalized_values = []
        for metric, value in zip(metrics, values):
            config = MetricsDisplay.METRIC_COLORS.get(metric, {"reverse": False})
            if config.get("reverse", False):
                # Invert so higher appears better on chart
                normalized_values.append(1 - value)
            else:
                normalized_values.append(value)
        
        # Close the polygon
        metrics.append(metrics[0])
        normalized_values.append(normalized_values[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=metrics,
            fill='toself',
            name='Scores',
            line_color='rgb(31, 119, 180)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            showlegend=False,
            title="Safety Scores Overview (Higher = Safer)",
            margin=dict(l=80, r=80, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_detailed_results(detailed_results: Dict[str, Any]):
        """Render detailed evaluation results for each evaluator"""
        if not detailed_results:
            return
        
        for evaluator, details in detailed_results.items():
            if not details:
                continue
                
            with st.expander(f"📋 {evaluator.replace('_', ' ').title()} Details"):
                if evaluator == "pii":
                    MetricsDisplay._render_pii_details(details)
                elif evaluator == "injection":
                    MetricsDisplay._render_injection_details(details)
                elif evaluator == "toxicity":
                    MetricsDisplay._render_toxicity_details(details)
                elif evaluator == "hallucination":
                    MetricsDisplay._render_hallucination_details(details)
                elif evaluator == "custom_rules":
                    MetricsDisplay._render_custom_rules_details(details)
                else:
                    st.json(details)
    
    @staticmethod
    def _render_pii_details(details: Dict):
        """Render PII detection details"""
        if details.get("pii_found"):
            st.warning("⚠️ PII Detected")
            categories = details.get("categories", {})
            
            for category, items in categories.items():
                if items:
                    st.markdown(f"**{category.title()}:** {len(items)} found")
                    for item in items:
                        # Mask sensitive data for display
                        masked = item[:2] + "*" * (len(item) - 4) + item[-2:] if len(item) > 4 else "***"
                        st.text(f"  • {masked}")
        else:
            st.success("✅ No PII detected")
    
    @staticmethod
    def _render_injection_details(details: Dict):
        """Render prompt injection detection details"""
        if details.get("injection_detected"):
            st.error("🚨 Potential Injection Detected")
            
            if details.get("categories"):
                st.markdown("**Detected patterns:**")
                for category in details["categories"]:
                    st.markdown(f"  • {category.replace('_', ' ').title()}")
            
            if details.get("severity"):
                severity_colors = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "⛔"}
                icon = severity_colors.get(details["severity"], "⚠️")
                st.markdown(f"**Severity:** {icon} {details['severity'].upper()}")
        else:
            st.success("✅ No injection patterns detected")
    
    @staticmethod
    def _render_toxicity_details(details: Dict):
        """Render toxicity detection details"""
        categories = details.get("categories", {})
        
        if categories:
            df = pd.DataFrame([
                {"Category": cat.replace('_', ' ').title(), "Score": score}
                for cat, score in categories.items()
            ])
            
            fig = px.bar(df, x="Category", y="Score", color="Score",
                        color_continuous_scale=["green", "yellow", "red"],
                        range_color=[0, 1])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_hallucination_details(details: Dict):
        """Render hallucination detection details"""
        indicators = details.get("indicators", {})
        
        if indicators:
            cols = st.columns(len(indicators))
            for i, (indicator, value) in enumerate(indicators.items()):
                with cols[i]:
                    icon = "✅" if value < 0.3 else "⚠️" if value < 0.6 else "❌"
                    st.metric(
                        label=f"{icon} {indicator.replace('_', ' ').title()}",
                        value=f"{value:.2f}"
                    )
    
    @staticmethod
    def _render_custom_rules_details(details: Dict):
        """Render custom rules evaluation details"""
        violations = details.get("violations", [])
        
        if violations:
            st.error(f"Found {len(violations)} rule violation(s)")
            for v in violations:
                st.markdown(f"• **{v.get('rule', 'Unknown')}**: {v.get('description', '')}")
        else:
            st.success("✅ All custom rules passed")


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