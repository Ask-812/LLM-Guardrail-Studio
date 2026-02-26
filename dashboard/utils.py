"""
Utility functions for dashboard components
"""

from typing import Dict, Any
import streamlit as st
from datetime import datetime


def format_score(score: float, metric_type: str) -> str:
    """Format score for display"""
    if metric_type == "percentage":
        return f"{score * 100:.1f}%"
    else:
        return f"{score:.3f}"


def get_status_color(score: float, metric_type: str) -> str:
    """Get color based on score and metric type"""
    # Metrics where lower is better
    if metric_type in ["toxicity", "hallucination_risk", "pii_risk", "injection_risk"]:
        if metric_type == "pii_risk":
            if score > 0.3:
                return "red"
            elif score > 0.1:
                return "orange"
            else:
                return "green"
        elif metric_type == "injection_risk":
            if score > 0.5:
                return "red"
            elif score > 0.2:
                return "orange"
            else:
                return "green"
        elif metric_type == "toxicity":
            if score > 0.7:
                return "red"
            elif score > 0.4:
                return "orange"
            else:
                return "green"
        else:  # hallucination_risk
            if score > 0.6:
                return "red"
            elif score > 0.3:
                return "orange"
            else:
                return "green"
    # Metrics where higher is better
    elif metric_type == "alignment":
        if score > 0.7:
            return "green"
        elif score > 0.4:
            return "orange"
        else:
            return "red"
    else:
        return "blue"


def display_metric_explanation():
    """Display explanation of metrics"""
    with st.expander("📊 Metric Explanations"):
        st.markdown("""
        ### Core Safety Metrics
        
        **Toxicity Score** (0-1, lower is better)
        Measures harmful, offensive, or inappropriate content using ML-based detection.
        - 🟢 < 0.3: Safe content
        - 🟡 0.3-0.6: Moderate concern - review recommended
        - 🔴 > 0.6: High toxicity - likely blocked
        
        **Alignment Score** (0-1, higher is better)
        Semantic similarity between prompt intent and response using embeddings.
        - 🔴 < 0.4: Poor alignment - response may be off-topic
        - 🟡 0.4-0.7: Moderate alignment - acceptable
        - 🟢 > 0.7: Good alignment - response matches intent
        
        **Hallucination Risk** (0-1, lower is better)
        Detects signs of fabricated information, overconfidence, or inconsistencies.
        - 🟢 < 0.3: Low risk - likely factual
        - 🟡 0.3-0.5: Moderate risk - verify claims
        - 🔴 > 0.5: High risk - likely contains fabrications
        
        ---
        
        ### Security Metrics
        
        **PII Risk** (0-1, lower is better)
        Detects personal identifiable information: emails, phone numbers, SSN, credit cards, addresses.
        - 🟢 0: No PII detected
        - 🟡 0.1-0.3: Minor PII (e.g., name mentioned)
        - 🔴 > 0.3: Sensitive PII detected - review required
        
        **Injection Risk** (0-1, lower is better)
        Detects prompt injection attempts: role manipulation, jailbreaks, extraction attacks.
        - 🟢 < 0.2: No injection patterns
        - 🟡 0.2-0.5: Suspicious patterns - review
        - 🔴 > 0.5: Likely injection attempt - blocked
        
        ---
        
        ### Custom Rules
        
        **Custom Violations** (count)
        Number of custom content policy rules violated.
        - 🟢 0: All rules passed
        - 🔴 > 0: Rules violated - check detailed results
        """)


def create_download_report(evaluation_data: Dict[str, Any]) -> str:
    """Create comprehensive downloadable evaluation report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# LLM Guardrail Studio - Evaluation Report

**Generated**: {timestamp}

---

## Input

### Prompt
```
{evaluation_data.get('prompt', 'N/A')}
```

### Response
```
{evaluation_data.get('response', 'N/A')}
```

---

## Evaluation Results

**Overall Status**: {'✅ PASSED' if evaluation_data.get('passed', False) else '❌ FAILED'}

### Scores
"""
    
    scores = evaluation_data.get('scores', {})
    for metric, score in scores.items():
        color = get_status_color(score, metric)
        icon = "🟢" if color == "green" else "🟡" if color == "orange" else "🔴"
        report += f"- {icon} **{metric.replace('_', ' ').title()}**: {score:.3f}\n"
    
    flags = evaluation_data.get('flags', [])
    if flags:
        report += "\n### ⚠️ Issues Detected\n"
        for flag in flags:
            report += f"- {flag}\n"
    else:
        report += "\n### ✅ No Issues Detected\n"
    
    # Detailed results
    detailed = evaluation_data.get('detailed_results', {})
    if detailed:
        report += "\n---\n\n## Detailed Analysis\n"
        
        for evaluator, details in detailed.items():
            if details:
                report += f"\n### {evaluator.replace('_', ' ').title()}\n"
                
                if evaluator == "pii" and details.get("categories"):
                    for cat, items in details["categories"].items():
                        if items:
                            report += f"- **{cat.title()}**: {len(items)} found\n"
                
                elif evaluator == "injection" and details.get("categories"):
                    report += "- **Detected patterns**: " + ", ".join(details["categories"]) + "\n"
                    if details.get("severity"):
                        report += f"- **Severity**: {details['severity']}\n"
                
                elif evaluator == "toxicity" and details.get("categories"):
                    report += "| Category | Score |\n|----------|-------|\n"
                    for cat, score in details["categories"].items():
                        report += f"| {cat.title()} | {score:.3f} |\n"
                
                elif evaluator == "hallucination" and details.get("indicators"):
                    report += "| Indicator | Score |\n|-----------|-------|\n"
                    for ind, score in details["indicators"].items():
                        report += f"| {ind.replace('_', ' ').title()} | {score:.3f} |\n"
                
                elif evaluator == "custom_rules" and details.get("violations"):
                    for v in details["violations"]:
                        report += f"- **{v.get('rule', 'Unknown')}**: {v.get('description', '')}\n"
    
    report += f"""
---

## Recommendations

"""
    
    # Add recommendations based on results
    if not evaluation_data.get('passed', True):
        if scores.get('toxicity', 0) > 0.5:
            report += "- ⚠️ Review response for harmful content and consider rephrasing\n"
        if scores.get('alignment', 1) < 0.5:
            report += "- ⚠️ Response may not address the prompt - consider regenerating\n"
        if scores.get('hallucination_risk', 0) > 0.5:
            report += "- ⚠️ Verify factual claims with authoritative sources\n"
        if scores.get('pii_risk', 0) > 0:
            report += "- 🔒 Remove or mask personal information before sharing\n"
        if scores.get('injection_risk', 0) > 0.3:
            report += "- 🛡️ Input contains suspicious patterns - validate user intent\n"
    else:
        report += "- ✅ Content appears safe for use\n"
        report += "- ✅ Consider periodic re-evaluation for changing content policies\n"
    
    report += f"""
---

*Generated by LLM Guardrail Studio*  
*https://github.com/your-repo/llm-guardrail-studio*
"""
    
    return report