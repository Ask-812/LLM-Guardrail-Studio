#!/usr/bin/env python3
"""
LLM Guardrail Studio - Command Line Interface

A comprehensive CLI for evaluating LLM outputs, managing configurations,
and running batch evaluations.

Usage:
    guardrail evaluate "prompt" "response"
    guardrail evaluate-file input.csv -o results.csv
    guardrail server --port 8000
    guardrail config show
    guardrail config set toxicity_threshold 0.5
"""

import sys
import os
import json
import csv
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from guardrails import (
    GuardrailPipeline, 
    GuardrailResult,
    GuardrailConfig,
    ConfigManager
)

console = Console()


# =============================================================================
# CLI Configuration
# =============================================================================

class Config:
    """CLI configuration container"""
    def __init__(self):
        self.verbose = False
        self.config_path = "config.json"
        self.output_format = "table"

pass_config = click.make_pass_decorator(Config, ensure=True)


# =============================================================================
# Main CLI Group
# =============================================================================

@click.group()
@click.version_option(version="1.0.0", prog_name="LLM Guardrail Studio")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', default='config.json', help='Path to config file')
@click.option('--format', '-f', 'output_format', 
              type=click.Choice(['table', 'json', 'csv']), 
              default='table', help='Output format')
@pass_config
def cli(config: Config, verbose: bool, config_path: str, output_format: str):
    """
    🛡️ LLM Guardrail Studio CLI
    
    A modular safety and moderation pipeline for local LLMs.
    Evaluate prompts and responses for toxicity, hallucinations, PII, and more.
    
    Examples:
    
        # Single evaluation
        guardrail evaluate "What is AI?" "AI is artificial intelligence."
        
        # Batch evaluation from file
        guardrail evaluate-file prompts.csv --output results.csv
        
        # Start API server
        guardrail server --port 8000
        
        # Show configuration
        guardrail config show
    """
    config.verbose = verbose
    config.config_path = config_path
    config.output_format = output_format


# =============================================================================
# Evaluate Commands
# =============================================================================

@cli.command()
@click.argument('prompt')
@click.argument('response')
@click.option('--toxicity/--no-toxicity', default=True, help='Enable toxicity detection')
@click.option('--hallucination/--no-hallucination', default=True, help='Enable hallucination detection')
@click.option('--alignment/--no-alignment', default=True, help='Enable alignment checking')
@click.option('--pii/--no-pii', default=False, help='Enable PII detection')
@click.option('--injection/--no-injection', default=False, help='Enable prompt injection detection')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed results')
@pass_config
def evaluate(
    config: Config,
    prompt: str,
    response: str,
    toxicity: bool,
    hallucination: bool,
    alignment: bool,
    pii: bool,
    injection: bool,
    detailed: bool
):
    """
    Evaluate a single prompt-response pair.
    
    Example:
        guardrail evaluate "What is Python?" "Python is a programming language."
    """
    with console.status("[bold green]Initializing pipeline..."):
        pipeline = GuardrailPipeline(
            enable_toxicity=toxicity,
            enable_hallucination=hallucination,
            enable_alignment=alignment,
            enable_pii=pii,
            enable_prompt_injection=injection
        )
    
    with console.status("[bold green]Evaluating..."):
        start_time = time.time()
        result = pipeline.evaluate(prompt, response)
        elapsed = (time.time() - start_time) * 1000
    
    # Display results
    _display_result(result, config.output_format, detailed, elapsed)


@cli.command('evaluate-file')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--prompt-col', default='prompt', help='Column name for prompts')
@click.option('--response-col', default='response', help='Column name for responses')
@click.option('--parallel', '-p', default=4, help='Number of parallel workers')
@click.option('--toxicity/--no-toxicity', default=True)
@click.option('--hallucination/--no-hallucination', default=True)
@click.option('--alignment/--no-alignment', default=True)
@click.option('--pii/--no-pii', default=False)
@click.option('--injection/--no-injection', default=False)
@pass_config
def evaluate_file(
    config: Config,
    input_file: str,
    output: Optional[str],
    prompt_col: str,
    response_col: str,
    parallel: int,
    toxicity: bool,
    hallucination: bool,
    alignment: bool,
    pii: bool,
    injection: bool
):
    """
    Evaluate prompt-response pairs from a CSV file.
    
    Example:
        guardrail evaluate-file data.csv --output results.csv
    """
    import pandas as pd
    
    # Read input file
    console.print(f"[bold blue]Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    if prompt_col not in df.columns:
        console.print(f"[bold red]Error: Column '{prompt_col}' not found in file")
        raise click.Abort()
    
    if response_col not in df.columns:
        console.print(f"[bold red]Error: Column '{response_col}' not found in file")
        raise click.Abort()
    
    console.print(f"[green]Found {len(df)} rows to evaluate")
    
    # Initialize pipeline
    with console.status("[bold green]Initializing pipeline..."):
        pipeline = GuardrailPipeline(
            enable_toxicity=toxicity,
            enable_hallucination=hallucination,
            enable_alignment=alignment,
            enable_pii=pii,
            enable_prompt_injection=injection
        )
    
    # Prepare pairs
    pairs = list(zip(df[prompt_col].tolist(), df[response_col].tolist()))
    
    # Run evaluation with progress bar
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(pairs))
        
        if parallel > 1:
            batch_results = pipeline.evaluate_batch_parallel(pairs, max_workers=parallel)
            results = batch_results
            progress.update(task, completed=len(pairs))
        else:
            for prompt, response in pairs:
                result = pipeline.evaluate(prompt, response)
                results.append(result)
                progress.advance(task)
    
    # Build output dataframe
    output_data = []
    for i, result in enumerate(results):
        row = {
            'id': i,
            'passed': result.passed,
            'flags_count': len(result.flags),
            'flags': '; '.join(result.flags)
        }
        row.update(result.scores)
        output_data.append(row)
    
    output_df = pd.DataFrame(output_data)
    
    # Summary statistics
    passed_count = output_df['passed'].sum()
    failed_count = len(output_df) - passed_count
    
    # Display summary
    table = Table(title="📊 Batch Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Evaluations", str(len(output_df)))
    table.add_row("Passed", f"✅ {passed_count}")
    table.add_row("Failed", f"❌ {failed_count}")
    table.add_row("Pass Rate", f"{passed_count/len(output_df)*100:.1f}%")
    
    if 'toxicity' in output_df.columns:
        table.add_row("Avg Toxicity", f"{output_df['toxicity'].mean():.3f}")
    if 'alignment' in output_df.columns:
        table.add_row("Avg Alignment", f"{output_df['alignment'].mean():.3f}")
    
    console.print(table)
    
    # Save output
    if output:
        output_df.to_csv(output, index=False)
        console.print(f"[bold green]Results saved to {output}")
    else:
        console.print("\n[bold]Results Preview:[/bold]")
        console.print(output_df.head(10).to_string())


@cli.command()
@click.argument('text')
@click.option('--type', '-t', 'check_type', 
              type=click.Choice(['toxicity', 'pii', 'injection', 'hallucination']),
              default='toxicity', help='Type of check to run')
@click.option('--detailed', '-d', is_flag=True, help='Show detailed analysis')
@pass_config
def check(config: Config, text: str, check_type: str, detailed: bool):
    """
    Run a single check on text.
    
    Example:
        guardrail check "some text" --type toxicity
    """
    from guardrails.evaluators import (
        ToxicityEvaluator, PIIDetector, 
        PromptInjectionDetector, HallucinationDetector
    )
    
    evaluators = {
        'toxicity': ToxicityEvaluator,
        'pii': PIIDetector,
        'injection': PromptInjectionDetector,
        'hallucination': HallucinationDetector
    }
    
    with console.status(f"[bold green]Running {check_type} check..."):
        evaluator = evaluators[check_type]()
        
        if detailed:
            result = evaluator.evaluate_detailed(text)
            score = result.score
            details = result.details
        else:
            score = evaluator.evaluate(text)
            details = None
    
    # Display result
    if score < 0.3:
        color = "green"
        status = "✅ Low Risk"
    elif score < 0.6:
        color = "yellow"
        status = "⚠️ Medium Risk"
    else:
        color = "red"
        status = "❌ High Risk"
    
    console.print(Panel(
        f"[bold {color}]{status}[/bold {color}]\n\n"
        f"Score: {score:.3f}",
        title=f"🔍 {check_type.title()} Check"
    ))
    
    if detailed and details:
        console.print("\n[bold]Detailed Analysis:[/bold]")
        console.print(Syntax(json.dumps(details, indent=2), "json"))


# =============================================================================
# Server Commands
# =============================================================================

@cli.command()
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind to')
@click.option('--port', '-p', default=8000, help='Port to bind to')
@click.option('--reload', '-r', is_flag=True, help='Enable auto-reload')
@click.option('--workers', '-w', default=1, help='Number of workers')
@pass_config
def server(config: Config, host: str, port: int, reload: bool, workers: int):
    """
    Start the REST API server.
    
    Example:
        guardrail server --port 8000 --reload
    """
    console.print(Panel(
        f"🚀 Starting LLM Guardrail Studio API Server\n\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Workers: {workers}\n"
        f"Reload: {'Enabled' if reload else 'Disabled'}\n\n"
        f"API Docs: http://{host}:{port}/docs\n"
        f"Health: http://{host}:{port}/health",
        title="🛡️ LLM Guardrail Studio"
    ))
    
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers
    )


# =============================================================================
# Configuration Commands
# =============================================================================

@cli.group()
@pass_config
def config(config: Config):
    """
    Manage pipeline configuration.
    """
    pass


@config.command('show')
@pass_config
def config_show(config: Config):
    """Show current configuration."""
    try:
        cfg = ConfigManager.load_or_create(config.config_path)
    except Exception:
        cfg = GuardrailConfig()
    
    table = Table(title="⚙️ Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in cfg.to_dict().items():
        table.add_row(key, str(value))
    
    console.print(table)


@config.command('set')
@click.argument('key')
@click.argument('value')
@pass_config
def config_set(config: Config, key: str, value: str):
    """Set a configuration value."""
    try:
        cfg = ConfigManager.load_or_create(config.config_path)
    except Exception:
        cfg = GuardrailConfig()
    
    # Parse value
    if value.lower() in ['true', 'false']:
        value = value.lower() == 'true'
    elif value.replace('.', '').isdigit():
        value = float(value) if '.' in value else int(value)
    
    if hasattr(cfg, key):
        setattr(cfg, key, value)
        cfg.save(config.config_path)
        console.print(f"[green]✅ Set {key} = {value}")
    else:
        console.print(f"[red]❌ Unknown configuration key: {key}")


@config.command('init')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing config')
@pass_config
def config_init(config: Config, force: bool):
    """Initialize a new configuration file."""
    path = Path(config.config_path)
    
    if path.exists() and not force:
        console.print(f"[yellow]⚠️ Config file already exists: {path}")
        if not click.confirm("Overwrite?"):
            return
    
    cfg = GuardrailConfig()
    cfg.save(str(path))
    console.print(f"[green]✅ Created configuration file: {path}")


# =============================================================================
# Rules Commands
# =============================================================================

@cli.group()
@pass_config
def rules(config: Config):
    """
    Manage custom filtering rules.
    """
    pass


@rules.command('add')
@click.option('--id', 'rule_id', required=True, help='Unique rule identifier')
@click.option('--name', required=True, help='Rule name')
@click.option('--pattern', required=True, help='Pattern (regex or keywords)')
@click.option('--type', 'rule_type', type=click.Choice(['regex', 'keyword']), 
              default='keyword', help='Pattern type')
@click.option('--severity', default=0.5, help='Severity score (0-1)')
@click.option('--action', type=click.Choice(['flag', 'block']), default='flag')
@pass_config  
def rules_add(config: Config, rule_id: str, name: str, pattern: str, 
              rule_type: str, severity: float, action: str):
    """Add a custom filtering rule."""
    rules_file = Path("custom_rules.json")
    
    # Load existing rules
    rules_list = []
    if rules_file.exists():
        rules_list = json.loads(rules_file.read_text())
    
    # Check for duplicate
    if any(r['id'] == rule_id for r in rules_list):
        console.print(f"[red]❌ Rule with ID '{rule_id}' already exists")
        return
    
    # Add new rule
    new_rule = {
        "id": rule_id,
        "name": name,
        "pattern": pattern,
        "type": rule_type,
        "severity": severity,
        "action": action,
        "enabled": True
    }
    
    rules_list.append(new_rule)
    rules_file.write_text(json.dumps(rules_list, indent=2))
    
    console.print(f"[green]✅ Added rule: {name}")


@rules.command('list')
@pass_config
def rules_list(config: Config):
    """List all custom rules."""
    rules_file = Path("custom_rules.json")
    
    if not rules_file.exists():
        console.print("[yellow]No custom rules defined")
        return
    
    rules_list = json.loads(rules_file.read_text())
    
    table = Table(title="📋 Custom Rules")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Type")
    table.add_column("Severity")
    table.add_column("Action")
    table.add_column("Enabled")
    
    for rule in rules_list:
        table.add_row(
            rule['id'],
            rule['name'],
            rule['type'],
            f"{rule['severity']:.2f}",
            rule['action'],
            "✅" if rule.get('enabled', True) else "❌"
        )
    
    console.print(table)


@rules.command('remove')
@click.argument('rule_id')
@pass_config
def rules_remove(config: Config, rule_id: str):
    """Remove a custom rule."""
    rules_file = Path("custom_rules.json")
    
    if not rules_file.exists():
        console.print("[red]No custom rules file found")
        return
    
    rules_list = json.loads(rules_file.read_text())
    new_rules = [r for r in rules_list if r['id'] != rule_id]
    
    if len(new_rules) == len(rules_list):
        console.print(f"[red]❌ Rule '{rule_id}' not found")
        return
    
    rules_file.write_text(json.dumps(new_rules, indent=2))
    console.print(f"[green]✅ Removed rule: {rule_id}")


# =============================================================================
# Info Commands
# =============================================================================

@cli.command()
@pass_config
def info(config: Config):
    """Show information about available evaluators."""
    from guardrails.evaluators import EvaluatorRegistry
    
    console.print(Panel(
        "🛡️ LLM Guardrail Studio\n\n"
        "A modular trust layer for local LLMs providing comprehensive\n"
        "safety and moderation capabilities.",
        title="About"
    ))
    
    table = Table(title="📦 Available Evaluators")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    
    evaluator_info = EvaluatorRegistry.get_info()
    for name, info in evaluator_info.items():
        table.add_row(name, info.get('description', ''))
    
    console.print(table)


@cli.command()
@pass_config
def version(config: Config):
    """Show version information."""
    from guardrails import __version__
    console.print(f"[bold]LLM Guardrail Studio[/bold] v{__version__}")


# =============================================================================
# Helper Functions
# =============================================================================

def _display_result(
    result: GuardrailResult, 
    format: str, 
    detailed: bool,
    elapsed_ms: float
):
    """Display evaluation result in specified format"""
    
    if format == 'json':
        console.print(Syntax(result.to_json(), "json"))
        return
    
    if format == 'csv':
        row = [result.evaluation_id, str(result.passed)]
        row.extend([f"{v:.3f}" for v in result.scores.values()])
        row.append(';'.join(result.flags))
        console.print(','.join(row))
        return
    
    # Table format
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    status_color = "green" if result.passed else "red"
    
    console.print(Panel(
        f"[bold {status_color}]{status}[/bold {status_color}]",
        title="🛡️ Evaluation Result"
    ))
    
    # Scores table
    if result.scores:
        table = Table(title="📊 Scores")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Status")
        
        for metric, score in result.scores.items():
            if metric == 'alignment':
                status_icon = "✅" if score >= 0.5 else "⚠️"
            else:
                status_icon = "✅" if score <= 0.5 else "⚠️"
            
            table.add_row(
                metric.replace('_', ' ').title(),
                f"{score:.3f}",
                status_icon
            )
        
        console.print(table)
    
    # Flags
    if result.flags:
        console.print("\n[bold red]🚩 Flags:[/bold red]")
        for flag in result.flags:
            console.print(f"  • {flag}")
    
    # Metadata
    console.print(f"\n[dim]Evaluation ID: {result.evaluation_id}")
    console.print(f"[dim]Time: {elapsed_ms:.1f}ms")
    
    # Detailed results
    if detailed and result._detailed_results:
        console.print("\n[bold]📋 Detailed Analysis:[/bold]")
        console.print(Syntax(
            json.dumps(result._detailed_results, indent=2), 
            "json"
        ))


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for CLI"""
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        if os.getenv('DEBUG'):
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
