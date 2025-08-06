"""Command-line interface for LangTree."""

import argparse
import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from .orchestrator import ToTOrchestrator

# Optional pretty printing
try:
    from rich import print as rprint
    from rich.tree import Tree as RichTree
except Exception:  # pragma: no cover
    rprint = print
    RichTree = None


def build_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    """
    Build an LLM instance. Replace with your provider if needed.
    Requires OPENAI_API_KEY in env for ChatOpenAI.
    """
    return ChatOpenAI(model=model_name, temperature=temperature)


def pretty_print_result(result: Dict[str, Any]) -> None:
    """Pretty print the result with optional tree visualization."""
    rprint(f"[bold green]Best score:[/bold green] {result['best_score']:.3f} (conf={result['best_confidence']:.2f})")
    rprint("[bold]Actions along best path:[/bold]")
    rprint(result["best_actions"])
    rprint("[bold]Partial solutions along path:[/bold]")
    for i, s in enumerate(result["best_path"]):
        rprint(f"  [{i}] {s!r}")

    if RichTree:
        tree_json = result["tree_snapshot"]["nodes"]
        root_id = next(nid for nid, n in tree_json.items() if n["parent_id"] is None)
        
        def build(rt: RichTree, nid: str):
            n = tree_json[nid]
            label = f"{nid[:6]} d={n['depth']} score={n['score']:.2f} conf={n['confidence']:.2f} status={n['status']}"
            child = rt.add(label)
            for cid in n["children"]:
                build(child, cid)
                
        rt = RichTree("Tree-of-Thought")
        build(rt, root_id)
        rprint(rt)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="LangTree: Tree-of-Thought Orchestrator over LangChain")
    parser.add_argument("--task", type=str, required=True, help="Task/problem statement")
    parser.add_argument("--constraints", type=str, default="", help="Optional constraints")
    parser.add_argument("--model-expand", type=str, default="gpt-4o-mini", help="Model for expansions")
    parser.add_argument("--model-score", type=str, default="gpt-4o-mini", help="Model for scoring")
    parser.add_argument("--k-children", type=int, default=3, help="Children per expansion")
    parser.add_argument("--max-select", type=int, default=3, help="Nodes to expand per iteration")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum depth")
    parser.add_argument("--max-nodes", type=int, default=50, help="Maximum nodes in search")
    parser.add_argument("--target-score", type=float, default=0.9, help="Early stop when best score >= target")
    parser.add_argument("--beam", type=int, default=3, help="Beam width per depth")
    parser.add_argument("--min-score", type=float, default=0.0, help="Min score to keep nodes at a depth")
    parser.add_argument("--print-tree", action="store_true", help="Pretty print tree at the end")
    args = parser.parse_args()

    # Build models (can be same or different)
    llm_expand = build_llm(args.model_expand, temperature=0.7)
    llm_score = build_llm(args.model_score, temperature=0.0)

    orchestrator = ToTOrchestrator(
        llm_expand=llm_expand,
        llm_score=llm_score,
        k_children=args.k_children,
        max_select=args.max_select,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        target_score=args.target_score,
        beam_per_depth=args.beam,
        min_score=args.min_score,
    )

    result = orchestrator.run(task=args.task, constraints=args.constraints)

    if args.print_tree:
        pretty_print_result(result)
    else:
        # Output compact JSON result to stdout
        print(json.dumps({
            "best_score": result["best_score"],
            "best_confidence": result["best_confidence"],
            "best_state": result["best_state"],
            "best_actions": result["best_actions"],
            "best_path": result["best_path"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()