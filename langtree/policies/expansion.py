"""Expansion policy for generating child candidates."""

import json
import time
import uuid
from typing import List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..models import ThoughtNode


class ExpansionPolicy:
    """
    Produces K child candidates per node using an LLM.
    Output schema: [{"thought": str, "action": str, "partial_solution": str}]
    """
    
    def __init__(self, llm: Runnable, k: int = 3, temperature: float = 0.7):
        self.k = k
        self.temperature = temperature
        self.llm = llm
        self.parser = JsonOutputParser()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a reasoning engine. Given a task and the current partial solution, "
                 f"propose {self.k} diverse next-step candidates. Output strict JSON list with fields: "
                 "thought, action, partial_solution. Encourage diversity: quick heuristic, rigorous, contrarian."),
                ("user",
                 "Task:\n{task}\n\nCurrent state (JSON):\n{state_json}\n\n"
                 "Constraints:\n{constraints}\n\n"
                 "Output JSON array only.")
            ]
        )

        self.chain: Runnable = (
            self.prompt
            | self.llm.bind(temperature=self.temperature)
            | self.parser
        )

    def expand(self, node: ThoughtNode, task: str, constraints: str = "") -> List[ThoughtNode]:
        """Generate child candidates for a node."""
        state_json = json.dumps(node.state or {}, ensure_ascii=False)
        try:
            results: List[dict[str, str]] = self.chain.invoke(
                {"task": task, "state_json": state_json, "constraints": constraints}
            )
        except Exception as e:
            # Fallback: single noop child
            results = [{"thought": f"Expansion failed: {e}", "action": "noop", "partial_solution": node.state.get("partial_solution", "")}]

        children: List[ThoughtNode] = []
        for item in results[: self.k]:
            child_state = {
                "reasoning_steps": (node.state.get("reasoning_steps", []) + [item.get("thought", "")]),
                "scratchpad": node.state.get("scratchpad", ""),
                "partial_solution": item.get("partial_solution", ""),
            }
            child = ThoughtNode(
                id=str(uuid.uuid4()),
                parent_id=node.id,
                depth=node.depth + 1,
                state=child_state,
                action=item.get("action", ""),
                status="open",
                metadata={"created_at": time.time()},
            )
            children.append(child)
        return children