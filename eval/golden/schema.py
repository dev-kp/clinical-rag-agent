from dataclasses import dataclass
from typing import Literal

QuestionCategory = Literal["single-hop", "multi-hop", "unanswerable"]


@dataclass(frozen=True)
class GoldenQuestion:
    """One hand-verified entry in the evaluation golden set.

    ground_truth_answer and ground_truth_chunk_ids are empty for
    "unanswerable" questions: there is no correct answer to compare
    against, the only thing being measured is whether the system
    correctly abstains instead of confabulating.
    """

    question_id: str
    question: str
    category: QuestionCategory
    ground_truth_answer: str
    ground_truth_chunk_ids: list[str]
    notes: str = ""
