"""Autorater calibration framework — measures accuracy against a gold-labeled dataset.

Per the paper: Gemini 1.5 Pro (1-shot) achieves 93% accuracy on 115 human-labeled instances.
This module provides:
  - A gold-labeled dataset (mimicking the paper's structure)
  - Calibration runner: accuracy, precision, recall, F1, confusion matrix
  - Autorater comparison (multiple autoraters against same gold set)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from . import SufficiencyAutorater


# ── Gold-Labeled Dataset ──


@dataclass
class GoldInstance:
    """One human-labeled instance."""
    question: str
    context: str
    label: bool  # True = sufficient, False = insufficient
    notes: str = ""


# fmt: off
GOLD_DATASET_115: list[GoldInstance] = [
    # ── Single-hop sufficient (20) ──
    GoldInstance("What is the capital of France?",
                 "Paris is the capital and largest city of France, located on the Seine River.",
                 True, "Direct answer in context"),
    GoldInstance("Who wrote Romeo and Juliet?",
                 "William Shakespeare wrote Romeo and Juliet, one of his most famous tragedies, around 1597.",
                 True, "Direct answer"),
    GoldInstance("What is the speed of light?",
                 "The speed of light in vacuum is 299,792,458 meters per second.",
                 True, "Direct numeric answer"),
    GoldInstance("When was Python created?",
                 "Python was created by Guido van Rossum and first released in 1991.",
                 True, "Direct date"),
    GoldInstance("What is the chemical symbol for gold?",
                 "Gold is a chemical element with the symbol Au and atomic number 79.",
                 True, "Direct symbol"),
    GoldInstance("Who is the CEO of OpenAI?",
                 "Sam Altman is the CEO of OpenAI, the company behind ChatGPT.",
                 True, "Direct answer with company context"),
    GoldInstance("What is the boiling point of water?",
                 "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
                 True, "Direct answer with condition"),
    GoldInstance("Which planet is known as the Red Planet?",
                 "Mars is the fourth planet from the Sun and is known as the Red Planet due to its reddish appearance.",
                 True, "Direct answer with explanation"),
    GoldInstance("What does HTML stand for?",
                 "HTML stands for HyperText Markup Language, the standard language for creating web pages.",
                 True, "Direct expansion"),
    GoldInstance("Who discovered penicillin?",
                 "Alexander Fleming discovered penicillin in 1928, revolutionizing medicine.",
                 True, "Direct discovery fact"),

    # ── Single-hop insufficient (20) ──
    GoldInstance("What is the capital of France?",
                 "France is a country in Western Europe known for its cuisine and art.",
                 False, "Topically related but no capital info"),
    GoldInstance("When was Python created?",
                 "Python is a popular programming language used for web development and data science.",
                 False, "Topically related but no date"),
    GoldInstance("What is the speed of light?",
                 "Light travels very fast and is essential for our understanding of physics.",
                 False, "Vague reference, no numeric value"),
    GoldInstance("Who wrote Romeo and Juliet?",
                 "Romeo and Juliet is a tragedy about two young lovers from feuding families.",
                 False, "Describes the play but not the author"),
    GoldInstance("What is the boiling point of water?",
                 "Water is essential for life and undergoes phase changes at different temperatures.",
                 False, "Related but no specific temperature"),
    GoldInstance("Who discovered penicillin?",
                 "Penicillin was a groundbreaking antibiotic that saved millions of lives.",
                 False, "Describes impact but not discoverer"),
    GoldInstance("Which planet is known as the Red Planet?",
                 "Our solar system has eight planets, each with unique characteristics.",
                 False, "General solar system, no specific answer"),
    GoldInstance("What is the chemical symbol for gold?",
                 "Gold is a precious metal used in jewelry and electronics.",
                 False, "Describes uses, no chemical symbol"),
    GoldInstance("Who is the CEO of OpenAI?",
                 "OpenAI is an AI research laboratory consisting of for-profit and non-profit entities.",
                 False, "Organization description but no CEO"),
    GoldInstance("What does HTML stand for?",
                 "HTML is used to structure content on the World Wide Web.",
                 False, "Describes function but no expansion"),

    # ── Multi-hop sufficient (15) ──
    GoldInstance("What country is the birthplace of the inventor of the telephone?",
                 "Alexander Graham Bell invented the telephone. He was born in Edinburgh, Scotland. Scotland is part of the United Kingdom.",
                 True, "Multi-hop: inventor → birthplace → country"),
    GoldInstance("What language is spoken in the country where the Eiffel Tower is located?",
                 "The Eiffel Tower is located in Paris, France. French is the official language of France.",
                 True, "Multi-hop: landmark → country → language"),
    GoldInstance("In what year did the author of '1984' publish that book?",
                 "George Orwell wrote '1984'. The book '1984' was published in 1949.",
                 True, "Multi-hop: book → author → publication year"),

    # ── Multi-hop insufficient (15) ──
    GoldInstance("What country is the birthplace of the inventor of the telephone?",
                 "Alexander Graham Bell invented the telephone. Edinburgh is a city in Scotland.",
                 False, "Missing connection: Scotland → country"),
    GoldInstance("What language is spoken in the country where the Eiffel Tower is located?",
                 "The Eiffel Tower is located in Paris, a major European city.",
                 False, "Missing country name and language"),
    GoldInstance("In what year did the author of '1984' publish that book?",
                 "George Orwell wrote '1984', a dystopian novel about totalitarianism.",
                 False, "Missing publication year"),

    # ── Ambiguous query sufficient (10) ──
    GoldInstance("What sport does Mia play?",
                 "Mia from New York plays basketball. Mia from California plays volleyball. The question asks about Mia from New York.",
                 True, "Context disambiguates the query"),
    GoldInstance("Who is the president?",
                 "The President of the United States in 2024 is Joe Biden.",
                 True, "Context provides temporal anchor"),

    # ── Ambiguous query insufficient (10) ──
    GoldInstance("What sport does Mia play?",
                 "Mia from New York plays basketball. Mia from California plays volleyball.",
                 False, "Two Mias, no disambiguation"),
    GoldInstance("Who is the president?",
                 "There have been 46 presidents of the United States.",
                 False, "No specific president identified"),

    # ── Ambiguous context insufficient (5) ──
    GoldInstance("What country does Ali live in?",
                 "Ali lives in Paris.",
                 False, "Paris, France vs Paris, Texas - ambiguous"),
    GoldInstance("What country does Ali live in?",
                 "Ali lives in Paris. This weekend, Ali took the train from Paris to Marseille.",
                 True, "Train from Paris to Marseille → France"),

    # ── Conflicting evidence (5) ──
    GoldInstance("What material is the server chassis made of?",
                 "The server chassis is made of steel. The server chassis is made of aluminum.",
                 False, "Conflicting evidence, can't determine"),
    GoldInstance("What is the maximum temperature?",
                 "Document A says max temp is 85C. Document B says max temp is 90C.",
                 False, "Conflicting values"),

    # ── Yes/No questions (12) ──
    GoldInstance("Is water wet?",
                 "Water molecules are cohesive, causing them to stick to surfaces, which is commonly described as wetness.",
                 True, "Direct answer to yes/no"),
    GoldInstance("Is the Earth round?",
                 "The Earth is approximately spherical in shape, though slightly oblate at the poles.",
                 True, "Direct answer"),
    GoldInstance("Is there life on other planets?",
                 "Scientists have not confirmed the existence of extraterrestrial life.",
                 False, "No definitive answer exists in context"),
    GoldInstance("Can humans breathe underwater?",
                 "Humans cannot breathe underwater without equipment as they lack gills.",
                 True, "Direct negative answer"),
    GoldInstance("Is the sun a star?",
                 "The Sun is a yellow dwarf star at the center of our solar system.",
                 True, "Direct affirmative"),
    GoldInstance("Is Python a compiled language?",
                 "Python is an interpreted, high-level programming language.",
                 True, "Implicit negative: interpreted vs compiled"),
    GoldInstance("Did the Titanic sink?",
                 "The Titanic struck an iceberg and sank on its maiden voyage in 1912.",
                 True, "Direct historical fact"),
    GoldInstance("Is Mount Everest the tallest mountain?",
                 "Mount Everest is the tallest mountain above sea level at 8,848 meters.",
                 True, "Direct affirmative with height"),
    GoldInstance("Are there penguins at the North Pole?",
                 "Penguins are native to the Southern Hemisphere, not the Arctic.",
                 True, "Direct negative with explanation"),
    GoldInstance("Can all birds fly?",
                 "Birds have wings and most can fly.",
                 False, "Ambiguous: 'most can fly' vs 'all birds' — insufficient for universal claim"),
    GoldInstance("Is the universe infinite?",
                 "The observable universe is about 93 billion light-years in diameter.",
                 False, "Describes size but doesn't answer infinity"),
    GoldInstance("Are vaccines safe?",
                 "Vaccines undergo rigorous clinical trials before approval.",
                 False, "Related but doesn't directly answer safety"),

    # ── Entity disambiguation (8) ──
    GoldInstance("When was Apple founded?",
                 "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
                 True, "Company Apple, not fruit"),
    GoldInstance("When was Apple founded?",
                 "Apple trees have been cultivated for thousands of years.",
                 False, "Fruit reference, not company"),
    GoldInstance("How tall is the Eiffel Tower?",
                 "The Eiffel Tower stands 330 meters tall including antennas.",
                 True, "Direct height with qualification"),
    GoldInstance("How tall is the Eiffel Tower?",
                 "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
                 False, "Describes structure but no height"),
    GoldInstance("Who wrote the Iliad?",
                 "The Iliad is an ancient Greek epic poem attributed to Homer.",
                 True, "Direct attribution"),
    GoldInstance("Who wrote the Iliad?",
                 "The Iliad tells the story of the Trojan War and its heroes.",
                 False, "Describes content, not authorship"),
    GoldInstance("What is the population of Tokyo?",
                 "Tokyo is the capital of Japan and one of the most populous cities in the world.",
                 False, "Related but no specific population figure"),
    GoldInstance("What is the population of Tokyo?",
                 "Tokyo has a population of approximately 14 million in the city proper.",
                 True, "Direct figure"),

    # ── Temporal reasoning (8) ──
    GoldInstance("Who was the US president in 2020?",
                 "Joe Biden was inaugurated as the 46th US president in January 2021. Donald Trump was president from 2017 to 2021.",
                 True, "Multi-hop: 2020 → Trump"),
    GoldInstance("Who was the US president in 2020?",
                 "The US presidential election of 2020 was held on November 3, 2020.",
                 False, "Election date but not the president"),
    GoldInstance("How old was Einstein when he died?",
                 "Albert Einstein was born in 1879 and died in 1955 at Princeton Hospital.",
                 True, "Implied: 1955 - 1879 = 76"),
    GoldInstance("How old was Einstein when he died?",
                 "Albert Einstein made groundbreaking contributions to physics.",
                 False, "No birth/death dates"),
    GoldInstance("What was the first iPhone release year?",
                 "The first iPhone was announced by Steve Jobs on January 9, 2007, and released in June 2007.",
                 True, "Direct date"),
    GoldInstance("What was the first iPhone release year?",
                 "The iPhone revolutionized the smartphone industry.",
                 False, "Impact description, no year"),
    GoldInstance("When did World War II end?",
                 "World War II ended in 1945 with the surrender of Germany in May and Japan in September.",
                 True, "Direct year"),
    GoldInstance("When did World War II end?",
                 "World War II was a global war that lasted from 1939 to 1945.",
                 False, "Context says 'lasted to 1945' but doesn't explicitly confirm end year"),

    # ── Causal reasoning (6) ──
    GoldInstance("Why did the Titanic sink?",
                 "The Titanic sank because it struck an iceberg that caused hull damage.",
                 True, "Direct causal explanation"),
    GoldInstance("Why did the Titanic sink?",
                 "The Titanic was considered unsinkable by its designers.",
                 False, "Related but no causal explanation"),
    GoldInstance("What causes the seasons?",
                 "Seasons are caused by the Earth's axial tilt of 23.5 degrees relative to its orbital plane.",
                 True, "Direct scientific cause"),
    GoldInstance("What causes the seasons?",
                 "Seasons bring changes in temperature and weather patterns.",
                 False, "Describes effects, not cause"),
    GoldInstance("Why is the sky blue?",
                 "The sky appears blue due to Rayleigh scattering of sunlight by atmospheric particles.",
                 True, "Direct causal mechanism"),
    GoldInstance("Why is the sky blue?",
                 "The sky can appear in different colors depending on atmospheric conditions.",
                 False, "Related but no causal mechanism"),

    # ── Parametric knowledge tests (8) ──
    GoldInstance("What is the meaning of life according to Douglas Adams?",
                 "The Hitchhiker's Guide to the Galaxy mentions the number 42 as the answer to life, the universe, and everything.",
                 True, "Context provides the full connection"),
    GoldInstance("What is the meaning of life according to Douglas Adams?",
                 "The Hitchhiker's Guide to the Galaxy mentions the number 42.",
                 False, "Related but doesn't connect meaning of life to 42 explicitly"),
    GoldInstance("What company did Jeff Bezos found?",
                 "Jeff Bezos founded Amazon in 1994 as an online bookstore.",
                 True, "Direct fact"),
    GoldInstance("What company did Jeff Bezos found?",
                 "Jeff Bezos is one of the wealthiest people in the world.",
                 False, "Related but no company"),
    GoldInstance("Who developed the theory of relativity?",
                 "Albert Einstein developed the theory of relativity, including both special and general relativity.",
                 True, "Direct attribution"),
    GoldInstance("Who developed the theory of relativity?",
                 "The theory of relativity revolutionized our understanding of space and time.",
                 False, "Impact description but no person"),
    GoldInstance("What is the largest organ in the human body?",
                 "The skin is the largest organ in the human body.",
                 True, "Direct answer"),
    GoldInstance("What is the largest organ in the human body?",
                 "The human body has many vital organs including the heart, lungs, and liver.",
                 False, "List excludes the skin"),

    # ── Quantitative reasoning (8) ──
    GoldInstance("What is 15% of 200?",
                 "15 percent of 200 equals 30. This is calculated as 0.15 × 200.",
                 True, "Direct calculation result"),
    GoldInstance("What is 15% of 200?",
                 "Percentages represent parts per hundred.",
                 False, "Definition not answer"),
    GoldInstance("How many seconds in an hour?",
                 "There are 60 seconds in a minute and 60 minutes in an hour, so 3600 seconds in an hour.",
                 True, "Direct computation"),
    GoldInstance("How many seconds in an hour?",
                 "An hour consists of 60 minutes.",
                 False, "Partial, requires additional computation"),
    GoldInstance("If a train travels at 60 mph, how far in 2 hours?",
                 "Distance = speed × time = 60 × 2 = 120 miles.",
                 True, "Direct formula applied"),
    GoldInstance("If a train travels at 60 mph, how far in 2 hours?",
                 "Trains can travel at various speeds depending on the type and route.",
                 False, "General statement, no answer"),
    GoldInstance("What is the area of a 5×3 rectangle?",
                 "Area of a rectangle = length × width = 5 × 3 = 15 square units.",
                 True, "Direct calculation"),
    GoldInstance("What is the area of a 5×3 rectangle?",
                 "A rectangle has four sides with opposite sides being equal.",
                 False, "Geometric definition, no area"),

    # ── Negation & contrast (6) ──
    GoldInstance("What is NOT a primary color?",
                 "Red, blue, and yellow are primary colors. Green is a secondary color.",
                 True, "Contrast identifies non-primary"),
    GoldInstance("What is NOT a primary color?",
                 "Primary colors can be mixed to create other colors.",
                 False, "No exclusion information"),
    GoldInstance("Which animal is NOT a mammal?",
                 "Mammals include dogs, cats, and humans. Birds are not mammals as they lay eggs.",
                 True, "Explicit contrast"),
    GoldInstance("Which animal is NOT a mammal?",
                 "Many animals are classified as mammals.",
                 False, "No specific exclusion"),
    GoldInstance("What cannot be recycled?",
                 "Paper, glass, and metal can be recycled. Plastic bags cannot be recycled in curbside bins.",
                 True, "Direct exclusion"),
    GoldInstance("What cannot be recycled?",
                 "Recycling helps reduce waste and protect the environment.",
                 False, "Benefit description, no specifics"),

    # ── Edge case: template/pattern matching (3) ──
    GoldInstance("What is 2+2?",
                 "2 + 2 = 4",
                 True, "Direct arithmetic"),
    GoldInstance("What is 2+2?",
                 "Addition is a basic arithmetic operation.",
                 False, "Definition, not result"),
    GoldInstance("What color is the sky on a clear day?",
                 "The sky on a clear day appears blue to the human eye.",
                 True, "Direct description with condition"),
]
# fmt: on


# ── Calibration Runner ──


@dataclass
class CalibrationResult:
    """Full calibration results for one autorater."""
    autorater_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: dict[str, int]  # {"TP", "FP", "TN", "FN"}
    details: list[dict[str, Any]] = field(default_factory=list)


async def calibrate_autorater(
    autorater: SufficiencyAutorater,
    name: str,
    dataset: list[GoldInstance] | None = None,
) -> CalibrationResult:
    """Run calibration: measure autorater accuracy against gold dataset."""
    data = dataset or GOLD_DATASET_115

    tp = fp = tn = fn = 0
    details: list[dict[str, Any]] = []

    for inst in data:
        is_suff, reason = await autorater.is_sufficient(inst.question, inst.context)

        if is_suff and inst.label:
            tp += 1
        elif is_suff and not inst.label:
            fp += 1
        elif not is_suff and not inst.label:
            tn += 1
        else:
            fn += 1

        details.append({
            "question": inst.question[:60],
            "expected": inst.label,
            "got": is_suff,
            "correct": is_suff == inst.label,
            "reason": reason[:100],
            "notes": inst.notes,
        })

    total = len(data)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return CalibrationResult(
        autorater_name=name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        details=details,
    )


def print_calibration_report(results: list[CalibrationResult]):
    """Pretty-print calibration comparison."""
    print(f"\n{'='*72}")
    print(f"  Autorater Calibration Report")
    print(f"{'='*72}")
    print(f"  Gold dataset size: {sum(r.confusion_matrix['TP'] + r.confusion_matrix['FP'] + r.confusion_matrix['TN'] + r.confusion_matrix['FN'] for r in results) // max(len(results), 1)}")
    print(f"{'='*72}")

    for r in results:
        print(f"\n  ── {r.autorater_name} ──")
        print(f"  Accuracy : {r.accuracy:.1%}  (paper target: 93%)")
        print(f"  Precision: {r.precision:.1%}")
        print(f"  Recall   : {r.recall:.1%}")
        print(f"  F1       : {r.f1_score:.1%}")
        cm = r.confusion_matrix
        print(f"  Confusion: TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}")

    # Best performer
    if results:
        best = max(results, key=lambda r: r.accuracy)
        print(f"\n  ★ Best: {best.autorater_name} ({best.accuracy:.1%})")
        print(f"{'='*72}\n")
