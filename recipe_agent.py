from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
LIBRARY_PATH = BASE_DIR / "reference_library.json"
ALIASES_PATH = BASE_DIR / "ingredient_aliases.json"
RUNS_DIR = BASE_DIR / "runs"

MODEL = os.getenv("RECIPE_AGENT_MODEL", "gpt-5.6-sol")
N_REPLICATES = 3
K_REFERENCES = 3


@dataclass
class RecipeInput:
    recipe_name: str
    default_portion: Optional[str] = None
    calories_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    carbohydrate_g: Optional[float] = None
    fat_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    known_ingredients: Optional[list[str]] = None

    @property
    def has_complete_nutrition(self) -> bool:
        return all(
            x is not None
            for x in [
                self.calories_kcal,
                self.protein_g,
                self.carbohydrate_g,
                self.fat_g,
                self.sodium_mg,
            ]
        )


class RecipeReconstructionAgent:
    def __init__(self, model: str = MODEL, create_client: bool = True):
        self.model = model
        self.client = OpenAI() if create_client else None
        self.library = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
        self.aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
        RUNS_DIR.mkdir(exist_ok=True)

        self._names = [r["recipe_name"] for r in self.library]
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w&]+\b",
        )
        self._name_matrix = self._tfidf.fit_transform(self._names)

        self._nutrition_matrix = np.array(
            [
                [
                    r["calories_kcal"],
                    r["protein_g"],
                    r["carbohydrate_g"],
                    r["fat_g"],
                    r["sodium_mg"],
                ]
                for r in self.library
            ],
            dtype=float,
        )
        self._scaler = StandardScaler().fit(self._nutrition_matrix)
        self._nutrition_z = self._scaler.transform(self._nutrition_matrix)

    @staticmethod
    def _clean_label(label: str) -> str:
        s = re.sub(r"[*_`]+", "", str(label)).strip().lower().replace("&", " and ")
        s = re.sub(r"[’']", "", s)
        s = re.sub(r"[^a-z0-9%]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def standardize_ingredient(self, label: str) -> str:
        raw = str(label).strip()
        exact = self.aliases.get(raw.lower())
        if exact:
            return exact

        cleaned = self._clean_label(raw)
        for alias, standard in self.aliases.items():
            if self._clean_label(alias) == cleaned:
                return standard

        # Unknown labels stay as their own generic concept.
        return re.sub(r"\s+", " ", raw).strip()

    def exact_match(self, target: RecipeInput):
        t = target.recipe_name.strip().lower()
        for recipe in self.library:
            if recipe["recipe_name"].strip().lower() == t:
                return recipe
        return None

    def retrieve(self, target: RecipeInput, k: int = K_REFERENCES):
        target_name_vec = self._tfidf.transform([target.recipe_name])
        name_scores = cosine_similarity(target_name_vec, self._name_matrix)[0]

        if target.has_complete_nutrition:
            target_n = np.array(
                [[
                    target.calories_kcal,
                    target.protein_g,
                    target.carbohydrate_g,
                    target.fat_g,
                    target.sodium_mg,
                ]],
                dtype=float,
            )
            target_z = self._scaler.transform(target_n)[0]
            distances = np.sqrt(((self._nutrition_z - target_z) ** 2).sum(axis=1))
            nutrition_scores = 1 / (1 + distances)
            combined = 0.50 * name_scores + 0.50 * nutrition_scores
            mode = "name+nutrition"
        else:
            nutrition_scores = np.full(len(self.library), np.nan)
            combined = name_scores
            mode = "name-only fallback"

        refs = []
        for i in np.argsort(-combined):
            if self.library[i]["recipe_name"].strip().lower() == target.recipe_name.strip().lower():
                continue
            item = dict(self.library[i])
            item["name_similarity"] = float(name_scores[i])
            item["nutrition_similarity"] = (
                None if math.isnan(float(nutrition_scores[i]))
                else float(nutrition_scores[i])
            )
            item["combined_similarity"] = float(combined[i])
            item["retrieval_mode"] = mode
            refs.append(item)
            if len(refs) == k:
                break
        return refs

    @staticmethod
    def _reference_block(ref: dict, rank: int) -> str:
        lines = [
            f"Reference recipe {rank}: {ref['recipe_name']}",
            "",
            "| Ingredient | % by weight |",
            "|---|---:|",
        ]
        for x in ref["ingredients"]:
            lines.append(f"| {x['ingredient']} | {x['percentage']:.2f} |")
        return "\n".join(lines)

    def build_prompt(self, target: RecipeInput, refs: list[dict]) -> str:
        info = []
        if target.default_portion:
            info.append(f"Default portion: {target.default_portion}")
        if target.has_complete_nutrition:
            info.extend([
                f"Calories: {target.calories_kcal:.0f} kcal",
                f"Protein: {target.protein_g:.2f} g",
                f"Carbohydrate: {target.carbohydrate_g:.2f} g",
                f"Fat: {target.fat_g:.2f} g",
                f"Sodium: {target.sodium_mg:.0f} mg",
            ])

        refs_text = "\n\n".join(
            self._reference_block(ref, i + 1)
            for i, ref in enumerate(refs)
        )

        known = target.known_ingredients or []
        if known:
            listed = "\n".join(f"- {name}" for name in known)
            task = f"""Your task is to estimate the percentage by weight of each listed ingredient in an institutional food-service recipe when the true formula is unavailable.

Use ONLY:
1. The target recipe name and any portion/nutrition information provided below.
2. The closed ingredient list provided below.
3. The retrieved reference recipes provided below as context for typical proportions.

Do not browse the web, search for the target recipe, use external sources, or ask for additional information.

The retrieved recipes are examples of other institutional food-service recipes. They are NOT the target recipe. Use them only as contextual reference.

Requirements:
- Use EVERY ingredient from the closed list exactly once.
- Do not add ingredients that are not on the closed list.
- Do not rename ingredients.
- Major components (vegetables, proteins, grains, liquids) should receive most of the weight.
- Seasonings, spices, garlic, pepper, and similar items are typically small percentages.
- Percentages should sum to 100%.
- Do not provide ranges, alternatives, explanations, or confidence scores.

CLOSED INGREDIENT LIST

{listed}
"""
        else:
            task = """Your task is to estimate the ingredient composition of an institutional food-service recipe when the true recipe is unavailable.

Use ONLY:
1. The target recipe name and any portion/nutrition information provided below.
2. The retrieved reference recipes provided below.

Do not browse the web, search for the target recipe, use external sources, or ask for additional information.

The retrieved recipes are examples of other institutional food-service recipes. They are NOT the target recipe. Use them only as contextual reference.

Estimate the individual ingredients most likely present and the approximate percentage by weight of each ingredient.

Requirements:
- Use generic ingredient names rather than brand names.
- List ingredients individually whenever possible.
- Include only ingredients you reasonably expect to be present.
- Percentages should sum to 100%.
- Use no more than 20 ingredients.
- Do not provide ranges, alternatives, explanations, or confidence scores.
"""

        return f"""I am conducting a research study evaluating AI-based recipe reconstruction.

{task}
TARGET RECIPE

Recipe name: {target.recipe_name}
{chr(10).join(info)}

RETRIEVED REFERENCE RECIPES

{refs_text}
"""

    def _one_model_run(self, prompt: str, known_ingredients: Optional[list[str]] = None) -> dict:
        if known_ingredients:
            names = list(known_ingredients)
            n = len(names)
            ingredient_schema: dict = {
                "type": "string",
                "enum": names,
            }
            schema = {
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "minItems": n,
                        "maxItems": n,
                        "items": {
                            "type": "object",
                            "properties": {
                                "ingredient": ingredient_schema,
                                "percentage": {"type": "number", "minimum": 0, "maximum": 100},
                            },
                            "required": ["ingredient", "percentage"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["ingredients"],
                "additionalProperties": False,
            }
        else:
            schema = {
                "type": "object",
                "properties": {
                    "ingredients": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "ingredient": {"type": "string"},
                                "percentage": {"type": "number", "minimum": 0, "maximum": 100},
                            },
                            "required": ["ingredient", "percentage"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["ingredients"],
                "additionalProperties": False,
            }

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "recipe_reconstruction",
                    "description": "Predicted ingredients and percentage by weight.",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        return json.loads(response.output_text)

    def _match_known_ingredient(self, label: str, known: list[str]) -> Optional[str]:
        cleaned = self._clean_label(label)
        std = self._clean_label(self.standardize_ingredient(label))
        for name in known:
            if name.lower() == str(label).strip().lower():
                return name
            if self._clean_label(name) == cleaned:
                return name
            if self._clean_label(self.standardize_ingredient(name)) == std:
                return name
        return None

    def _align_to_known(self, values: dict[str, float], known: list[str]) -> dict[str, float]:
        allocated = {name: 0.0 for name in known}
        for label, pct in values.items():
            match = self._match_known_ingredient(label, known)
            if match:
                allocated[match] += float(pct)
        normalized = self._normalize(allocated)
        return {name: normalized.get(name, 0.0) for name in known}

    def _standardize_run(
        self,
        raw_run: dict,
        known_ingredients: Optional[list[str]] = None,
    ) -> dict[str, float]:
        out = {}
        for item in raw_run["ingredients"]:
            raw_label = item["ingredient"]
            if known_ingredients:
                label = self._match_known_ingredient(raw_label, known_ingredients) or raw_label
            else:
                label = self.standardize_ingredient(raw_label)
            pct = float(item["percentage"])
            out[label] = out.get(label, 0.0) + pct
        if known_ingredients:
            return self._align_to_known(out, known_ingredients)
        return out

    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        total = sum(values.values())
        if total <= 0:
            return {}
        return {k: 100.0 * v / total for k, v in values.items()}

    def consensus(self, standardized_runs: list[dict[str, float]]) -> list[dict]:
        concepts = sorted(set().union(*(r.keys() for r in standardized_runs)))
        rows = []
        mean_values = {}

        for concept in concepts:
            vals = [run.get(concept, 0.0) for run in standardized_runs]
            mean = sum(vals) / len(standardized_runs)
            present = sum(v > 0 for v in vals)
            mean_values[concept] = mean
            rows.append({
                "ingredient": concept,
                "runs_present": present,
                "appearance_rate": present / len(standardized_runs),
                "replicate_percentages": vals,
                "mean_percentage_before_normalization": mean,
                "sd_percentage": float(np.std(vals)),
                "core": present >= 2,
                "stability": (
                    "High" if present == 3
                    else "Moderate" if present == 2
                    else "Low"
                ),
            })

        normalized = self._normalize(mean_values)
        for row in rows:
            row["estimated_percentage"] = normalized.get(row["ingredient"], 0.0)

        rows.sort(key=lambda x: -x["estimated_percentage"])
        return rows

    def estimate(
        self,
        target: RecipeInput,
        use_exact_match: bool = True,
        n_replicates: Optional[int] = None,
        write_audit: bool = True,
    ) -> dict:
        known = [x.strip() for x in (target.known_ingredients or []) if str(x).strip()]
        if known:
            known = list(dict.fromkeys(known))
            target.known_ingredients = known

        if len(known) == 1:
            return {
                "mode": "single-ingredient",
                "model": None,
                "input": asdict(target),
                "retrieved_references": [],
                "raw_runs": [],
                "standardized_runs": [],
                "consensus": [
                    {
                        "ingredient": known[0],
                        "estimated_percentage": 100.0,
                        "runs_present": None,
                        "appearance_rate": None,
                        "replicate_percentages": None,
                        "sd_percentage": None,
                        "core": True,
                        "stability": "Single ingredient",
                    }
                ],
            }

        exact = self.exact_match(target)
        if exact and use_exact_match and not known:
            return {
                "mode": "exact-reference",
                "model": None,
                "input": asdict(target),
                "retrieved_references": [],
                "raw_runs": [],
                "standardized_runs": [],
                "consensus": [
                    {
                        "ingredient": x["ingredient"],
                        "estimated_percentage": x["percentage"],
                        "runs_present": None,
                        "appearance_rate": None,
                        "replicate_percentages": None,
                        "sd_percentage": None,
                        "core": True,
                        "stability": "Known recipe",
                    }
                    for x in exact["ingredients"]
                ],
            }

        refs = self.retrieve(target)
        prompt = self.build_prompt(target, refs)
        replicates = N_REPLICATES if n_replicates is None else max(1, int(n_replicates))

        raw_runs = [
            self._one_model_run(prompt, known_ingredients=known or None)
            for _ in range(replicates)
        ]
        standardized_runs = [
            self._standardize_run(x, known_ingredients=known or None)
            for x in raw_runs
        ]
        consensus = self.consensus(standardized_runs)
        if known:
            aligned = self._align_to_known(
                {row["ingredient"]: row["estimated_percentage"] for row in consensus},
                known,
            )
            by_name = {row["ingredient"]: row for row in consensus}
            consensus = []
            for name in known:
                src = by_name.get(name, {})
                consensus.append({
                    "ingredient": name,
                    "estimated_percentage": aligned.get(name, 0.0),
                    "runs_present": src.get("runs_present"),
                    "appearance_rate": src.get("appearance_rate"),
                    "replicate_percentages": src.get("replicate_percentages"),
                    "sd_percentage": src.get("sd_percentage"),
                    "core": src.get("core", True),
                    "stability": src.get("stability", "Closed list"),
                })

        result = {
            "mode": refs[0]["retrieval_mode"] if refs else "unknown",
            "model": self.model,
            "input": asdict(target),
            "retrieved_references": [
                {
                    "recipe_id": x["recipe_id"],
                    "recipe_name": x["recipe_name"],
                    "combined_similarity": x["combined_similarity"],
                    "name_similarity": x["name_similarity"],
                    "nutrition_similarity": x["nutrition_similarity"],
                }
                for x in refs
            ],
            "raw_runs": raw_runs,
            "standardized_runs": standardized_runs,
            "consensus": consensus,
        }
        if known:
            result["mode"] = f"{result['mode']} + closed ingredient list"

        if write_audit:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", target.recipe_name).strip("_")[:80]
            audit_path = RUNS_DIR / f"{timestamp}_{safe_name}.json"
            audit_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            result["audit_file"] = str(audit_path)
        return result
