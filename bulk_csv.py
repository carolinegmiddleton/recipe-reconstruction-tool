from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Optional


def parse_top_level_ingredients(text: str) -> list[str]:
    """Split a food-service ingredient string on top-level commas only.

    Nested parentheses stay attached to the parent item, then the
    parenthetical recipe is stripped so the output label is the listed
    common name (e.g. Vegetable Stock, not Water / Vegetable Base).
    Duplicate names are merged, keeping first-seen order.
    """
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []

    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in raw:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        name = _display_name(part)
        if not name:
            continue
        key = re.sub(r"\s+", " ", name).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def _display_name(part: str) -> str:
    cut = None
    for i, ch in enumerate(part):
        if ch == "(":
            cut = i
            break
    name = part[:cut] if cut is not None else part
    return re.sub(r"\s+", " ", name).strip(" .")


def _norm_header(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _pick_column(headers: list[str], candidates: list[str]) -> Optional[str]:
    lookup = {_norm_header(h): h for h in headers}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _parse_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"-", "—", "n/a", "na", "none"}:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _recipe_number(value) -> str:
    text = str(value or "").strip()
    if text.startswith("'"):
        text = text[1:]
    return text.strip()


@dataclass
class BulkRecipe:
    recipe_number: str
    short_name: str
    ingredients: list[str]
    default_portion: Optional[str]
    calories_kcal: Optional[float]
    protein_g: Optional[float]
    carbohydrate_g: Optional[float]
    fat_g: Optional[float]
    sodium_mg: Optional[float]
    source_row_count: int
    error: Optional[str] = None


def parse_bulk_csv(text: str) -> list[BulkRecipe]:
    """Parse a menu CSV into unique recipes (first row wins for nutrition)."""
    sample = text.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(sample))
    if not reader.fieldnames:
        raise ValueError("The CSV has no header row.")

    headers = list(reader.fieldnames)
    col_number = _pick_column(headers, ["recipe number", "recipe_number", "recipe id", "recipe_id"])
    col_name = _pick_column(headers, ["short name", "recipe name", "recipe", "name"])
    col_ing = _pick_column(
        headers,
        ["ingredients common name", "ingredients", "ingredient list", "ingredients list"],
    )
    if not col_name or not col_ing:
        raise ValueError(
            "The CSV must include a recipe name column (e.g. Short Name) "
            "and an ingredients column (e.g. Ingredients Common Name)."
        )

    col_portion = _pick_column(headers, ["menu portion size", "default portion", "portion"])
    col_kcal = _pick_column(headers, ["kcal", "calories", "calories (kcal)", "calories_kcal"])
    col_fat = _pick_column(headers, ["fat (g)", "fat", "fat_g"])
    col_cho = _pick_column(headers, ["cho (g)", "carbohydrate (g)", "carbohydrate", "carb", "carb_g"])
    col_pro = _pick_column(headers, ["pro (g)", "protein (g)", "protein", "protein_g"])
    col_na = _pick_column(headers, ["na (mg)", "sodium (mg)", "sodium", "sodium_mg"])

    unique: dict[str, BulkRecipe] = {}
    order: list[str] = []

    for row in reader:
        name = str(row.get(col_name) or "").strip()
        number = _recipe_number(row.get(col_number) if col_number else "")
        ingredients = parse_top_level_ingredients(row.get(col_ing) or "")
        key = number or f"{name}::{'|'.join(ingredients)}"
        if not key.strip() and not name:
            continue

        if key in unique:
            unique[key].source_row_count += 1
            continue

        error = None
        if not name:
            error = "Missing recipe name."
        elif not ingredients:
            error = "Missing ingredient list."

        unique[key] = BulkRecipe(
            recipe_number=number,
            short_name=name,
            ingredients=ingredients,
            default_portion=(str(row.get(col_portion)).strip() if col_portion and row.get(col_portion) else None) or None,
            calories_kcal=_parse_float(row.get(col_kcal) if col_kcal else None),
            protein_g=_parse_float(row.get(col_pro) if col_pro else None),
            carbohydrate_g=_parse_float(row.get(col_cho) if col_cho else None),
            fat_g=_parse_float(row.get(col_fat) if col_fat else None),
            sodium_mg=_parse_float(row.get(col_na) if col_na else None),
            source_row_count=1,
            error=error,
        )
        order.append(key)

    return [unique[k] for k in order]


def rows_to_csv(rows: list[dict]) -> str:
    fieldnames = ["Recipe Number", "Short Name", "Ingredient", "Estimated % by weight"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()
