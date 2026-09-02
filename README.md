# Recipe Reconstruction Tool v1

This is the simple user-facing wrapper around the validated recipe reconstruction workflow.

## The user experience

Open the app, type:

> Beef Stroganoff

and click **Estimate Recipe**.

The main output is only:

| Ingredient | Estimated % by weight |
|---|---:|
| ... | ... |
| TOTAL | 100.0 |

The research details are saved automatically and are available under **Research audit trail**.

## What happens behind the scenes

1. The recipe name is compared with the 33 validated institutional reference recipes.
2. The three most similar non-target reference recipes are selected.
3. GPT-5.6 Sol is called three independent times.
4. Ingredient names are standardized with the validated crosswalk.
5. Missing ingredients in a replicate are treated as 0%.
6. The three proportions are averaged.
7. The ensemble is normalized to exactly 100%.
8. Stability and CORE status are calculated.
9. A JSON audit file is automatically saved in `outputs/`.

The audit contains:
- date/time
- model version
- method used
- retrieved references
- all three raw LLM outputs
- standardized outputs
- final ensemble
- stability

## Simple mode versus validated high-accuracy mode

### Simple one-box mode

If you enter only a recipe name, the tool uses:

**recipe name → name-based retrieval → 3 reconstructions → consensus**

This is the convenient one-box workflow, but name-only retrieval was not separately validated in the original 297-run experiment.

### Validated high-accuracy mode

Open the optional nutrition panel and provide:
- calories
- protein
- carbohydrate
- fat
- sodium

The tool then uses the validated:

**50% name similarity + 50% nutrition similarity → top 3 references → 3 reconstructions → consensus**

This is the mode most directly supported by the validation study.

## Install

Python 3.10+ is recommended.

Create an environment:

```bash
python -m venv .venv
```

Activate it.

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key.

macOS/Linux:

```bash
export OPENAI_API_KEY="your_key_here"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Run:

```bash
streamlit run app.py
```

## Command-line version

You can also run:

```bash
python predict.py "Beef Stroganoff"
```

or with nutrition:

```bash
python predict.py "Beef Stroganoff" \
  --portion "6 fl oz" \
  --calories 285 \
  --protein 18 \
  --carb 20 \
  --fat 15 \
  --sodium 340
```

## Research warning

The validation study used GPT-5.6 Sol in the ChatGPT product. This wrapper uses the OpenAI API. Before treating the deployed tool as having exactly the same measured performance, rerun a frozen validation sample through this implementation.
