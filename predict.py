import argparse
from recipe_agent import RecipeInput, RecipeReconstructionAgent

parser = argparse.ArgumentParser(description="Estimate ingredient proportions for a recipe.")
parser.add_argument("recipe_name")
parser.add_argument("--portion")
parser.add_argument("--calories", type=float)
parser.add_argument("--protein", type=float)
parser.add_argument("--carb", type=float)
parser.add_argument("--fat", type=float)
parser.add_argument("--sodium", type=float)
args = parser.parse_args()

values = [args.calories, args.protein, args.carb, args.fat, args.sodium]
if any(x is not None for x in values) and not all(x is not None for x in values):
    raise SystemExit("Provide all five nutrition values or none of them.")

target = RecipeInput(
    recipe_name=args.recipe_name,
    default_portion=args.portion,
    calories_kcal=args.calories,
    protein_g=args.protein,
    carbohydrate_g=args.carb,
    fat_g=args.fat,
    sodium_mg=args.sodium,
)

agent = RecipeReconstructionAgent()
result = agent.estimate(target, use_exact_match=False)

print(f"\nRecipe: {args.recipe_name}\n")
print("| Ingredient | Estimated % by weight |")
print("|---|---:|")
for x in result["consensus"]:
    print(f"| {x['ingredient']} | {x['estimated_percentage']:.1f} |")
print("| TOTAL | 100.0 |")
print(f"\nMethod: {result['mode']}")
print(f"Model: {result['model']}")
print(f"Audit: {result.get('audit_file', '')}")
