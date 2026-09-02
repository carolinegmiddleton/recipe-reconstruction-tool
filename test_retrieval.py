from recipe_agent import RecipeInput, RecipeReconstructionAgent

agent = RecipeReconstructionAgent(create_client=False)

target = RecipeInput(
    recipe_name="Beef Stroganoff",
    default_portion="6 fl oz",
    calories_kcal=285,
    protein_g=18,
    carbohydrate_g=20,
    fat_g=15,
    sodium_mg=340,
)

print("Top retrievals:")
for ref in agent.retrieve(target):
    print(ref["recipe_name"], round(ref["combined_similarity"], 3))
