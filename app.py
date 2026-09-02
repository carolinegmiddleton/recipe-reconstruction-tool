import json
import streamlit as st

from bulk_csv import parse_bulk_csv, rows_to_csv
from recipe_agent import RecipeInput, RecipeReconstructionAgent

st.set_page_config(
    page_title="Recipe Reconstruction Tool",
    page_icon="🍽️",
    layout="wide",
)

st.title("Recipe Reconstruction Tool")

single_tab, bulk_tab = st.tabs(["Single recipe", "Bulk CSV"])


def render_recipe_table(recipe_name: str, result: dict) -> None:
    final = result["consensus"]
    display_rows = [
        {
            "Ingredient": x["ingredient"],
            "Estimated % by weight": round(x["estimated_percentage"], 1),
        }
        for x in final
        if x["estimated_percentage"] > 0
    ]

    st.subheader(recipe_name)
    st.dataframe(
        display_rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ingredient": st.column_config.TextColumn("Ingredient"),
            "Estimated % by weight": st.column_config.NumberColumn(
                "Estimated % by weight",
                format="%.1f",
            ),
        },
    )
    st.markdown(f"**TOTAL: {sum(x['estimated_percentage'] for x in final):.1f}%**")

    with st.expander("Research audit trail"):
        st.write(f"**Date/time:** {result.get('audit_file', '').split('/')[-1].split('_')[0] or 'saved automatically'}")
        st.write(f"**Model:** {result.get('model')}")
        st.write(f"**Method:** {result.get('mode')}")
        st.write("**Retrieved reference recipes:**")
        if result.get("retrieved_references"):
            st.dataframe(result["retrieved_references"], hide_index=True, use_container_width=True)

        st.write("**Three raw LLM outputs:**")
        for i, raw in enumerate(result.get("raw_runs", []), 1):
            st.markdown(f"Run {i}")
            st.json(raw)

        st.write("**Standardized outputs:**")
        for i, standardized in enumerate(result.get("standardized_runs", []), 1):
            st.markdown(f"Run {i}")
            st.json(standardized)

        st.write("**Final ensemble + stability:**")
        audit_rows = [
            {
                "Ingredient": x["ingredient"],
                "Final %": round(x["estimated_percentage"], 2),
                "Runs present": x.get("runs_present"),
                "Stability": x.get("stability"),
                "CORE": x.get("core"),
                "Run 1 %": None if x.get("replicate_percentages") is None else x["replicate_percentages"][0],
                "Run 2 %": None if x.get("replicate_percentages") is None else (
                    x["replicate_percentages"][1] if len(x["replicate_percentages"]) > 1 else None
                ),
                "Run 3 %": None if x.get("replicate_percentages") is None else (
                    x["replicate_percentages"][2] if len(x["replicate_percentages"]) > 2 else None
                ),
            }
            for x in final
        ]
        st.dataframe(audit_rows, hide_index=True, use_container_width=True)

        st.download_button(
            "Download audit JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="recipe_reconstruction_audit.json",
            mime="application/json",
            use_container_width=True,
        )

    if result.get("mode", "").startswith("name-only fallback"):
        st.caption(
            "This estimate used the name-only retrieval fallback. "
            "Providing nutrition information activates the better-validated retrieval mode."
        )


with single_tab:
    st.write("Enter a recipe name and estimate its ingredient proportions.")

    recipe_name = st.text_input(
        "Recipe name",
        placeholder="Beef Stroganoff",
        label_visibility="collapsed",
    )

    with st.expander("Optional: add nutrition information for the validated high-accuracy mode"):
        st.caption(
            "If all five nutrition values are supplied, the tool uses the validated "
            "name + nutrition + retrieval workflow. If they are blank, it uses the "
            "name-only retrieval fallback."
        )
        portion = st.text_input("Default portion", placeholder="e.g., 6 fl oz ladle")
        c1, c2 = st.columns(2)
        with c1:
            kcal = st.number_input("Calories (kcal)", min_value=0.0, value=None, step=1.0)
            carb = st.number_input("Carbohydrate (g)", min_value=0.0, value=None, step=0.1)
            sodium = st.number_input("Sodium (mg)", min_value=0.0, value=None, step=1.0)
        with c2:
            protein = st.number_input("Protein (g)", min_value=0.0, value=None, step=0.1)
            fat = st.number_input("Fat (g)", min_value=0.0, value=None, step=0.1)

    estimate = st.button(
        "Estimate Recipe",
        type="primary",
        use_container_width=True,
        disabled=not bool(recipe_name.strip()),
        key="single_estimate",
    )

    if estimate:
        nutrition_values = [kcal, protein, carb, fat, sodium]
        supplied = [x is not None for x in nutrition_values]

        if any(supplied) and not all(supplied):
            st.error(
                "Either leave all nutrition fields blank or provide all five: "
                "calories, protein, carbohydrate, fat, and sodium."
            )
            st.stop()

        target = RecipeInput(
            recipe_name=recipe_name.strip(),
            default_portion=portion.strip() or None,
            calories_kcal=kcal,
            protein_g=protein,
            carbohydrate_g=carb,
            fat_g=fat,
            sodium_mg=sodium,
        )

        with st.spinner("Estimating recipe..."):
            try:
                agent = RecipeReconstructionAgent()
                result = agent.estimate(target, use_exact_match=False)
            except Exception as exc:
                st.error(f"Could not run the recipe estimator: {exc}")
                st.stop()

        render_recipe_table(recipe_name.strip(), result)


with bulk_tab:
    st.write(
        "Upload or paste a menu CSV of recipe names and ingredient lists. "
        "The tool estimates % by weight for the listed ingredients only."
    )
    st.caption(
        "Expected columns include Short Name, Ingredients Common Name, and optionally "
        "Recipe Number plus KCAL, PRO (g), CHO (g), FAT (g), and Na (mg). "
        "Nested items such as Vegetable Stock (Water, Vegetable Base) stay as one ingredient. "
        "Duplicate recipe numbers are estimated once."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    pasted = st.text_area("Or paste CSV text", height=160, placeholder="Week,Day of Week/Date,...")

    higher_accuracy = st.checkbox(
        "Higher accuracy (3 model runs per recipe)",
        value=False,
        help="Off uses 1 model run per recipe, which is much faster for large files.",
    )

    csv_text = ""
    if uploaded is not None:
        csv_text = uploaded.getvalue().decode("utf-8-sig")
    elif pasted.strip():
        csv_text = pasted

    recipes = []
    parse_error = None
    if csv_text.strip():
        try:
            recipes = parse_bulk_csv(csv_text)
        except Exception as exc:
            parse_error = str(exc)

    if parse_error:
        st.error(f"Could not parse the CSV: {parse_error}")
    elif recipes:
        valid = [r for r in recipes if not r.error]
        skipped = [r for r in recipes if r.error]
        st.success(
            f"Parsed {len(recipes)} unique recipes "
            f"({len(valid)} ready, {len(skipped)} skipped)."
        )
        preview = [
            {
                "Recipe Number": r.recipe_number,
                "Short Name": r.short_name,
                "Ingredients": ", ".join(r.ingredients),
                "Nutrition": "yes" if all(
                    x is not None
                    for x in [r.calories_kcal, r.protein_g, r.carbohydrate_g, r.fat_g, r.sodium_mg]
                ) else "no",
                "Rows in file": r.source_row_count,
                "Note": r.error or "",
            }
            for r in recipes[:25]
        ]
        st.dataframe(preview, hide_index=True, use_container_width=True)
        if skipped:
            st.warning("Recipes with missing names or ingredient lists will be skipped.")

        run_bulk = st.button(
            "Estimate all recipes",
            type="primary",
            use_container_width=True,
            disabled=not valid,
            key="bulk_estimate",
        )

        if run_bulk:
            n_replicates = 3 if higher_accuracy else 1
            output_rows = []
            errors = []
            progress = st.progress(0.0, text="Starting bulk estimate...")
            try:
                agent = RecipeReconstructionAgent()
            except Exception as exc:
                st.error(f"Could not start the recipe estimator: {exc}")
                st.stop()

            for i, recipe in enumerate(valid):
                progress.progress(
                    i / max(len(valid), 1),
                    text=f"Estimating {i + 1} of {len(valid)}: {recipe.short_name}",
                )
                target = RecipeInput(
                    recipe_name=recipe.short_name,
                    default_portion=recipe.default_portion,
                    calories_kcal=recipe.calories_kcal,
                    protein_g=recipe.protein_g,
                    carbohydrate_g=recipe.carbohydrate_g,
                    fat_g=recipe.fat_g,
                    sodium_mg=recipe.sodium_mg,
                    known_ingredients=recipe.ingredients,
                )
                try:
                    result = agent.estimate(
                        target,
                        use_exact_match=False,
                        n_replicates=n_replicates,
                        write_audit=False,
                    )
                    for item in result["consensus"]:
                        output_rows.append(
                            {
                                "Recipe Number": recipe.recipe_number,
                                "Short Name": recipe.short_name,
                                "Ingredient": item["ingredient"],
                                "Estimated % by weight": round(item["estimated_percentage"], 1),
                            }
                        )
                except Exception as exc:
                    errors.append(
                        {
                            "Recipe Number": recipe.recipe_number,
                            "Short Name": recipe.short_name,
                            "Error": str(exc),
                        }
                    )

            progress.progress(1.0, text="Finished.")
            st.session_state["bulk_output_rows"] = output_rows
            st.session_state["bulk_errors"] = errors

    if st.session_state.get("bulk_output_rows") is not None:
        output_rows = st.session_state["bulk_output_rows"]
        errors = st.session_state.get("bulk_errors") or []
        if output_rows:
            st.subheader("Estimated proportions")
            st.dataframe(output_rows, hide_index=True, use_container_width=True)
            st.download_button(
                "Download results CSV",
                data=rows_to_csv(output_rows),
                file_name="recipe_proportions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if errors:
            st.error(f"{len(errors)} recipes failed.")
            st.dataframe(errors, hide_index=True, use_container_width=True)
        elif output_rows:
            st.success(f"Estimated {len({r['Short Name'] for r in output_rows})} recipes.")
