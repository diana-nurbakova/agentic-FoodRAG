"""Mock data for Flow A (Interactive Exploration).

Ported verbatim from the v4 React mockup so the UI is fully clickable
end-to-end before the real RAG backend is wired in.
Replace these structures with calls into ``foodrag`` services later; the
component layer only depends on the shapes defined here.
"""

from __future__ import annotations

MEALS = ["Breakfast", "Lunch", "Dinner", "Snack"]

# ── Recipe sets, keyed by plan ────────────────────────────────────────────
RECIPES_SETS = {
    "balanced": [
        {"name": "Grilled Salmon & Quinoa Bowl", "time": "25min", "nutriscore": "A", "who": 8, "fsa": 2,
         "cal": 420, "protein": 38, "carbs": 32, "fat": 14, "fiber": 6, "sugar": 4, "sodium": 380,
         "img": "🐟", "skill": "Easy", "servings": 2, "tags": ["High-protein", "Omega-3", "Gluten-free"],
         "ingredients": ["200g salmon fillet", "100g quinoa", "1 avocado", "100g cherry tomatoes",
                         "50g spinach", "1 tbsp olive oil", "Lemon juice", "Salt & pepper"],
         "steps": ["Rinse quinoa, cook 15 min.", "Season salmon, grill 4 min/side.",
                   "Assemble bowls.", "Drizzle lemon, serve."],
         "substitutions": [{"orig": "200g salmon fillet", "sub": "200g trout fillet",
                            "reason": "Lower mercury, similar omega-3 profile"}]},
        {"name": "Mediterranean Chickpea Salad", "time": "15min", "nutriscore": "A", "who": 9, "fsa": 1,
         "cal": 310, "protein": 14, "carbs": 42, "fat": 12, "fiber": 9, "sugar": 6, "sodium": 290,
         "img": "🥗", "skill": "Easy", "servings": 2, "tags": ["Vegan", "High-fiber", "Heart-healthy"],
         "ingredients": ["400g chickpeas", "1 cucumber", "200g tomatoes", "1 red onion", "Olives",
                         "2 tbsp olive oil", "Red wine vinegar"],
         "steps": ["Drain chickpeas.", "Chop vegs.", "Combine.", "Dress and serve."],
         "substitutions": []},
        {"name": "Turkey & Vegetable Stir-fry", "time": "20min", "nutriscore": "B", "who": 7, "fsa": 3,
         "cal": 380, "protein": 32, "carbs": 28, "fat": 16, "fiber": 5, "sugar": 7, "sodium": 520,
         "img": "🍳", "skill": "Medium", "servings": 2, "tags": ["Low-carb", "Lean protein"],
         "ingredients": ["250g turkey breast", "1 bell pepper", "1 zucchini", "100g broccoli",
                         "2 tbsp low-sodium soy sauce", "Sesame oil", "Ginger"],
         "steps": ["Slice turkey and veg.", "Stir-fry turkey 4 min.", "Add veg 5 min.",
                   "Combine, add soy sauce."],
         "substitutions": [{"orig": "2 tbsp low-sodium soy sauce", "sub": "2 tbsp coconut aminos",
                            "reason": "Lower sodium, diabetic-friendly"}]},
        {"name": "Greek Yogurt & Berry Parfait", "time": "5min", "nutriscore": "A", "who": 7, "fsa": 2,
         "cal": 180, "protein": 12, "carbs": 24, "fat": 4, "fiber": 3, "sugar": 14, "sodium": 60,
         "img": "🥣", "skill": "Easy", "servings": 1, "tags": ["Quick", "No-cook", "Snack"],
         "ingredients": ["150g Greek yogurt", "Mixed berries", "1 tsp honey", "Chia seeds"],
         "steps": ["Layer yogurt.", "Add berries.", "Drizzle honey, top chia."],
         "substitutions": [{"orig": "1 tsp honey", "sub": "Stevia drops",
                            "reason": "Zero glycemic impact for diabetes management"}]},
    ],
    "highProtein": [
        {"name": "Chicken Breast & Sweet Potato", "time": "30min", "nutriscore": "A", "who": 8, "fsa": 2,
         "cal": 480, "protein": 45, "carbs": 38, "fat": 12, "fiber": 5, "sugar": 8, "sodium": 350,
         "img": "🍗", "skill": "Easy", "servings": 2, "tags": ["High-protein", "Muscle gain"],
         "ingredients": ["300g chicken breast", "2 sweet potatoes", "Broccoli", "Olive oil",
                         "Garlic", "Paprika"],
         "steps": ["Roast sweet potato 20 min.", "Pan-sear chicken 6 min/side.", "Steam broccoli.", "Plate."],
         "substitutions": []},
        {"name": "Lentil & Egg Power Bowl", "time": "20min", "nutriscore": "A", "who": 9, "fsa": 1,
         "cal": 410, "protein": 28, "carbs": 45, "fat": 14, "fiber": 12, "sugar": 3, "sodium": 280,
         "img": "🥚", "skill": "Easy", "servings": 2, "tags": ["Vegetarian", "Iron-rich"],
         "ingredients": ["200g green lentils", "2 eggs", "Spinach", "Cherry tomatoes", "Feta",
                         "Lemon", "Cumin"],
         "steps": ["Cook lentils 15 min.", "Soft-boil eggs.", "Assemble.", "Top with feta."],
         "substitutions": []},
        {"name": "Tuna Steak & Quinoa", "time": "20min", "nutriscore": "A", "who": 8, "fsa": 2,
         "cal": 450, "protein": 42, "carbs": 30, "fat": 18, "fiber": 4, "sugar": 2, "sodium": 400,
         "img": "🐟", "skill": "Medium", "servings": 2, "tags": ["Omega-3", "Low-sugar"],
         "ingredients": ["2 tuna steaks", "100g quinoa", "Edamame", "Avocado", "Soy sauce", "Sesame seeds"],
         "steps": ["Cook quinoa.", "Sear tuna 2 min/side.", "Plate with edamame."],
         "substitutions": []},
        {"name": "Protein Smoothie Bowl", "time": "5min", "nutriscore": "B", "who": 6, "fsa": 3,
         "cal": 290, "protein": 24, "carbs": 30, "fat": 8, "fiber": 4, "sugar": 16, "sodium": 80,
         "img": "🥤", "skill": "Easy", "servings": 1, "tags": ["Quick", "Post-workout"],
         "ingredients": ["1 scoop whey protein", "1 banana", "100ml almond milk", "Peanut butter", "Granola"],
         "steps": ["Blend protein, banana, milk.", "Pour in bowl.", "Top with granola and peanut butter."],
         "substitutions": []},
    ],
    "quickEasy": [
        {"name": "Avocado Toast & Eggs", "time": "10min", "nutriscore": "B", "who": 7, "fsa": 3,
         "cal": 350, "protein": 16, "carbs": 28, "fat": 20, "fiber": 7, "sugar": 2, "sodium": 320,
         "img": "🥑", "skill": "Easy", "servings": 1, "tags": ["Quick", "Vegetarian"],
         "ingredients": ["2 eggs", "1 avocado", "Whole grain bread", "Chili flakes", "Lime"],
         "steps": ["Toast bread.", "Mash avocado.", "Fry eggs.", "Assemble."],
         "substitutions": []},
        {"name": "Caprese Wrap", "time": "8min", "nutriscore": "B", "who": 6, "fsa": 4,
         "cal": 320, "protein": 14, "carbs": 30, "fat": 16, "fiber": 3, "sugar": 5, "sodium": 450,
         "img": "🌯", "skill": "Easy", "servings": 1, "tags": ["No-cook", "Italian"],
         "ingredients": ["Whole wheat wrap", "Fresh mozzarella", "Tomato", "Basil", "Balsamic glaze"],
         "steps": ["Layer mozzarella and tomato.", "Add basil.", "Drizzle balsamic.", "Roll."],
         "substitutions": []},
        {"name": "Veggie Omelette", "time": "10min", "nutriscore": "A", "who": 8, "fsa": 2,
         "cal": 280, "protein": 20, "carbs": 8, "fat": 18, "fiber": 3, "sugar": 3, "sodium": 380,
         "img": "🍳", "skill": "Easy", "servings": 1, "tags": ["Low-carb", "Quick"],
         "ingredients": ["3 eggs", "Bell pepper", "Mushrooms", "Spinach", "Cheese", "Salt & pepper"],
         "steps": ["Whisk eggs.", "Sauté vegs.", "Pour eggs, cook 3 min.", "Fold and serve."],
         "substitutions": []},
        {"name": "Apple & Nut Butter", "time": "3min", "nutriscore": "A", "who": 7, "fsa": 2,
         "cal": 200, "protein": 6, "carbs": 28, "fat": 10, "fiber": 4, "sugar": 18, "sodium": 40,
         "img": "🍎", "skill": "Easy", "servings": 1, "tags": ["No-cook", "Snack"],
         "ingredients": ["1 apple", "2 tbsp almond butter", "Cinnamon"],
         "steps": ["Slice apple.", "Serve with nut butter.", "Sprinkle cinnamon."],
         "substitutions": []},
    ],
}

PLAN_META = {
    "balanced": {"label": "Balanced", "icon": "⚖️",
                 "summary": {"cal": "1,290", "ns": "A", "prot": "96g", "fiber": "23g", "skill": "Easy–Med"}},
    "highProtein": {"label": "High-Protein", "icon": "💪",
                    "summary": {"cal": "1,630", "ns": "A", "prot": "139g", "fiber": "25g", "skill": "Easy–Med"}},
    "quickEasy": {"label": "Quick & Easy", "icon": "⚡",
                  "summary": {"cal": "1,150", "ns": "A–B", "prot": "56g", "fiber": "17g", "skill": "Easy"}},
}

CONSTRAINTS = [
    {"label": "Nutriscore ≤ B", "type": "nutriscore", "removable": True},
    {"label": "No Gluten", "type": "allergy", "removable": True},
    {"label": "Calories ≤ 500", "type": "nutrition", "removable": True},
    {"label": "Sugar ≤ 10g", "type": "nutrition", "removable": True},
    {"label": "Diabetes-friendly", "type": "health", "removable": False},
]

MOCK_QUERY = """df_filtered = df[
  (df['nutriscore'].isin(['A', 'B'])) &
  (df['allergens'].apply(lambda x: 'gluten' not in x)) &
  (df['calories'] <= 500) & (df['sugar_g'] <= 10)
].sort_values('health_score', ascending=False)"""

MOCK_SPLITS = [
    {"meal": "Breakfast", "q": "Suggest a gluten-free, low-sugar breakfast under 350 kcal suitable for Type 2 Diabetes"},
    {"meal": "Lunch", "q": "Recommend a heart-healthy, high-fiber lunch without gluten, max 500 kcal"},
    {"meal": "Dinner", "q": "Find a protein-rich dinner with Nutriscore A or B, no gluten, under 500 kcal"},
    {"meal": "Snack", "q": "Suggest a diabetes-friendly snack under 200 kcal, no gluten"},
]

# NB: illustrative strings only — nothing here is produced by the live pipeline.
# Keep the stated mechanisms truthful anyway: these surface in demos and screenshots.
# FoodOn expansion is rdfs:subClassOf only, so it resolves "wheat food product" →
# "wheat flour"; it does NOT derive gluten from wheat (docs/ARCHITECTURE.md §2.3).
CORRECTION_LOG = [
    {"meal": "Dinner", "round": 1, "status": "INVALID",
     "reason": "Recipe contained wheat flour (blacklisted ingredient detected)",
     "action": "Constraint relaxation level 1: removed tag preference 'Italian', kept allergen constraint"},
    {"meal": "Dinner", "round": 2, "status": "OK", "reason": "Replacement recipe passes all constraints", "action": ""},
    {"meal": "Snack", "round": 1, "status": "OK", "reason": "All constraints satisfied", "action": ""},
]

HEXAD_Q = [
    "I like to compete with others", "I enjoy helping others", "I like exploring new things",
    "I enjoy being part of a team", "I like completing collections", "I like to follow my own path",
    "Rewards motivate me", "Meaningful activities matter", "I try new approaches",
    "I prefer to interact with others", "I strive to complete every task", "I challenge the status quo",
]

PERSONAS = [
    {"id": 1, "name": "Marie", "age": 45, "emoji": "👩‍⚕️", "condition": "Type 2 Diabetes",
     "allergies": ["Gluten"], "likedTags": ["Mediterranean"], "dislikedTags": ["Fried", "Fast food"],
     "likedIng": ["Salmon"], "dislikedIng": ["White bread"], "nutrientConstraint": "Sugar ≤ 10g/meal",
     "cs": "High", "fs": "Medium"},
    {"id": 2, "name": "Lucas", "age": 28, "emoji": "🏋️", "condition": "None",
     "allergies": ["Nuts"], "likedTags": ["High-protein"], "dislikedTags": ["Vegan"],
     "likedIng": [], "dislikedIng": ["Tofu"], "nutrientConstraint": "Protein ≥ 30g/meal",
     "cs": "Low", "fs": "Low"},
    {"id": 3, "name": "Sophie", "age": 35, "emoji": "🌱", "condition": "High cholesterol",
     "allergies": [], "likedTags": ["Heart-healthy"], "dislikedTags": [],
     "likedIng": ["Olive oil"], "dislikedIng": ["Butter"], "nutrientConstraint": "Fat ≤ 20g/meal",
     "cs": "Medium", "fs": "High"},
    {"id": 4, "name": "No Profile", "age": None, "emoji": "👤", "condition": "None",
     "allergies": [], "likedTags": [], "dislikedTags": [], "likedIng": [], "dislikedIng": [],
     "nutrientConstraint": "None", "cs": "—", "fs": "—"},
]

# ── Knowledge-graph path (Flow A → KG tab) ────────────────────────────────
KG_NODES = [
    {"id": "user", "label": "Marie", "x": 40, "y": 120, "type": "user"},
    {"id": "cond", "label": "Diabetes T2", "x": 155, "y": 40, "type": "condition"},
    {"id": "cs", "label": "CS: High", "x": 155, "y": 200, "type": "skill"},
    {"id": "constr", "label": "Low Sugar", "x": 280, "y": 40, "type": "constraint"},
    {"id": "cat", "label": "Fish/Salads", "x": 280, "y": 200, "type": "category"},
    {"id": "recipe", "label": "Salmon Bowl", "x": 410, "y": 120, "type": "recipe"},
]
KG_EDGES = [
    {"from": "user", "to": "cond", "label": "has_condition"},
    {"from": "user", "to": "cs", "label": "cooking_skill"},
    {"from": "cond", "to": "constr", "label": "requires"},
    {"from": "cs", "to": "cat", "label": "can_prepare"},
    {"from": "constr", "to": "recipe", "label": "satisfied_by"},
    {"from": "cat", "to": "recipe", "label": "contains"},
]

RETRIEVER_LABELS = {
    "dense": "Dense", "sparse": "BM25", "hybrid": "Hybrid",
    "text-to-pandas": "Text→Pandas", "fusion": "Fusion (3-way)",
}
# Estimated latencies, used to fake the processing timer
RETRIEVER_LATENCY = {"dense": 6.0, "sparse": 1.5, "hybrid": 3.5, "text-to-pandas": 2.5, "fusion": 5.0}
