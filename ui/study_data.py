"""Mock data + instruments for Flow B (Structured Evaluation).

The questionnaire instruments and flow parameters are: 5-point ResQue-based
Likert (not 7-point ad-hoc), attention checks, a post-study block, random
assignment, 3 plans, Latin-square counterbalancing.

Replace ``STUDY_PLANS`` with real pipeline output later; the screens depend only
on the shapes defined here. Instruments/config would move to the HF Dataset repo
config so the admin can edit them without redeployment.
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Study configuration (hard-coded defaults for now) ──────────────────────
STUDY_CONFIG = {
    "study_id": "recsys2026-review",
    "profile_assignment": "random_round_robin",
    "plans_per_profile": 3,
    "counterbalancing": "latin_square",
    "attention_checks": True,
    "likert_scale": 5,
}

# 5-point Likert, all points labelled (ResQue-compatible)
LIKERT_5 = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

# ── Demographics (GDPR-compliant) ─────────────────────────────────────────
DEMO_AGE = ["Under 18", "18–24", "25–34", "35–44", "45–54", "55–64", "65+"]
DEMO_EMPLOYMENT = ["Student", "Employed", "Self-employed", "Other"]
DEMO_COOK_FREQ = ["Daily", "Few times/week", "Once/week", "Rarely", "Never"]
DEMO_COOK_SKILL = ["Beginner", "Intermediate", "Advanced", "Expert"]
# Representative subset of the ISO country list (broad geographic context only;
# swap for the full ISO 3166 list when wiring the real config).
DEMO_COUNTRIES = [
    "Argentina", "Australia", "Austria", "Belgium", "Brazil", "Bulgaria", "Canada",
    "Chile", "China", "Colombia", "Croatia", "Czech Republic", "Denmark", "Egypt",
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "India",
    "Indonesia", "Ireland", "Israel", "Italy", "Japan", "Kenya", "South Korea",
    "Latvia", "Lithuania", "Luxembourg", "Malaysia", "Mexico", "Morocco",
    "Netherlands", "New Zealand", "Nigeria", "Norway", "Pakistan", "Peru",
    "Philippines", "Poland", "Portugal", "Romania", "Russia", "Saudi Arabia",
    "Singapore", "Slovakia", "Slovenia", "South Africa", "Spain", "Sweden",
    "Switzerland", "Thailand", "Tunisia", "Turkey", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States", "Vietnam", "Other (not listed)",
]

# ── Profiles (between-subjects). "baseline" = no-constraint control ────────
STUDY_PROFILES = [
    {"id": "A", "icon": "🥗", "specificity": "general", "name": "Profile A",
     "goal": "Weight loss · Vegetarian",
     "desc": "Vegetarian wanting to lose weight. Prefers simple meals, limited cooking time, dislikes mushrooms.",
     "tags": ["Vegetarian", "Low-calorie", "Quick prep"],
     "prompt": "I am vegetarian, want to lose weight, and need a meal plan. I prefer simple meals, have limited cooking time, and dislike mushrooms."},
    {"id": "B", "icon": "💪", "specificity": "general", "name": "Profile B",
     "goal": "Muscle gain · Budget",
     "desc": "Meat-eater wanting high-protein meals for muscle gain. Limited budget, easy preparation.",
     "tags": ["High-protein", "Budget", "Easy prep"],
     "prompt": "I want a high-protein meal plan for muscle gain. I eat meat, have a limited budget, and want meals that are easy to prepare."},
    {"id": "C", "icon": "👨‍👧", "specificity": "general", "name": "Profile C",
     "goal": "Balanced routine · Family",
     "desc": "Cooking for self and one child. Balanced weekday meals, no spicy food, family-friendly.",
     "tags": ["Family-friendly", "Balanced", "No spice"],
     "prompt": "I want a balanced meal plan for a busy weekday routine. I cook for myself and one child, and want to avoid spicy food."},
    {"id": "D", "icon": "🌿", "specificity": "specific", "name": "Profile D",
     "goal": "Complex constraints · Specific",
     "desc": "Vegan with gluten intolerance and nut allergy. Needs high-iron meals, cooks only on weekends, student budget.",
     "tags": ["Vegan", "Gluten-free", "Nut-free", "High-iron", "Budget"],
     "prompt": "I am vegan, gluten-intolerant, and have a nut allergy. I need meals rich in iron and B12. I can only cook on weekends and eat leftovers during the week. I have a tight student budget."},
    {"id": "baseline", "icon": "🔲", "specificity": "none", "name": "Baseline (No Profile)",
     "goal": "No constraints",
     "desc": "Generic healthy meal plans with no dietary profile — the between-subjects baseline for general plan quality.",
     "tags": ["Neutral", "Open-ended"],
     "prompt": "Generate a general healthy meal plan for an average adult with no specific dietary restrictions."},
]

# ── Pre-generated plans per profile (3 used by default) ────────────────────
# Each plan: {cond, label, explanation:{summary, reasons:[{icon,text}]}, meals:[{type,icon,name,time,kcal,nutri}]}
STUDY_PLANS = {
    "A": [
        {"cond": "C1", "label": "Plan 1",
         "explanation": {"summary": "Fully vegetarian, low in calories (~1,290 kcal/day), minimal cooking time. No mushroom dishes.",
                         "reasons": [{"icon": "🥗", "text": "All meals vegetarian — no meat, fish or hidden meat stocks."},
                                     {"icon": "⏱", "text": "Longest prep 25 min (dinner); all others under 10 min."},
                                     {"icon": "🚫", "text": "No mushrooms; chickpeas used as the legume base."},
                                     {"icon": "⚖️", "text": "~1,290 kcal/day — suitable for a moderate deficit."}]},
         "meals": [{"type": "Breakfast", "icon": "🥣", "name": "Greek Yogurt with Berries & Granola", "time": "5 min", "kcal": 290, "nutri": "A"},
                   {"type": "Lunch", "icon": "🥗", "name": "Caprese Salad with Chickpeas", "time": "10 min", "kcal": 370, "nutri": "A"},
                   {"type": "Dinner", "icon": "🍝", "name": "Tomato Lentil Pasta", "time": "25 min", "kcal": 440, "nutri": "B"},
                   {"type": "Snack", "icon": "🍎", "name": "Apple & Almond Butter", "time": "2 min", "kcal": 190, "nutri": "A"}]},
        {"cond": "C2", "label": "Plan 2",
         "explanation": {"summary": "Slightly higher-calorie vegetarian plan (~1,350 kcal) prioritising nutrient density and satiety. No mushrooms.",
                         "reasons": [{"icon": "🥗", "text": "Vegetarian — eggs and dairy as protein sources."},
                                     {"icon": "💪", "text": "Quinoa and cottage cheese give complete protein to preserve muscle."},
                                     {"icon": "🚫", "text": "No mushrooms; all vegetables profile-safe."},
                                     {"icon": "⏱", "text": "Moderate prep (12–30 min) — more varied."}]},
         "meals": [{"type": "Breakfast", "icon": "🥑", "name": "Avocado Toast with Poached Egg", "time": "12 min", "kcal": 340, "nutri": "A"},
                   {"type": "Lunch", "icon": "🥗", "name": "Spinach & Quinoa Power Bowl", "time": "15 min", "kcal": 390, "nutri": "A"},
                   {"type": "Dinner", "icon": "🫘", "name": "Vegetable Curry with Brown Rice", "time": "30 min", "kcal": 460, "nutri": "B"},
                   {"type": "Snack", "icon": "🫐", "name": "Mixed Berries with Cottage Cheese", "time": "3 min", "kcal": 160, "nutri": "A"}]},
        {"cond": "C3", "label": "Plan 3",
         "explanation": {"summary": "Mediterranean-inspired vegetarian plan (~1,250 kcal) centred on chickpeas and falafel. No mushrooms.",
                         "reasons": [{"icon": "🥗", "text": "Vegetarian and vegan-compatible — no eggs or dairy."},
                                     {"icon": "⏱", "text": "All meals under 20 min."},
                                     {"icon": "🚫", "text": "Mushroom-free; hummus and tahini for healthy plant fats."},
                                     {"icon": "🌍", "text": "Mediterranean patterns are well-evidenced for weight management."}]},
         "meals": [{"type": "Breakfast", "icon": "🥣", "name": "Overnight Oats with Chia & Banana", "time": "5 min", "kcal": 320, "nutri": "A"},
                   {"type": "Lunch", "icon": "🌯", "name": "Hummus & Roasted Veggie Wrap", "time": "10 min", "kcal": 360, "nutri": "A"},
                   {"type": "Dinner", "icon": "🧆", "name": "Falafel Bowl with Tabbouleh", "time": "20 min", "kcal": 430, "nutri": "A"},
                   {"type": "Snack", "icon": "🥕", "name": "Carrot Sticks & Tahini", "time": "3 min", "kcal": 140, "nutri": "A"}]},
    ],
    "B": [
        {"cond": "C1", "label": "Plan 1",
         "explanation": {"summary": "High-protein plan (~1,950 kcal) on lean meats and eggs for muscle gain. Simple, affordable staples.",
                         "reasons": [{"icon": "💪", "text": "~150g protein — meets hypertrophy targets for most body weights."},
                                     {"icon": "🐟", "text": "Tuna and chicken are cost-effective, high-quality protein."},
                                     {"icon": "⏱", "text": "No meal over 20 min."},
                                     {"icon": "💶", "text": "Eggs, oats, tuna, chicken — among the cheapest proteins."}]},
         "meals": [{"type": "Breakfast", "icon": "🍳", "name": "Scrambled Eggs with Oats & Banana", "time": "10 min", "kcal": 510, "nutri": "B"},
                   {"type": "Lunch", "icon": "🍗", "name": "Grilled Chicken Rice Bowl", "time": "20 min", "kcal": 580, "nutri": "A"},
                   {"type": "Dinner", "icon": "🐟", "name": "Tuna Pasta with Olive Oil", "time": "20 min", "kcal": 540, "nutri": "B"},
                   {"type": "Snack", "icon": "🥤", "name": "Milk & Peanut Butter Toast", "time": "3 min", "kcal": 320, "nutri": "B"}]},
        {"cond": "C2", "label": "Plan 2",
         "explanation": {"summary": "Varied high-protein plan (~1,870 kcal): eggs, chicken, beef, cottage cheese. Healthy fats from walnuts and beef.",
                         "reasons": [{"icon": "💪", "text": "Multiple protein sources to maximise muscle protein synthesis."},
                                     {"icon": "🥩", "text": "Lean beef provides creatine and heme iron."},
                                     {"icon": "🧠", "text": "Walnuts add omega-3 for recovery."},
                                     {"icon": "⏱", "text": "Longest prep 25 min."}]},
         "meals": [{"type": "Breakfast", "icon": "🥚", "name": "Boiled Eggs with Whole Grain Bread", "time": "8 min", "kcal": 480, "nutri": "A"},
                   {"type": "Lunch", "icon": "🍗", "name": "Chicken & Lentil Soup", "time": "25 min", "kcal": 520, "nutri": "A"},
                   {"type": "Dinner", "icon": "🥩", "name": "Beef Stir-fry with Broccoli & Rice", "time": "25 min", "kcal": 590, "nutri": "B"},
                   {"type": "Snack", "icon": "🫙", "name": "Cottage Cheese & Walnuts", "time": "2 min", "kcal": 280, "nutri": "A"}]},
        {"cond": "C3", "label": "Plan 3",
         "explanation": {"summary": "Performance plan (~1,840 kcal) with salmon, turkey and a protein smoothie. Higher omega-3 for recovery.",
                         "reasons": [{"icon": "🐟", "text": "Salmon — top omega-3 source for post-workout inflammation."},
                                     {"icon": "🦃", "text": "Turkey is lean, budget-friendly, high in leucine."},
                                     {"icon": "⏱", "text": "Smoothie breakfast in 5 min — ideal pre-training."},
                                     {"icon": "💪", "text": "Hard-boiled eggs as portable protein."}]},
         "meals": [{"type": "Breakfast", "icon": "🥤", "name": "Protein Smoothie with Oats & Berries", "time": "5 min", "kcal": 490, "nutri": "B"},
                   {"type": "Lunch", "icon": "🍗", "name": "Turkey & Quinoa Salad", "time": "15 min", "kcal": 530, "nutri": "A"},
                   {"type": "Dinner", "icon": "🐟", "name": "Baked Salmon with Sweet Potato", "time": "30 min", "kcal": 560, "nutri": "A"},
                   {"type": "Snack", "icon": "🥚", "name": "Hard-boiled Eggs & Almonds", "time": "10 min", "kcal": 260, "nutri": "A"}]},
    ],
    "C": [
        {"cond": "C1", "label": "Plan 1",
         "explanation": {"summary": "Family-friendly plan (~1,450 kcal) with mild, child-approved flavours. No spicy ingredients.",
                         "reasons": [{"icon": "👧", "text": "Porridge and banana widely accepted by children."},
                                     {"icon": "🌶️", "text": "Zero spice — herbs only."},
                                     {"icon": "🍝", "text": "Pasta salad is quick and child-friendly."},
                                     {"icon": "⏱", "text": "Baked chicken 35 min but minimal active prep."}]},
         "meals": [{"type": "Breakfast", "icon": "🥣", "name": "Banana Porridge with Cinnamon", "time": "8 min", "kcal": 340, "nutri": "A"},
                   {"type": "Lunch", "icon": "🥗", "name": "Pasta Salad with Mozzarella & Tomatoes", "time": "15 min", "kcal": 420, "nutri": "B"},
                   {"type": "Dinner", "icon": "🍗", "name": "Baked Chicken with Roasted Vegetables", "time": "35 min", "kcal": 480, "nutri": "A"},
                   {"type": "Snack", "icon": "🍌", "name": "Banana & Peanut Butter", "time": "2 min", "kcal": 210, "nutri": "B"}]},
        {"cond": "C2", "label": "Plan 2",
         "explanation": {"summary": "Wholesome family plan (~1,480 kcal): weekend-feel pancakes and classic fish fingers. No spice.",
                         "reasons": [{"icon": "👧", "text": "Fish fingers and sweet potato wedges popular with kids."},
                                     {"icon": "🌶️", "text": "All flavours mild and familiar."},
                                     {"icon": "🥦", "text": "Rice bowl encourages vegetable eating."},
                                     {"icon": "⏱", "text": "Pancakes batch-prep and freeze."}]},
         "meals": [{"type": "Breakfast", "icon": "🥞", "name": "Whole Grain Pancakes with Maple Syrup", "time": "15 min", "kcal": 360, "nutri": "B"},
                   {"type": "Lunch", "icon": "🍱", "name": "Rice Bowl with Grilled Chicken & Broccoli", "time": "20 min", "kcal": 440, "nutri": "A"},
                   {"type": "Dinner", "icon": "🐟", "name": "Fish Fingers with Sweet Potato Wedges", "time": "30 min", "kcal": 490, "nutri": "B"},
                   {"type": "Snack", "icon": "🍎", "name": "Apple Slices with Cheese", "time": "3 min", "kcal": 190, "nutri": "A"}]},
        {"cond": "C3", "label": "Plan 3",
         "explanation": {"summary": "Comforting family plan (~1,440 kcal) with bolognese centrepiece and a mild chicken soup. No spice.",
                         "reasons": [{"icon": "👧", "text": "Bolognese with wholemeal pasta is highly child-accepted."},
                                     {"icon": "🌶️", "text": "Soup explicitly mild — stock-based, no chilli."},
                                     {"icon": "🧇", "text": "French toast is a familiar sweet breakfast."},
                                     {"icon": "⏱", "text": "Bolognese scales for next-day lunch."}]},
         "meals": [{"type": "Breakfast", "icon": "🧇", "name": "French Toast with Berries", "time": "12 min", "kcal": 350, "nutri": "B"},
                   {"type": "Lunch", "icon": "🍲", "name": "Mild Chicken & Vegetable Soup", "time": "30 min", "kcal": 400, "nutri": "A"},
                   {"type": "Dinner", "icon": "🍝", "name": "Bolognese with Wholemeal Spaghetti", "time": "35 min", "kcal": 510, "nutri": "B"},
                   {"type": "Snack", "icon": "🫐", "name": "Yogurt with Berries", "time": "3 min", "kcal": 180, "nutri": "A"}]},
    ],
    "D": [
        {"cond": "C1", "label": "Plan 1",
         "explanation": {"summary": "Iron-rich legumes and leafy greens, naturally gluten-free grains, no nuts. All meals batch-cookable on Sunday.",
                         "reasons": [{"icon": "✅", "text": "No gluten — rice and quinoa as base carbs."},
                                     {"icon": "✅", "text": "Nut-free; seeds (sunflower, hemp) for fats."},
                                     {"icon": "🩸", "text": "Lentils, spinach, tofu are top plant iron sources."},
                                     {"icon": "📦", "text": "Batch-cook stew and bowls Sunday; keeps 4 days."}]},
         "meals": [{"type": "Breakfast", "icon": "🥤", "name": "Spinach & Fortified Oat Smoothie", "time": "5 min", "kcal": 310, "nutri": "A"},
                   {"type": "Lunch", "icon": "🥗", "name": "Quinoa & Roasted Chickpea Bowl", "time": "30 min (batch)", "kcal": 440, "nutri": "A"},
                   {"type": "Dinner", "icon": "🍲", "name": "Red Lentil & Kale Stew", "time": "40 min (batch)", "kcal": 420, "nutri": "A"},
                   {"type": "Snack", "icon": "🌱", "name": "Pumpkin Seeds & Dried Apricots", "time": "1 min", "kcal": 180, "nutri": "B"}]},
        {"cond": "C2", "label": "Plan 2",
         "explanation": {"summary": "Protein-forward vegan plan on tofu and tempeh (iron + B12). Gluten-free; sesame replaces nuts.",
                         "reasons": [{"icon": "✅", "text": "Tamari instead of soy sauce — gluten-free."},
                                     {"icon": "✅", "text": "Sesame and tahini for fats, no nut allergens."},
                                     {"icon": "🩸", "text": "Tempeh and edamame deliver iron and complete amino acids."},
                                     {"icon": "📦", "text": "Stir-fry and patties prepped Sunday."}]},
         "meals": [{"type": "Breakfast", "icon": "🍚", "name": "Rice Porridge with Hemp Seeds & Blueberries", "time": "10 min", "kcal": 340, "nutri": "A"},
                   {"type": "Lunch", "icon": "🥙", "name": "Baked Tofu Wrap (Corn Tortilla) with Avocado", "time": "25 min (batch)", "kcal": 460, "nutri": "A"},
                   {"type": "Dinner", "icon": "🍜", "name": "Tempeh & Broccoli Stir-fry on Brown Rice", "time": "35 min (batch)", "kcal": 450, "nutri": "A"},
                   {"type": "Snack", "icon": "🫘", "name": "Edamame with Sea Salt", "time": "5 min", "kcal": 160, "nutri": "A"}]},
        {"cond": "C3", "label": "Plan 3",
         "explanation": {"summary": "Mediterranean-vegan hybrid with black beans and beet for iron, plus vitamin-C pairings to boost absorption.",
                         "reasons": [{"icon": "✅", "text": "Corn and potato base — gluten-free and allergen-safe."},
                                     {"icon": "🍋", "text": "Vitamin-C pairings (peppers, lemon) enhance iron absorption."},
                                     {"icon": "🩸", "text": "Black beans ~3.6 mg iron per cup."},
                                     {"icon": "📦", "text": "Stuffed peppers and chili in one Sunday session."}]},
         "meals": [{"type": "Breakfast", "icon": "🫐", "name": "Chia Pudding with Fortified Plant Milk", "time": "5 min (night before)", "kcal": 290, "nutri": "A"},
                   {"type": "Lunch", "icon": "🌮", "name": "Black Bean & Roasted Pepper Tacos (Corn)", "time": "30 min (batch)", "kcal": 430, "nutri": "A"},
                   {"type": "Dinner", "icon": "🫑", "name": "Stuffed Bell Peppers with Quinoa & Tomato", "time": "45 min (batch)", "kcal": 400, "nutri": "A"},
                   {"type": "Snack", "icon": "🥒", "name": "Cucumber & Sunflower Seed Hummus", "time": "3 min", "kcal": 150, "nutri": "A"}]},
    ],
    "baseline": [
        {"cond": "C1", "label": "Plan 1",
         "explanation": {"summary": "Balanced general-purpose plan with moderate calories and varied macronutrients for an average adult.",
                         "reasons": [{"icon": "⚖️", "text": "~50% carbs, 25% protein, 25% fat."},
                                     {"icon": "🥦", "text": "Every meal includes a vegetable component."},
                                     {"icon": "⏱", "text": "All prep under 30 min."}]},
         "meals": [{"type": "Breakfast", "icon": "🥣", "name": "Muesli with Milk & Banana", "time": "3 min", "kcal": 370, "nutri": "B"},
                   {"type": "Lunch", "icon": "🥗", "name": "Mixed Salad with Grilled Chicken & Croutons", "time": "15 min", "kcal": 420, "nutri": "A"},
                   {"type": "Dinner", "icon": "🍝", "name": "Spaghetti Bolognese", "time": "30 min", "kcal": 530, "nutri": "B"},
                   {"type": "Snack", "icon": "🍎", "name": "Apple & Cheddar Cheese", "time": "1 min", "kcal": 200, "nutri": "B"}]},
        {"cond": "C2", "label": "Plan 2",
         "explanation": {"summary": "Mediterranean-style plan with healthy fats, lean proteins and plenty of vegetables.",
                         "reasons": [{"icon": "🫒", "text": "Olive oil and fish provide omega-3 and monounsaturated fats."},
                                     {"icon": "🥦", "text": "Vegetables in every meal."},
                                     {"icon": "⚖️", "text": "1,800–2,000 kcal range for an average adult."}]},
         "meals": [{"type": "Breakfast", "icon": "🧆", "name": "Greek Yogurt with Honey & Walnuts", "time": "3 min", "kcal": 350, "nutri": "A"},
                   {"type": "Lunch", "icon": "🥙", "name": "Falafel Pita with Tzatziki & Tomato", "time": "10 min", "kcal": 430, "nutri": "A"},
                   {"type": "Dinner", "icon": "🐟", "name": "Grilled Sea Bass with Ratatouille", "time": "30 min", "kcal": 480, "nutri": "A"},
                   {"type": "Snack", "icon": "🫐", "name": "Handful of Mixed Berries", "time": "1 min", "kcal": 90, "nutri": "A"}]},
        {"cond": "C3", "label": "Plan 3",
         "explanation": {"summary": "Hearty omnivore plan for quick weekday prep, with red meat once for iron and fish once for omega-3.",
                         "reasons": [{"icon": "⏱", "text": "No meal over 25 min of active cooking."},
                                     {"icon": "🩸", "text": "Red meat once provides heme iron."},
                                     {"icon": "🥦", "text": "Plant-based lunches add fibre, cut saturated fat."}]},
         "meals": [{"type": "Breakfast", "icon": "🍳", "name": "Poached Eggs on Wholegrain Toast", "time": "8 min", "kcal": 340, "nutri": "A"},
                   {"type": "Lunch", "icon": "🥗", "name": "Lentil & Roasted Vegetable Salad", "time": "15 min", "kcal": 380, "nutri": "A"},
                   {"type": "Dinner", "icon": "🥩", "name": "Beef Steak with Green Beans & Potatoes", "time": "25 min", "kcal": 560, "nutri": "B"},
                   {"type": "Snack", "icon": "🥛", "name": "Glass of Milk & Digestive Biscuit", "time": "1 min", "kcal": 180, "nutri": "B"}]},
    ],
}

# ── Study conditions (C1/C2/C3) → real system configuration ───────────────
# Each participant rates ``plans_per_profile`` plans; these are the within-subjects
# CONDITIONS. Every condition maps to a real pipeline configuration that the offline
# build script (``scripts/build_study_plans.py``) selects to produce the plan. The
# ``config`` is recorded with each plan and logged with every rating, but NOT shown
# in the GUI. ``retriever`` ∈ {text-to-pandas, dense, sparse, hybrid, fusion};
# ``hypothesis`` ∈ {H1 = full agentic pipeline, H2 = recommend-only}.
#
# Each condition maps to an **already-run** full-day evaluation result file (the
# recommendations were computed offline; the study selects plans from them — it does
# not re-run the pipeline). ``config_id`` is the evaluation config; ``result_file`` is
# the per-question results holding ``{meal: food.com URL}`` plans + explanations.
#
# This is only the DEFAULT axis. The **effective** conditions are edited by the admin
# in the "🧪 Study (Flow B)" tab and persisted in the study config;
# resolve them via ``get_study_conditions()`` — never read this constant directly.
DEFAULT_STUDY_CONDITIONS = [
    {"cond": "C1", "label": "Plan 1",
     "config": {"config_id": "foodrag_h1_dense_fullday", "hypothesis": "H1", "retriever": "dense",
                "result_file": "eval_results/foodrag_h1_dense_fullday_per_question.parquet"}},
    {"cond": "C2", "label": "Plan 2",
     "config": {"config_id": "foodrag_h2_fullday", "hypothesis": "H2", "retriever": "dense",
                "result_file": "eval_results/foodrag_h2_fullday_per_question.pkl"}},
    {"cond": "C3", "label": "Plan 3",
     "config": {"config_id": "foodrag_h2_t2p_fullday", "hypothesis": "H2", "retriever": "text-to-pandas",
                "result_file": "eval_results/foodrag_h2_t2p_fullday_per_question.pkl"}},
]


def get_study_conditions() -> list[dict]:
    """Effective study conditions, resolved in priority order:

    1. the live admin session (``st.session_state['admin_config']`` — reflects
       unsaved edits in the Conditions tab),
    2. the persisted study config (``config_service.effective_config()`` — used by
       the offline generator, which has no Streamlit session), then
    3. :data:`DEFAULT_STUDY_CONDITIONS`.
    """
    try:  # 1. live session (admin edits, most current) — no-op outside Streamlit
        import streamlit as st

        conds = (st.session_state.get("admin_config") or {}).get("study_conditions")
        if conds:
            return conds
    except Exception:  # noqa: BLE001
        pass
    try:  # 2. persisted config (offline generator / fresh process)
        from .services import config_service

        conds = config_service.effective_config().get("study_conditions")
        if conds:
            return conds
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_STUDY_CONDITIONS  # 3. module default

# Pre-generated real-plan artifacts (one JSON per profile) produced offline by
# ``scripts/build_study_plans.py``. When present they replace the mock STUDY_PLANS;
# when absent the study flow falls back to the mock so the UI still runs.
STUDY_PLANS_DIR = Path(__file__).parent / "study_plans"


def _config_for(cond: str) -> dict:
    for c in get_study_conditions():
        if c["cond"] == cond:
            return c["config"]
    return {}


def _mock_plan_to_unified(plan: dict) -> dict:
    """Adapt a hand-authored mock ``STUDY_PLANS`` entry to the unified plan shape
    (real artifacts already use this shape). Ingredients/steps/food.com link are
    empty for the mock — the real generator fills them from the dataset."""
    recipes = []
    for m in plan.get("meals", []):
        recipes.append({
            "meal": m.get("type", ""),
            "img": m.get("icon", "🍽"),
            "name": m.get("name", ""),
            "time": m.get("time", "—"),
            "cal": m.get("kcal", 0),
            "nutriscore": m.get("nutri", "C"),
            "protein": 0, "carbs": 0, "fat": 0, "fiber": 0, "sugar": 0, "sodium": 0,
            "serving_size_g": None, "who": 5, "fsa": 5, "skill": "Easy",
            "servings": 2, "tags": [], "ingredients": [], "steps": [],
            "recipe_url": "", "recipe_id": None, "substitutions": [],
        })
    return {
        "cond": plan.get("cond", ""),
        "label": plan.get("label", ""),
        "config": _config_for(plan.get("cond", "")),
        "explanation": plan.get("explanation", {}),
        "recipes": recipes,
        "total_kcal": sum(r["cal"] for r in recipes),
        "source": "mock",
    }


def load_study_plans(profile_id: str, n_plans: int | None = None) -> list[dict]:
    """Unified plan list for a profile, in condition order (C1, C2, C3, …).

    Prefers the real pre-generated artifact ``study_plans/{profile_id}.json``; falls
    back to the hand-authored ``STUDY_PLANS`` mock (adapted to the same shape) when no
    artifact exists. Each plan carries a hidden ``config`` (the pipeline configuration
    that produced it) and real recipe cards with ingredients, steps, per-serving
    nutrition and a food.com link."""
    limit = n_plans if n_plans is not None else STUDY_CONFIG["plans_per_profile"]
    artifact = STUDY_PLANS_DIR / f"{profile_id}.json"
    if artifact.exists():
        try:
            data = json.loads(artifact.read_text(encoding="utf-8"))
            plans = data.get("plans", data) if isinstance(data, dict) else data
            if plans:
                return plans[:limit]
        except (json.JSONDecodeError, OSError, KeyError):
            pass  # fall back to mock on any read/parse failure
    return [_mock_plan_to_unified(p) for p in STUDY_PLANS.get(profile_id, [])][:limit]


# ── Per-plan questionnaire: 7 ResQue-based items ───────────────────────────
PLAN_EVAL_ITEMS = [
    {"id": "pq1", "construct": "Perceived Quality", "text": "The recommended meals are appealing to me."},
    {"id": "pq2", "construct": "Perceived Quality", "text": "The meal plan suits my needs."},
    {"id": "ai1", "construct": "Adequacy of Info", "text": "I have enough information to judge this plan."},
    {"id": "us1", "construct": "Usefulness", "text": "This meal plan is practical for everyday life."},
    {"id": "tr1", "construct": "Trust", "text": "I trust this recommendation."},
    {"id": "tp1", "construct": "Transparency", "text": "I understand why this plan was recommended."},
    {"id": "st1", "construct": "Satisfaction", "text": "Overall, I am satisfied with this meal plan."},
]

# Attention check (instructed response). Expected = "Disagree" → value 2
ATTENTION_ITEM = {"id": "attn", "text": "Please select 'Disagree' for this item.", "expected": 2}
# 1-indexed insertion position of the attention check within the 8-item block,
# rotated per plan so participants can't learn to skip it.
ATTENTION_POSITIONS = [3, 6, 2]

# ── Post-study questionnaire (asked once) ─────────────────────────────────
POST_STUDY_ITEMS = [
    {"id": "eq1", "construct": "Explanation Quality", "text": "The explanations helped me understand why each plan was recommended."},
    {"id": "eq2", "construct": "Explanation Quality", "text": "The reasons given for each recommendation were convincing."},
    {"id": "cc1", "construct": "Constraint Compliance", "text": "The meal plans respected the dietary constraints of the profile."},
    {"id": "cc2", "construct": "Constraint Compliance", "text": "I believe the system correctly handled food allergies and health restrictions."},
    {"id": "bi1", "construct": "Behavioural Intention", "text": "I would use this system again for meal planning."},
    {"id": "bi2", "construct": "Behavioural Intention", "text": "I would recommend this system to a friend or colleague."},
    {"id": "su1", "construct": "System Usability", "text": "The meal plans were easy to understand and evaluate."},
]


STUDY_PROFILES_FILE = STUDY_PLANS_DIR / "profiles.json"


def get_study_profiles() -> list[dict]:
    """Effective study profiles (each maps to an evaluation ``question_id``), resolved:

    1. the live admin session, 2. the persisted study config, 3. the generated
    ``study_plans/profiles.json`` default (from ``scripts/build_study_plans.py --select``),
    then 4. the legacy hand-authored :data:`STUDY_PROFILES`.
    """
    try:  # 1. live admin session
        import streamlit as st

        profs = (st.session_state.get("admin_config") or {}).get("study_profiles")
        if profs:
            return profs
    except Exception:  # noqa: BLE001
        pass
    try:  # 2. persisted config
        from .services import config_service

        profs = config_service.effective_config().get("study_profiles")
        if profs:
            return profs
    except Exception:  # noqa: BLE001
        pass
    try:  # 3. generated default artifact
        if STUDY_PROFILES_FILE.exists():
            data = json.loads(STUDY_PROFILES_FILE.read_text(encoding="utf-8"))
            profs = data.get("profiles", data) if isinstance(data, dict) else data
            if profs:
                return profs
    except (json.JSONDecodeError, OSError):
        pass
    return STUDY_PROFILES  # 4. legacy default


def get_profile(profile_id: str) -> dict:
    return next(p for p in get_study_profiles() if p["id"] == profile_id)


# ── Flow A real profiles ──────────────────────────────────────────────────
# The interactive recommender's persona cards are drawn from the SAME evaluation
# profiles used in the study (ui/study_plans/profiles.json) so live runs reproduce
# evaluated behaviour. Three admin-selected profiles + one user-defined (custom) slot.
def load_all_profiles() -> list[dict]:
    """Every evaluation profile in profiles.json (with full constraints), or []."""
    try:
        data = json.loads(STUDY_PROFILES_FILE.read_text(encoding="utf-8"))
        return data.get("profiles", data) if isinstance(data, dict) else data
    except (json.JSONDecodeError, OSError):
        return []


def _nutrient_constraint(c: dict) -> str:
    """Build a parseable nutrient-constraint string from a profile's STRUCTURED limits.

    Uses the upper/lower bound explicitly (e.g. "totalFat ≤ 15g") rather than the
    human-readable line ("totalFat: low (0–15 per 100 g)"), which the deterministic
    parser would mis-read as "≤ 0" (it grabs the first number)."""
    parts: list[str] = []
    for col, spec in (c.get("relative_nutrients") or {}).items():
        name = str(col).split("[")[0].strip()
        limits = (spec or {}).get("limits") or {}
        pref = (spec or {}).get("preference")
        if pref == "low" and limits.get("upper") is not None:
            parts.append(f"{name} ≤ {float(limits['upper']):g}g")
        elif pref == "high" and limits.get("lower") is not None:
            parts.append(f"{name} ≥ {float(limits['lower']):g}g")
    for col, val in (c.get("total_nutrients") or {}).items():
        name = str(col).split("[")[0].strip()
        parts.append(f"{name} ≤ {float(val):g}")
    for name, val in (c.get("nutrition_scores") or {}).items():
        parts.append(f"{name} ≥ {float(val):g}")
    return "; ".join(parts) or "None"


def _profile_to_persona(p: dict) -> dict:
    """Map an evaluation profile (constraints + prompt) to the recommender persona shape."""
    c = p.get("constraints", {}) or {}
    tags = c.get("tags", {}) or {}
    ings = c.get("ingredients", {}) or {}
    return {
        "id": p["id"], "name": p.get("name", p["id"]), "age": None,
        "emoji": p.get("icon", "🍽"), "condition": "None",
        "allergies": [],
        "likedTags": tags.get("likes", []) or [],
        "dislikedTags": tags.get("dislikes", []) or [],
        "likedIng": ings.get("likes", []) or [],
        "dislikedIng": ings.get("dislikes", []) or [],
        "nutrientConstraint": _nutrient_constraint(c),
        "cs": "—", "fs": "—",
        # Real-profile extras (used by the card + live query prefill)
        "prompt": p.get("prompt", ""),
        "desc": p.get("desc", ""),
        "constraint_lines": p.get("constraint_lines", []),
        "real_profile": True,
    }


def flow_a_personas() -> list[dict]:
    """The Flow A persona cards: the admin-selected real profiles + the custom slot."""
    from .admin_config import flow_a_profile_ids
    from .mock_data import PERSONAS

    by_id = {p["id"]: p for p in load_all_profiles()}
    personas = [_profile_to_persona(by_id[i]) for i in flow_a_profile_ids() if i in by_id]
    custom = next((p for p in PERSONAS if p["id"] == 4), None)  # user-defined builder card
    if custom:
        personas.append(custom)
    return personas


def plan_eval_items_for(plan_index: int) -> list[dict]:
    """7 ResQue items + the attention check inserted at this plan's rotated position."""
    items = list(PLAN_EVAL_ITEMS)
    if STUDY_CONFIG["attention_checks"]:
        pos = ATTENTION_POSITIONS[plan_index % len(ATTENTION_POSITIONS)]
        items.insert(pos - 1, ATTENTION_ITEM)
    return items
