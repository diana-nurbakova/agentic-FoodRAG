"""Service layer bridging the Streamlit GUI to the ``foodrag`` backend.

- ``backend``    — cached heavy singletons (LLM, embedding, HUMMUS df, query engine)
                   + agent-workflow assembly, async run, and state extraction.
- ``mapping``    — convert backend ``MealPlan``/recipe dicts into the GUI shapes.
- ``rag_service``— ``generate_meal_plan(persona, query, config)`` the GUI calls.

The real backend is opt-in (admin toggle / ``FOODRAG_REAL_BACKEND`` env). When it
is off or unavailable, the GUI uses the mock data in ``ui.mock_data``.
"""
