import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


def compute_answer_stats_df(question_answer_sets):
    answer_counts = []
    requirement_counts = []
    nutri_scores = []
    fsa_scores = []

    for entry in question_answer_sets:
        answers = entry['answer']
        answer_counts.append(len(answers))

        # Requirements
        persona = entry.get('persona', {})
        tags_len = len(persona.get('tags', {}).get('likes', [])) + len(persona.get('tags', {}).get('dislikes', []))
        ingredients_len = len(persona.get('ingredients', {}).get('likes', [])) + len(persona.get('ingredients', {}).get('dislikes', []))
        relative_nutrients_len = len(persona.get('relative_nutrients', {}))
        total_nutrients_len = len(persona.get('total_nutrients', {}))
        nutrition_scores_len = len(persona.get('nutrition_scores', {}))
        total_requirements = tags_len + ingredients_len + relative_nutrients_len + total_nutrients_len + nutrition_scores_len
        requirement_counts.append(total_requirements)

        # Nutrition scores (average per answer set)
        if 'nutri_score' in answers.columns:
            nutri_scores.append(answers['nutri_score'].mean())
        else:
            nutri_scores.append(np.nan)

        if 'fsa_score' in answers.columns:
            fsa_scores.append(answers['fsa_score'].mean())
        else:
            fsa_scores.append(np.nan)

    answer_series = pd.Series(answer_counts)
    requirement_series = pd.Series(requirement_counts)
    nutri_series = pd.Series(nutri_scores)
    fsa_series = pd.Series(fsa_scores)

    percentiles = np.arange(5, 101, 5)

    def basic_stats(series):
        stats = {
            'mean': series.mean(),
            'variance': series.var(),
            'std_dev': series.std(),
            'min': series.min(),
            'max': series.max(),
            'median': series.median(),
            'num_questions': len(series)
        }
        return stats
    def split_by_percentiles(target_series):
        """
        Splits indices of given series into percentile bins.

        :return: dict mapping percentile bin labels to index lists
        """

        thresholds = np.percentile(target_series.values, percentiles)
        masks = {}

        for i in range(len(thresholds) - 1):
            low, high = thresholds[i], thresholds[i + 1]
            label = f'percentile_{int(percentiles[i])}'

            if i < len(thresholds) - 2:
                mask = (target_series.values >= low) & (target_series.values < high)
            else: # last bin includes the high end
                mask = (target_series.values >= low) & (target_series.values <= high)

            masks[label] = mask

        return masks

    def compute_grouped_means(metric_series: pd.Series, bins: dict):
        """
        Computes the mean of metric_series for each index group in bins.

        :param metric_series: Series of metric values
        :param bins: dict from split_by_percentiles mapping to index lists
        :return: dict mapping percentile labels to means
        """
        stats = {}
        for label, indices in bins.items():
            values = metric_series.loc[indices]
            stats[label] = values.mean() if not values.empty else np.nan
        return stats

    def finalize(stats):
        df = pd.DataFrame([stats])
        # Round and convert float values to formatted strings with 3 decimal places (no scientific notation)
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
        return df

    answer_stats = basic_stats(answer_series)
    requirement_stats = basic_stats(requirement_series)
    nutri_stats = basic_stats(nutri_series)
    fsa_stats = basic_stats(fsa_series)

    # Replace percentile values with filtered means based on answer percentiles
    bins = split_by_percentiles(answer_series) # get percentile bins based on answer counts

    answer_stats.update(compute_grouped_means(answer_series, bins))
    requirement_stats.update(compute_grouped_means(requirement_series, bins))
    nutri_stats.update(compute_grouped_means(nutri_series, bins))
    fsa_stats.update(compute_grouped_means(fsa_series, bins))

    return (
        finalize(answer_stats),
        finalize(requirement_stats),
        finalize(nutri_stats),
        finalize(fsa_stats)
    )



def compute_percentile_curves(question_answer_sets, percentiles, score_type=None, include_requirements=False):
    """
    Computes percentile-aligned average values for answers, scores, and/or requirements.

    Returns:
        - answer_curve: percentile distribution of number of answers
        - score_curve: average nutrition score per answer percentile (if score_type given)
        - requirement_curve: average number of requirements per answer percentile (if include_requirements)
    """
    # Precompute all values
    answer_counts = np.empty(len(question_answer_sets), dtype=int)
    avg_scores = np.full(len(question_answer_sets), np.nan)
    requirement_counts = np.full(len(question_answer_sets), np.nan)

    for i, entry in enumerate(question_answer_sets):
        answer_counts[i] = len(entry['answer'])

        # Avg nutrition score per question
        if score_type and score_type in entry['answer'].columns:
            scores = entry['answer'][score_type].values
            avg_scores[i] = np.nanmean(scores) if len(scores) > 0 else np.nan

        # Requirement count per question
        if include_requirements:
            persona = entry.get('persona', {})
            tags = persona.get('tags', {})
            ingredients = persona.get('ingredients', {})
            requirements = (
                    len(tags.get('likes', [])) + len(tags.get('dislikes', [])) +
                    len(ingredients.get('likes', [])) + len(ingredients.get('dislikes', [])) +
                    len(persona.get('relative_nutrients', {})) +
                    len(persona.get('total_nutrients', {})) +
                    len(persona.get('nutrition_scores', {}))
            )
            requirement_counts[i] = requirements

    # Compute answer percentile thresholds
    thresholds = np.percentile(answer_counts, percentiles)

    # Compute curves
    answer_curve = np.percentile(answer_counts, percentiles)

    score_curve = None
    if score_type:
        score_curve = [
            np.nanmean(avg_scores[answer_counts > t]) if np.any(answer_counts > t) else np.nan
            for t in thresholds
        ]

    requirement_curve = None
    if include_requirements:
        requirement_curve = [
            np.nanmean(requirement_counts[answer_counts > t]) if np.any(answer_counts > t) else np.nan
            for t in thresholds
        ]

    return answer_curve, score_curve, requirement_curve


def plot_fine_percentiles(question_answer_sets, lower_bound=0, upper_bound=100, plot_requirements=False,
                          score_type=None, log_y=False, log_y_base=10):
    percentiles = np.arange(lower_bound, upper_bound + 1, 1)

    # Compute all relevant curves
    answer_curve, score_curve, requirement_curve = compute_percentile_curves(
        question_answer_sets,
        percentiles,
        score_type=score_type,
        include_requirements=plot_requirements
    )

    # Prepare plot
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot answer percentiles
    ax1.plot(percentiles, answer_curve, color='tab:blue', label='Answers')
    ax1.set_xlabel('Percentile')
    ax1.set_ylabel('Number of Answers', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    if log_y: # Optional log scale y-axis for answers
        if np.any(np.asarray(answer_curve) <= 0):
            ax1.set_yscale('symlog', linthresh=1)
        else:
            try:
                ax1.set_yscale('log', base=log_y_base)  # newer Matplotlib
            except TypeError:
                ax1.set_yscale('log')           # fallback
    else:
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Optionally plot requirements
    if plot_requirements and requirement_curve is not None:
        ax2 = ax1.twinx()
        ax2.plot(percentiles, requirement_curve, color='tab:red', label='Requirements')
        ax2.set_ylabel('Number of Requirements', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Optionally plot nutrition score
    if score_type and score_curve is not None:
        if plot_requirements:
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.1))  # Shift third axis to the right
            ax3.plot(percentiles, score_curve, color='tab:green', linestyle='--', label=f'{score_type.upper()}')
            ax3.set_ylabel(f'Avg. {score_type.upper()}', color='tab:green')
            ax3.tick_params(axis='y', labelcolor='tab:green')
        else:
            ax2 = ax1.twinx()
            ax2.plot(percentiles, score_curve, color='tab:green', linestyle='--', label=f'{score_type.upper()}')
            ax2.set_ylabel(f'Avg. {score_type.upper()}', color='tab:green')
            ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title("Percentile Distribution (Answers, Requirements, and Scores)")
    fig.tight_layout()
    plt.grid(True)
    plt.show()


def plot_food_score_visualizations(recipe_score_dfs, drop_duplicates=True):
    """
    Plots health-related nutrition score visualizations from a minimal input format.

    :param recipe_score_dfs: list of DataFrames with only columns:
                             'recipe_id', 'nutri_score', 'who_score', 'fsa_score'
    :param drop_duplicates: whether to drop duplicate recipes by 'recipe_id'
    """
    # Merge and deduplicate recipes by ID
    df = pd.concat(recipe_score_dfs, ignore_index=True)

    if drop_duplicates:
        # Keep the first occurrence of each recipe_id
        df = df.drop_duplicates(subset='recipe_id')

    # --- Nutri Score Pie Chart (from normalized values) ---
    nutri_bins = {
        'A': (0.875, 1.1),  # include 1.0
        'B': (0.625, 0.875),
        'C': (0.375, 0.625),
        'D': (0.125, 0.375),
        'E': (0.0, 0.125)
    }
    nutri_counts = [
        ((df['nutri_score'] >= low) & (df['nutri_score'] < high)).sum()
        for low, high in nutri_bins.values()
    ]

    plt.figure()
    plt.pie(nutri_counts,
            colors=['#008A4B', '#80C342', '#FFCB04', '#F5821F', '#EF4123'],
            labels=list(nutri_bins.keys()),
            startangle=90, pctdistance=0.8, autopct='%1.1f%%', radius=1)
    plt.title(f'NutriScore{" (unique recipes only)" if drop_duplicates else ""}')
    plt.tight_layout()
    plt.show()

    # --- FSA Score Pie Chart (from normalized values) ---
    fsa_bins = list(reversed(range(9)))  # 8 to 0
    fsa_counts = []
    for i in fsa_bins:
        lower = i / 8
        upper = 1.01 if i == 8 else (i + 1) / 8  # make top bin inclusive
        fsa_counts.append(((df['fsa_score'] >= lower) & (df['fsa_score'] < upper)).sum())

    plt.figure()
    plt.pie(fsa_counts,
            colors=['#006638', '#649b31', '#80C342', '#ffcc00', '#F5821F',
                    '#ff6600', '#EF4123', '#cc0000', '#cc0066'],
            labels=[str(i) for i in fsa_bins],
            startangle=90, pctdistance=0.8, autopct='%1.1f%%', radius=1)
    plt.title(f'FSA Score{" (unique recipes only)" if drop_duplicates else ""}')
    plt.tight_layout()
    plt.show()
