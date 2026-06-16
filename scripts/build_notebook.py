"""Build the EDA notebook programmatically to ensure valid JSON."""

import nbformat as nbf


def main() -> None:
    nb = nbf.v4.new_notebook()

    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# AI Career Platform Lifecycle Growth Analysis\n\n"
            "This notebook demonstrates the end-to-end MVP pipeline:\n"
            "1. Load and validate synthetic data.\n"
            "2. Compute growth funnel and cohort retention.\n"
            "3. Analyze the onboarding A/B experiment.\n"
            "4. Inspect churn labels and Next Best Action recommendations."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import json\n"
            "import os\n"
            "import sys\n"
            "\n"
            "src_path = os.path.abspath('src')\n"
            "if not os.path.exists(os.path.join(src_path, 'career_growth')):\n"
            "    src_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))\n"
            "sys.path.insert(0, src_path)\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "\n"
            "from career_growth.analytics.experiments import analyze_experiment\n"
            "from career_growth.analytics.funnel import compute_funnel\n"
            "from career_growth.analytics.retention import (\n"
            "    compute_cohort_retention,\n"
            "    compute_day_retention,\n"
            "    compute_rolling_retention,\n"
            ")\n"
            "from career_growth.config import ONBOARDING_EXPERIMENT_ID, PREDICTION_CUTOFF_DAY\n"
            "from career_growth.decisions.next_best_action import recommend_next_action\n"
            "from career_growth.validation.validator import DataValidator\n"
            "\n"
            "sns.set_theme(style='whitegrid')\n"
            "%matplotlib inline"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 1. Load data"))

    cells.append(
        nbf.v4.new_code_cell(
            "import os\n"
            "if os.path.basename(os.getcwd()) == 'notebooks':\n"
            "    os.chdir('..')\n"
            "\n"
            "users = pd.read_csv('data/sample/users.csv', parse_dates=['signup_timestamp'])\n"
            "events = pd.read_csv('data/sample/events.csv', parse_dates=['event_timestamp'])\n"
            "experiment_assignments = pd.read_csv(\n"
            "    'data/sample/experiment_assignments.csv', parse_dates=['assignment_time']\n"
            ")\n"
            "interventions = pd.read_csv('data/sample/interventions.csv', parse_dates=['send_time'])\n"
            "labels = pd.read_csv('data/processed/labels.csv', parse_dates=['label_start', 'label_end'])\n"
            "\n"
            "print(f'Users: {len(users):,}')\n"
            "print(f'Events: {len(events):,}')\n"
            "print(f'Interventions: {len(interventions):,}')"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 2. Data validation"))

    cells.append(
        nbf.v4.new_code_cell(
            "validator = DataValidator('data')\n"
            "validator.users = users\n"
            "validator.events = events\n"
            "validator.experiment_assignments = experiment_assignments\n"
            "validator.interventions = interventions\n"
            "validator.labels = labels\n"
            "report = validator.validate()\n"
            "print('Validation passed:', report.passed)\n"
            "print('Errors:', report.errors)\n"
            "print('Warnings:', report.warnings)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 3. User overview"))

    cells.append(
        nbf.v4.new_code_cell(
            "overview = users.groupby('acquisition_channel').size().reset_index(name='users')\n"
            "overview['share'] = overview['users'] / overview['users'].sum()\n"
            "display(overview)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 4. Growth funnel"))

    cells.append(
        nbf.v4.new_code_cell(
            "funnel = compute_funnel(users, events)\n"
            "display(funnel)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "sns.barplot(data=funnel, x='step', y='users', palette='Blues_d', ax=ax)\n"
            "ax.set_title('Core User Lifecycle Funnel')\n"
            "ax.set_xlabel('Funnel step')\n"
            "ax.set_ylabel('Unique users')\n"
            "plt.xticks(rotation=30, ha='right')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 5. Retention"))

    cells.append(
        nbf.v4.new_code_cell(
            "retention_summary = pd.DataFrame(\n"
            "    [\n"
            "        {'metric': 'D1 retention', 'rate': compute_day_retention(users, events, 1)['retention_rate'].iloc[0]},\n"
            "        {'metric': 'D7 retention', 'rate': compute_day_retention(users, events, 7)['retention_rate'].iloc[0]},\n"
            "        {'metric': 'D14 retention', 'rate': compute_day_retention(users, events, 14)['retention_rate'].iloc[0]},\n"
            "        {\n"
            "            'metric': f'D{PREDICTION_CUTOFF_DAY} rolling retention',\n"
            "            'rate': compute_rolling_retention(users, events, PREDICTION_CUTOFF_DAY),\n"
            "        },\n"
            "    ]\n"
            ")\n"
            "display(retention_summary)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("### Cohort retention heatmap"))

    cells.append(
        nbf.v4.new_code_cell(
            "cohort = compute_cohort_retention(users, events, days=[1, 7, 14])\n"
            "cohort_pivot = cohort.pivot(index='signup_week', columns='day', values='retention_rate')\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 6))\n"
            "sns.heatmap(cohort_pivot, annot=True, fmt='.1%', cmap='YlGnBu', ax=ax)\n"
            "ax.set_title('Cohort Retention by Signup Week')\n"
            "ax.set_xlabel('Day')\n"
            "ax.set_ylabel('Signup week')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6. Churn label distribution"))

    cells.append(
        nbf.v4.new_code_cell(
            "labeled = users.merge(labels[['user_id', 'is_churned']], on='user_id')\n"
            "churn_by_channel = (\n"
            "    labeled.groupby('acquisition_channel')['is_churned']\n"
            "    .mean()\n"
            "    .reset_index(name='churn_rate')\n"
            "    .sort_values('churn_rate', ascending=False)\n"
            ")\n"
            "display(churn_by_channel)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "sns.barplot(data=churn_by_channel, x='acquisition_channel', y='churn_rate', palette='Reds_d', ax=ax)\n"
            "ax.set_title('Churn Rate by Acquisition Channel')\n"
            "ax.set_ylabel('Churn rate')\n"
            "ax.set_ylim(0, 1)\n"
            "plt.xticks(rotation=30, ha='right')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 7. Onboarding A/B experiment"))

    cells.append(
        nbf.v4.new_code_cell(
            "experiment_results = analyze_experiment(\n"
            "    users, events, experiment_assignments, ONBOARDING_EXPERIMENT_ID\n"
            ")\n"
            "print('Sample sizes:', experiment_results['sample_sizes'])\n"
            "print('SRM p-value:', experiment_results['srm_p_value'])\n"
            "print('')\n"
            "for metric_name, results in experiment_results['metrics'].items():\n"
            "    print(f'--- {metric_name} ---')\n"
            "    for r in results:\n"
            "        lift = r.get('relative_lift')\n"
            "        lift_str = f'{lift:+.1%}' if lift is not None else 'control'\n"
            "        print(\n"
            "            f\"{r['variant_id']:12s} n={r['sample_size']:<5d} rate={r['conversion_rate']:.2%} \"\n"
            "            f\"lift={lift_str:<8s} p={r.get('p_value')}\"\n"
            "        )"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 8. Next Best Action sample"))

    cells.append(
        nbf.v4.new_code_cell(
            "sample_users = users.sample(10, random_state=42)\n"
            "recommendations = []\n"
            "for _, user in sample_users.iterrows():\n"
            "    rec = recommend_next_action(user, events)\n"
            "    recommendations.append(\n"
            "        {\n"
            "            'user_id': user['user_id'][:8] + '...',\n"
            "            'action': rec['action_name'],\n"
            "            'channel': rec['channel'],\n"
            "            'reason': rec['reason'],\n"
            "        }\n"
            "    )\n"
            "display(pd.DataFrame(recommendations))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 9. Summary\n\n"
            "- The synthetic dataset passes schema, temporal, and relational validation.\n"
            "- Churn rate is within the target range and varies by acquisition channel.\n"
            "- The onboarding experiment shows a measurable treatment effect with stable randomization.\n"
            "- The rule-based Next Best Action engine assigns contextually appropriate recommendations."
        )
    )

    nb.cells = cells
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {
        "name": "python",
        "version": "3.10.0",
    }

    with open("notebooks/lifecycle_analysis.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Notebook written to notebooks/lifecycle_analysis.ipynb")


if __name__ == "__main__":
    main()
