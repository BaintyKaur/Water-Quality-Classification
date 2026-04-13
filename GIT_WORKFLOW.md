# 🤝 Git Collaboration Guide
## Water Quality Project — Team Workflow

---

## Initial Setup (Do Once Per Member)

```bash
# Clone the repo (after one member creates it on GitHub)
git clone https://github.com/YOUR_ORG/water-quality-project.git
cd water-quality-project

# Set your identity
git config user.name "Your Full Name"
git config user.email "your@email.com"

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Daily Workflow

```bash
# ALWAYS start by pulling latest changes
git checkout main
git pull origin main

# Create your feature branch
git checkout -b feature/your-name-stage-4-eda

# Work on your notebook/code...

# Stage and commit with MEANINGFUL messages
git add notebooks/04_eda.ipynb data/processed/eda_*.png
git commit -m "feat(eda): Add correlation heatmap and violin plots for class comparison

- Added Pearson correlation heatmap with mask for upper triangle
- Violin plots for all 9 features split by Potability class
- Mann-Whitney U test results showing ph and Sulfate are most significant
- Saved plots to data/processed/"

# Push your branch
git push origin feature/your-name-stage-4-eda
```

---

## Opening a Pull Request (PR)

1. Go to GitHub → your repo
2. Click **"Compare & pull request"**
3. Fill in the PR description:
   ```
   ## What this PR does
   - Completes Stage 4: EDA
   - Adds 6 visualization plots
   - Runs Mann-Whitney U tests for all features

   ## Screenshots
   [paste plot images]

   ## Checklist
   - [ ] Notebook runs end-to-end without errors
   - [ ] Plots saved to data/processed/
   - [ ] No sensitive data committed
   ```
4. Assign a **reviewer** (teammate)
5. Wait for review before merging

---

## Reviewing a PR

1. Go to the PR on GitHub
2. Click **"Files changed"** tab
3. Leave **inline comments** on specific lines
4. Either:
   - **Approve** → if everything looks good
   - **Request changes** → if fixes needed
5. Never just comment "LGTM" — add a real observation

---

## Using GitHub Issues

Create an issue for each task:
```
Title: [Stage 5] Feature Engineering - WHO compliance flags
Assignee: @member2
Labels: enhancement
Body:
  - Create binary flags for each WHO parameter
  - Create composite total_who_compliant score
  - Run RF importance to select features
  - Save engineered CSV to data/processed/
```

---

## Commit Message Format

```
type(scope): Short description (max 72 chars)

[Optional body explaining WHY, not WHAT]
```

**Types:**
- `feat` — new feature or notebook section
- `fix` — bug fix
- `data` — data file changes
- `docs` — documentation changes
- `refactor` — code reorganization
- `model` — model training/saving

**Examples of GOOD commit messages:**
```
feat(preprocessing): Add class-wise median imputation for ph, Sulfate, THM

fix(model): Correct SMOTE applied to test data (data leakage bug)

data(eda): Save correlation heatmap and violin plots to processed/

model(xgboost): Tune n_estimators 200→300, improves AUC by 0.02

docs(readme): Add model comparison table and live deployment link
```

**Examples of BAD commit messages:**
```
update
fixed stuff
notebook changes
wip
asdfgh
```

---

## Branch Naming Convention

```
feature/[member-name]-stage-[N]-[short-desc]

Examples:
  feature/alice-stage-4-eda
  feature/bob-stage-6-model-training
  feature/carol-stage-8-shap
  feature/dave-stage-9-streamlit
```

---

## Resolving Merge Conflicts

```bash
# If you get a merge conflict:
git checkout main
git pull origin main
git checkout your-branch
git merge main

# Fix conflicts in files (look for <<<<<<< HEAD markers)
# Then:
git add .
git commit -m "fix: Resolve merge conflict in notebook 04_eda"
git push origin your-branch
```

---

## End of Project Checklist

- [ ] All 10 notebooks run end-to-end
- [ ] `requirements.txt` is updated
- [ ] All models saved in `models/`
- [ ] All plots saved in `data/processed/`
- [ ] Streamlit app runs locally (`streamlit run app/app.py`)
- [ ] App deployed on Streamlit Cloud — URL in README
- [ ] README has all sections + screenshots
- [ ] PPT uploaded to `presentation/`
- [ ] Individual GitHub activity screenshots in `individual_profiles/`
- [ ] GitHub Issues tracker shows completed tasks
- [ ] All branches merged into main via PRs
