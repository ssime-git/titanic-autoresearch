# PRD: titanic-autoresearch

## Objectif

Créer un dépôt GitHub démontrant le pattern **autoresearch itératif** (concept Karpathy) appliqué au feature engineering MLOps. Un agent Claude Code itère sur des hypothèses de features, teste chaque itération, analyse les résultats, et raffine l'hypothèse suivante de manière réflexive.

## Scope

### IN
- Dataset Titanic brut (titanic_original.csv) depuis https://github.com/Geoyi/Cleaning-Titanic-Data
- 7-10 itérations d'hypothèses de feature engineering, chacune avec hypothèse explicite → test → analyse
- Logging structuré en JSONL de chaque itération
- Visualisations de convergence des métriques

### OUT
- Repo GitHub complet (`titanic-autoresearch`, username: `toruve`)
- Structure répliquée et fonctionnelle end-to-end
- Data pipeline exécutable (preprocessing + feature engineering itératif)
- Logs structurés avec métriques (AUC-ROC, Precision, Recall, F1)
- Visualisations PNG: convergence AUC, feature importance, metrics dashboard
- README.md + PROGRAM.md complets et opérationnels


## Deliverables Techniques

### 1. Repo Structure

```
titanic-autoresearch/
├── README.md                      # Overview + Quick Start
├── PROGRAM.md                     # Guide opérationnel autonome
├── requirements.txt               # Python dependencies
├── .gitignore
├── data/
│   ├── raw/
│   │   └── titanic_original.csv   # Source brut (à télécharger)
│   └── processed/
│       └── (générés par le pipeline)
├── notebooks/
│   └── autoresearch_loop.py       # Script principal (SEUL fichier modifié par l'agent)
├── logs/
│   └── iterations.jsonl           # Logs structurés (créés à la 1ère itération)
└── plots/
    └── (PNGs générées après chaque 3-5 itérations)
```

### 2. Data Pipeline

**Source**: titanic_original.csv depuis Geoyi
- URL: https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv

**Colonnes attendues**: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

**Missing values connus**:
- Age: ~20%
- Cabin: ~77%
- Embarked: ~0.2%

**Baseline preprocessing** (non-modifiable par l'agent):
1. Load CSV
2. Suppression: PassengerId, Ticket, Cabin (trop sparse)
3. Imputation Age → median par Pclass
4. One-hot encoding: Sex (male/female), Embarked (C/Q/S)
5. Train/test split 80/20 stratifié sur Survived
6. Standardization des features (si applicable)

**Output du pipeline**: X_train, X_test, y_train, y_test prêts pour l'entraînement

### 3. Autoresearch Loop

Chaque itération suit le pattern strict:

```
Itération N:
1. [HYPOTHÈSE] Formulation explicite + intuition scientifique
2. [FEATURE CREATION] Génération de la feature en Python
3. [TRAINING] Entraînement LogisticRegression sur X_train/y_train
4. [EVALUATION] Calcul sur X_test: AUC-ROC, Precision, Recall, F1
5. [ANALYSIS] Analyse réflexive: "Ça marche car..." ou "Ça ne marche pas car..."
6. [LOGGING] Append ligne JSON à logs/iterations.jsonl
7. [DECISION] Keep ou discard la feature
8. [NEXT] Formule hypothèse suivante basée sur analyse
```

**Baseline (Itération 0)**: Exécuter sans feature engineering. Enregistrer AUC comme référence.

### 4. Hypothèses Sugérées (Modifiables)

L'agent peut les adapter ou en proposer d'autres. Voici 7 exemples:

#### Itération 1: Age × Fare Interaction
**Hypothèse**: "Les passagers riches ET âgés ont survécu plus — l'interaction de la richesse et du statut de vie compte"
```python
df['Age_Fare_Interaction'] = df['Age'] * df['Fare']
```
**Analyse attendue**: Capture subgroupe riches-âgés vs pauvres-jeunes. Effet petit mais propre?

#### Itération 2: Family Size
**Hypothèse**: "Les familles nombreuses avaient moins de chances (chaos, ressources limitées)"
```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
```
**Analyse attendue**: Distribution survie par taille famille? Seuls vs famille?

#### Itération 3: Title Extraction
**Hypothèse**: "Le titre (Mr/Mrs/Master/Miss/Dr) indique le statut social, prédicteur plus fort que Age/Sex seul"
```python
import re
def extract_title(name):
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    return title_search.group(1) if title_search else 'Unknown'
df['Title'] = df['Name'].apply(extract_title)
# Simplifier: Mr, Mrs, Miss, Master, Other
df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', ...], 'Other')
df = pd.get_dummies(df, columns=['Title'], drop_first=True)
```
**Analyse attendue**: Title plus prédictif que Sex? Capture du statut social?

#### Itération 4: Fare per Family
**Hypothèse**: "La richesse distribuée dans la famille (Fare/FamilySize) mieux capture l'aisance réelle que Fare brut"
```python
df['FarePerFamily'] = df['Fare'] / df['FamilySize']
df['FarePerFamily'] = df['FarePerFamily'].fillna(df['FarePerFamily'].median())
```
**Analyse attendue**: Meilleure séparation survie/non-survie que Fare seul?

#### Itération 5: Age Groups × Sex Interaction
**Hypothèse**: "L'interaction Age démographique × Genre crée des sous-groupes avec survie divergente (enfants prioritaires, hommes sacrifiés)"
```python
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 40, 60, 100], 
                        labels=['Infant', 'Child', 'Teen', 'Adult', 'Senior', 'Elderly'])
df['AgeGroup_Sex'] = df['AgeGroup'].astype(str) + '_' + df['Sex']
df = pd.get_dummies(df, columns=['AgeGroup_Sex'], drop_first=True)
```
**Analyse attendue**: Certains groupes (enfants, jeunes femmes) surreprésentés chez survivants?

#### Itération 6: Embarked Port Proxy
**Hypothèse**: "Le port d'embarquement corrèle avec destination/classe → prédictif indépendamment de Pclass"
```python
embarked_survival_rate = df[df['Embarked'].notna()].groupby('Embarked')['Survived'].mean()
df['Embarked_SurvivalProxy'] = df['Embarked'].map(embarked_survival_rate)
```
**Analyse attendue**: Feature engineered ajoute signal au-delà de Pclass + Sex?

#### Itération 7: Cabin Deck Letter (si présent)
**Hypothèse**: "Même si Cabin est 77% missing, extraire la lettre du deck (A/B/C...) quand présent ajoute signal local"
```python
df['CabinDeck'] = df['Cabin'].fillna('X').str[0]
df = pd.get_dummies(df, columns=['CabinDeck'], drop_first=True)
```
**Analyse attendue**: Peu de données mais concentration spatiale du navire corrèle survie?

### 5. Output Format — logs/iterations.jsonl

Format: JSONL (une ligne JSON par itération, NOT comma-separated).

Chaque ligne doit contenir:

```json
{
  "iteration": 1,
  "hypothesis": "Les passagers riches et âgés ont survécu plus — l'interaction capture ce sous-groupe",
  "feature_name": "Age_Fare_Interaction",
  "feature_code": "df['Age_Fare_Interaction'] = df['Age'] * df['Fare']",
  "metrics": {
    "auc_roc": 0.812,
    "precision": 0.78,
    "recall": 0.65,
    "f1": 0.71
  },
  "baseline_auc": 0.805,
  "improvement": "+0.007",
  "status": "keep",
  "analysis": "Interaction capture un sous-groupe distinct (riches-âgés). AUC amélioration modérée (+0.7%), mais feature est claire et monotone. Garde. Prochaine: tester Family Size.",
  "timestamp": "2025-04-01T10:23:45Z"
}
```

**Champs obligatoires**:
- `iteration`: entier (0 pour baseline, puis 1, 2, 3...)
- `hypothesis`: string, explication en français clair de la logique
- `feature_name`: short name pour utiliser en code
- `feature_code`: Python code exact (peut être multi-ligne avec \n)
- `metrics`: dict avec floats à 3-4 décimales: auc_roc, precision, recall, f1
- `baseline_auc`: AUC de la meilleure itération précédente (pour comparaison)
- `improvement`: "+X.XXX" ou "-X.XXX" vs baseline_auc
- `status`: `keep` ou `discard`
- `analysis`: explanation réflexive: pourquoi ça marche/marche pas? Quel sous-groupe bénéficie? Complexité vs gain?
- `timestamp`: ISO 8601 format

### 6. Visualisations (generées tous les 3-5 itérations)

Générer 3 fichiers PNG dans `plots/`:

#### 6.1 convergence_auc.png
- **Type**: Line plot matplotlib
- **X-axis**: Iteration number (0, 1, 2, 3...)
- **Y-axis**: AUC-ROC (0.7 à 0.85 range probable)
- **Lines**: 
  - Baseline AUC comme horizontal line en gris pointillé
  - AUC de chaque itération en bleu
- **Labels**: "Iteration" (x), "AUC-ROC" (y), "Convergence of Feature Engineering"
- **Purpose**: Montrer la progression/stagnation des itérations

#### 6.2 feature_importance_latest.png
- **Type**: Horizontal bar plot
- **X-axis**: Importance score (0 à 1 range)
- **Y-axis**: Feature names (Age, Fare, Sex_male, Family_Size, Title_Mr, ...)
- **Colors**: Gradient (important = dark, non-important = light)
- **Purpose**: Voir quelles features contributent le plus à la dernière version du modèle

#### 6.3 metrics_dashboard.png
- **Type**: Heatmap seaborn
- **Rows**: Iteration numbers
- **Cols**: Metric names (AUC, Precision, Recall, F1)
- **Values**: Metric values, color intensity = performance
- **Colorbar**: Gradient bleu→rouge
- **Purpose**: Vue d'ensemble 2D des trade-offs (ex: itération 3 a haut Recall mais bas Precision)

### 7. Code Requirements

**Language**: Python 3.10+

**Dependencies (requirements.txt)**:
```
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
numpy>=1.23.0
```

**Patterns obligatoires dans `notebooks/autoresearch_loop.py`**:
1. Chaque itération = fonction nommée: `iteration_N_feature_name()`
2. Logging: append à `logs/iterations.jsonl` avec `json.dumps() + "\n"`
3. Reproducibilité: `random_state=42` partout (train_test_split, LogisticRegression)
4. Pas de hardcoding: utiliser variables config si possible
5. Commentaires: "Hypothèse:", "Résultat observé:", "Prochaine itération:"
6. Try/except sur data loading pour erreurs gracieuses

**Baseline model**: LogisticRegression(random_state=42, max_iter=1000)

**Evaluation**:
- Metric = AUC-ROC sur X_test
- Also report: Precision, Recall, F1 (sklearn.metrics)

### 8. README.md — Contenu Minimum

```markdown
# titanic-autoresearch

Démonstration du pattern **autoresearch itératif** (Karpathy) appliqué au feature engineering MLOps.

## Overview

Ce repo montre comment un agent peut autonomously itérer sur des hypothèses de feature engineering:

1. Formule une hypothèse scientifique
2. Implémente la feature
3. Teste et mesure l'impact (AUC-ROC)
4. Analyse les résultats
5. Raffine l'hypothèse suivante

## Quick Start

```bash
# 1. Clone
git clone https://github.com/toruve/titanic-autoresearch.git
cd titanic-autoresearch

# 2. Install
pip install -r requirements.txt

# 3. Download data (ou copie depuis Geoyi repo)
# Place titanic_original.csv in data/raw/

# 4. Run
python notebooks/autoresearch_loop.py

# 5. Review results
cat logs/iterations.jsonl
ls plots/
```

## Results

Au fur et à mesure des itérations, AUC-ROC converge. Voir `plots/convergence_auc.png`.

## Architecture

Voir `PROGRAM.md` pour guide complet.

## References

- Karpathy's autoresearch: https://github.com/karpathy/autoresearch
- Titanic dataset: https://github.com/Geoyi/Cleaning-Titanic-Data
```

---

## Acceptance Criteria

✅ Repo créé et fonctionnel end-to-end sans erreurs
✅ `data/raw/titanic_original.csv` téléchargé et chargé correctement
✅ Baseline itération exécutée et loggée (AUC ~0.80 attendu)
✅ 7-10 itérations complétées avec hypothèses explicites
✅ 7-10 lignes dans `logs/iterations.jsonl` (une par itération)
✅ Chaque hypothèse a logique scientifique (pas random)
✅ AUC final > 0.75 au minimum (attendre amélioration progressive)
✅ Features imbriquées logiquement (itération N+1 basée sur analyse itération N)
✅ 3 visualisations PNG générées (`convergence_auc.png`, `feature_importance_latest.png`, `metrics_dashboard.png`)
✅ `logs/iterations.jsonl` bien formé (JSONL valide, parsable)
✅ README.md complet et opérationnel
✅ PROGRAM.md complet et opérationnel
✅ Code lisible, commenté, pas de dead code
✅ `requirements.txt` exact et testé
✅ `.gitignore` inclut: `*.pyc`, `__pycache__/`, `data/processed/`, `plots/`, `logs/`, `.DS_Store`

---

## Notes Additionnelles

- **L'agent DOIT modifier SEULEMENT** `notebooks/autoresearch_loop.py` pour les expériences
- **Ne PAS modifier**: data/raw, prepare.py (s'il existe), requirements.txt sans raison
- **Commit pattern**: `git commit -m "Iteration N: feature_name - short description"`
- **Logs**: JAMAIS committer `logs/iterations.jsonl` ou `plots/` sur git — ce sont des données/outputs
- **Autonomie**: L'agent peut proposer des features différentes si les suggérées semblent sub-optimales

---

## Timeline Estimée

- Setup: 5 min
- Baseline: 5 min
- Itération moyenne: 2-3 min
- **Total 7 itérations**: ~30 min
- **Total 10 itérations**: ~40 min

Agent peut fonctionner 100% autonome une fois que le user dit "go".
