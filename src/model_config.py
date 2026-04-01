"""Model configuration - agent can modify this to select model and hyperparameters."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_model():
    """Return configured classifier for current iteration.

    Agent modifies this function to try different models and hyperparameters.
    Must return an unfitted sklearn classifier with fit() and predict_proba() methods.

    Examples:
        LogisticRegression(random_state=42, max_iter=1000)
        RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        SVC(kernel='rbf', probability=True, random_state=42)
        DecisionTreeClassifier(random_state=42, max_depth=5)

    Returns:
        Unfitted sklearn classifier instance.
    """
    return LogisticRegression(random_state=42, max_iter=1000)
