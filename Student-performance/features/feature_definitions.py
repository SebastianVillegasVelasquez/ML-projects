def consistency_score(X):
    """Combines attendance and previous performance to capture study consistency.

    The score is defined as Attendance * Previous_Scores. Expects those
    columns to exist in `X`.

    Args:
        X: pandas.DataFrame

    Returns:
        Series-like object with computed consistency score.
    """
    return X['Attendance'] * X['Previous_Scores']


def motivation_adjusted_study(X):
    """Adjusts study hours by motivation level intensity.

    Maps Motivation_Level from ['Low','Medium','High'] to [1,2,3] and
    multiplies Hours_Studied by that factor.

    Args:
        X: pandas.DataFrame

    Returns:
        Series-like object with motivation adjusted study measure.
    """
    return (
        X['Hours_Studied']
        * X['Motivation_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
    )
