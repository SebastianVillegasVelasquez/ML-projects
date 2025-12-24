# Feature Store Documentation

This document describes all features used in the final predictive model,
including their definition, motivation, and observed impact.

## Core Academic Features

### Attendance
- **Type:** Numerical
- **Source:** Raw dataset
- **Description:** Percentage of classes attended by the student.
- **Rationale:** Strong proxy for engagement and discipline.
- **Impact:** Highest predictive importance across all models.
- **Risks:** None

### Hours_Studied
- **Type:** Numerical
- **Source:** Raw dataset
- **Description:** Total weekly study hours.
- **Rationale:** Represents direct academic effort.
- **Impact:** Second most important feature.
- **Risks:** Self-reported bias.

### Access_to_Resources
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Level of access to educational resources (Low, Medium, High).
- **Rationale:** Facilitates learning opportunities.
- **Impact:** Low–Moderate importance, non-principal signal for prediction.
- **Risks:** Socioeconomic bias.

### Previous_Scores
- **Type:** Numerical (Scaled)
- **Source:** Raw dataset
- **Description:** Average scores from previous assessments.
- **Rationale:** Indicates baseline academic ability.
- **Impact:** Low–Moderate importance, non-principal signal for prediction.
- **Risks:** May reflect prior educational quality.

### Motivation_Level
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Student's self-reported motivation (Low, Medium, High).
- **Rationale:** Influences study habits and performance.
- **Impact:** Low–Moderate importance, non-principal signal for prediction.
- **Risks:** Subjective bias.

### Parental_Involvement
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Frequency of parental engagement in academic activities (Low, Medium, High).
- **Rationale:** Supportive home environment enhances learning.
- **Impact:** Low–Moderate importance, non-principal signal for prediction.
- **Risks:** Cultural differences in reporting.

### Tutoring_Sessions
- **Type:** Numerical (Scaled)
- **Source:** Raw dataset
- **Description:** Number of tutoring sessions attended per month.
- **Rationale:** Additional academic support.
- **Impact:** Low–Moderate importance, non-principal signal for prediction.
- **Risks:** Access inequality.

### Family_Income
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Income bracket of the student's family (Low, Medium, High).
- **Rationale:** Economic stability can affect academic resources.
- **Impact:** Low–Moderate importance, non-principal signal for prediction.
- **Risks:** Socioeconomic bias.

### Parental_Education_Level
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Highest education level attained by parents (High School, College, Postgraduate).
- **Rationale:** Reflects educational support at home.
- **Impact:** Low–Moderate importance, non-principal signal for prediction.
- **Risks:** Cultural bias.

### Peer_Influence
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Influence of peers on academic behavior (Negative, Neutral, Positive).
- **Rationale:** Social environment impacts study habits.
- **Impact:** Low–Moderate importance, non-principal signal for prediction. May interact with Parental_Involvement.
- **Risks:** Subjective bias.

### Teacher_Quality	
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Student's perception of teacher effectiveness (Low, Medium, High).
- **Rationale:** Quality instruction is critical for learning.
- **Impact:** Low–Moderate importance, non-principal signal for prediction.
- **Risks:** Subjective bias.

### Distance_from_Home	
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Distance from home to school (Near, Medium, Far).
- **Rationale:** Affects fatigue and time management.
- **Impact:** Low importance, non-principal signal for prediction.
- **Risks:** Geographic bias.

### Extracurricular_Activities	
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Participation in extracurricular activities (Yes, No).
- **Rationale:** Balances academic and personal development.
- **Impact:** Low importance, non-principal signal for prediction.
- **Risks:** Time management bias.

### Learning_Disabilities
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Presence of diagnosed learning disabilities (Yes, No).
- **Rationale:** Affects learning processes and outcomes.
- **Impact:** Low importance, non-principal signal for prediction.
- **Risks:** Privacy concerns.

### Internet_Access
- **Type:** Categorical (Encoded)
- **Source:** Raw dataset
- **Description:** Availability of internet access at home (Yes, No).
- **Rationale:** Essential for research and assignments.
- **Impact:** Low importance, non-principal signal for prediction. it may interact with Access_to_Resources.
- **Risks:** Socioeconomic bias.

## Engineered Features

### Consistency_Score
- **Formula:** `Attendance * Previous_Scores`
- **Type:** Numerical
- **Motivation:** Captures consistency between effort and past performance.
- **Observed Impact:** Medium–High (Permutation Importance).
- **Usage:** Secondary correction signal.
- **Risks:** Correlated with Attendance.

### Motivation_Adjusted_Study
- **Formula:** `Hours_Studied * Motivation_Level_encoded`
- **Type:** Numerical
- **Motivation:** Adjust raw study time by motivation intensity.
- **Observed Impact:** Moderate.
- **Notes:** Adds non-linear interaction.


## Deprecated Features

### Sleep_Hours
- **Reason:** No significant correlation with performance.
- **Decision:** Removed to reduce noise.
- Notes:** Consider re-evaluation if new data is available.

### Gender
- **Reason:** Ethical concerns and potential bias.
- **Decision:** Excluded from the model.
- **Notes:** Focus on performance-related features only.

### School_Type
- **Reason:** No added predictive value.
- **Decision:** Removed after feature importance analysis.
- **Notes:** May be reconsidered in different contexts.

### Physical_Activity
- **Reason:** Low correlation with academic outcomes.
- **Decision:** Excluded to streamline the feature set.
- **Notes:** Could be revisited with additional health data.