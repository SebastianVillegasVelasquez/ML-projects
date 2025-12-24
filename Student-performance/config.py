import pandas as pd
import os
DATASET_SLUG = "lainguyn123/student-performance-factors"
DATASET_FILE = "StudentPerformanceFactors.csv"


def apply_settings():
    options = {
        'future.no_silent_downcasting': True
    }
    for key, value in options.items():
        pd.set_option(key, value)

if os.environ.get('DISABLE_CONFIG_AUTOAPPLY') != '1':
    apply_settings()