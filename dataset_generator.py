# dataset_generator.py

import pandas as pd
import numpy as np

def generate_alt_data(n=1000):
    np.random.seed(42)

    data = pd.DataFrame({
        'monthly_mobile_recharge': np.random.normal(200, 50, n),
        'call_drop_rate': np.random.uniform(0, 0.2, n),
        'social_media_usage_hrs': np.random.normal(3, 1, n),
        'electricity_bill_paid_on_time': np.random.choice([1, 0], size=n, p=[0.85, 0.15]),
        'gas_bill_paid_on_time': np.random.choice([1, 0], size=n, p=[0.80, 0.20]),
        'app_installs_last_30_days': np.random.poisson(4, n),
        'smartphone_type': np.random.choice(['low_end', 'mid_range', 'premium'], size=n, p=[0.3, 0.5, 0.2]),
        'emergency_contacts_count': np.random.poisson(2, n),
        'geo_location_change_rate': np.random.uniform(0, 0.5, n),
    })

    # Simulate default risk score and convert into binary label
    data['default'] = (
        (data['electricity_bill_paid_on_time'] == 0).astype(int) +
        (data['gas_bill_paid_on_time'] == 0).astype(int) +
        (data['smartphone_type'] == 'low_end').astype(int) +
        (data['geo_location_change_rate'] > 0.3).astype(int)
    )
    data['default'] = (data['default'] >= 2).astype(int)

    return data
