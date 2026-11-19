import os
import numpy as np
import pandas as pd

def generate_house_prices():
    np.random.seed(42)
    n_samples = 1000

    square_feet = np.random.uniform(800, 4500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.round(bedrooms * 0.6 + np.random.uniform(0.5, 1.5, n_samples), 1)
    bathrooms = np.clip(bathrooms, 1.0, 5.0)
    age_years = np.random.randint(1, 51, n_samples)
    distance_city_center = np.round(np.random.uniform(1.0, 30.0, n_samples), 2)
    location_score = np.random.randint(1, 11, n_samples)

    # Realistic price model ($ in thousands):
    # - Baseline price: 50
    # - Size effect: 0.11 * square_feet ($110 per sq ft)
    # - Bedroom bonus: 15 * bedrooms
    # - Bathroom bonus: 20 * bathrooms
    # - Age depreciation: -2.2 * age_years
    # - Distance penalty: -3.5 * distance
    # - Location quality multiplier: 12.0 * location_score
    # - Interaction: (square_feet / 1000) * location_score * 2.5
    # - Gaussian noise: std 18.0
    base_price = 50.0
    size_effect = 0.11 * square_feet
    bed_effect = 15.0 * bedrooms
    bath_effect = 20.0 * bathrooms
    age_effect = -2.2 * age_years
    dist_effect = -3.5 * distance_city_center
    loc_effect = 12.0 * location_score
    interaction = (square_feet / 1000.0) * location_score * 2.5
    noise = np.random.normal(0, 18.0, n_samples)

    price = base_price + size_effect + bed_effect + bath_effect + age_effect + dist_effect + loc_effect + interaction + noise
    price = np.round(np.clip(price, 80.0, 950.0), 2)

    df = pd.DataFrame({
        'Square_Feet': np.round(square_feet, 1),
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age_Years': age_years,
        'Distance_City_Center': distance_city_center,
        'Location_Score': location_score,
        'Price': price
    })

    base_dir = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(base_dir, 'house_prices.csv'), index=False)
    print(f"Generated house_prices.csv with {len(df)} samples.")
    print(df.head())

    # Mini 5-sample dataset for handwritten demonstration
    # Features: Square_Feet (thousands), Age (decades) -> Target: Price ($k)
    sample_df = pd.DataFrame({
        'Square_Feet_k': [1.0, 2.0, 2.0, 3.0, 4.0],
        'Age_Decades':   [1.0, 2.0, 1.0, 3.0, 2.0],
        'Price':         [120.0, 180.0, 210.0, 250.0, 340.0]
    })
    sample_df.to_csv(os.path.join(base_dir, 'sample_5_points.csv'), index=False)
    print("\nSaved sample_5_points.csv:")
    print(sample_df)

if __name__ == '__main__':
    generate_house_prices()
