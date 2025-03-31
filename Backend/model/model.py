import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# from google.colab import files, drive
from datetime import datetime

# Mount Google Drive (optional)
# drive.mount('/content/drive')

# Set random seed for reproducibility
np.random.seed(42)

try:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    Tk().withdraw()  # Hide the root window
    file_path = askopenfilename(title="Select your marketing campaign dataset", filetypes=[("CSV Files", "*.csv")])

    if not file_path:
        raise FileNotFoundError("No file selected. Please select a CSV file.")

    print(f"Using dataset: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")

except ImportError:
    print("Tkinter is not available. Please manually specify the file path.")

except FileNotFoundError as e:
    print(e)

# Load data
def load_data(file_path):
    """Load the marketing campaign dataset"""
    df = pd.read_csv(file_path)

    # Convert Acquisition_Cost from string to float
    df['Acquisition_Cost'] = df['Acquisition_Cost'].str.replace('$', '').str.replace(',', '').astype(float)

    # Convert Duration to days
    df['Duration_Days'] = df['Duration'].str.extract('(\d+)').astype(int)

    # Extract date components
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    return df

# Load the dataset
df = load_data(file_path)

# Display dataset info
print("\nDataset info:")
print(df.info())

print("\nSample of the dataset:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

# Plot distributions of target variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Conversion_Rate'], kde=True, ax=axes[0])
axes[0].set_title('Conversion Rate Distribution')

sns.histplot(df['ROI'], kde=True, ax=axes[1])
axes[1].set_title('ROI Distribution')

sns.histplot(df['Engagement_Score'], kde=True, ax=axes[2])
axes[2].set_title('Engagement Score Distribution')

plt.tight_layout()
plt.show()

# Analyze channel performance
plt.figure(figsize=(12, 6))
sns.boxplot(x='Channel_Used', y='Conversion_Rate', data=df)
plt.title('Conversion Rate by Channel')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Channel_Used', y='ROI', data=df)
plt.title('ROI by Channel')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze target audience performance
plt.figure(figsize=(14, 6))
sns.boxplot(x='Target_Audience', y='Conversion_Rate', data=df)
plt.title('Conversion Rate by Target Audience')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze campaign type performance
plt.figure(figsize=(12, 6))
sns.boxplot(x='Campaign_Type', y='ROI', data=df)
plt.title('ROI by Campaign Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()

# Data preprocessing
def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Features and target variables
    features = df[['Campaign_Type', 'Target_Audience', 'Duration_Days', 'Channel_Used',
                  'Location', 'Language', 'Customer_Segment', 'Month', 'Year']]

    # We'll create models to predict these performance metrics
    targets = {
        'conversion_rate': df['Conversion_Rate'],
        'roi': df['ROI'],
        'engagement': df['Engagement_Score']
    }

    # Define categorical and numerical features
    categorical_features = ['Campaign_Type', 'Target_Audience', 'Channel_Used',
                           'Location', 'Language', 'Customer_Segment']
    numerical_features = ['Duration_Days', 'Month', 'Year']

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    return features, targets, preprocessor

# Preprocess the data
features, targets, preprocessor = preprocess_data(df)

# Print shapes
print(f"Features shape: {features.shape}")
for target_name, target_values in targets.items():
    print(f"{target_name} shape: {target_values.shape}")

# Model building
def build_models(features, targets, preprocessor):
    """Build and train prediction models for each target variable"""
    # Split the data into training and testing sets
    X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}

    # Create train/test splits for each target
    for target_name, target_values in targets.items():
        X_train[target_name], X_test[target_name], y_train_dict[target_name], y_test_dict[target_name] = \
            train_test_split(features, target_values, test_size=0.2, random_state=42)

    # Build models for each target
    models = {}
    for target_name in targets.keys():
        print(f"\nTraining model for {target_name}...")

        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(random_state=42))
        ])

        # Define parameter grid for hyperparameter tuning
        # Using a small grid for faster execution in Colab
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5]
        }

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        # Train the model
        grid_search.fit(X_train[target_name], y_train_dict[target_name])

        # Save the best model
        models[target_name] = grid_search.best_estimator_

        # Evaluate the model
        y_pred = models[target_name].predict(X_test[target_name])
        mse = mean_squared_error(y_test_dict[target_name], y_pred)
        r2 = r2_score(y_test_dict[target_name], y_pred)

        print(f"{target_name} model - MSE: {mse:.4f}, R²: {r2:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")

    return models, X_test, y_test_dict

# Build and train the models
models, X_test, y_test_dict = build_models(features, targets, preprocessor)


# Feature importance analysis
def analyze_feature_importance(models, features):
    """Analyze and visualize feature importance for each model"""
    for target_name, model in models.items():
        print(f"\nFeature importance for {target_name} model:")

        # Get feature importances (for gradient boosting)
        importances = model.named_steps['model'].feature_importances_

        # Extract the fitted preprocessor from the model's pipeline
        fitted_preprocessor = model.named_steps['preprocessor']

        # Get feature names using the fitted preprocessor
        feature_names = fitted_preprocessor.get_feature_names_out()

        # Create a DataFrame with feature names and importances
        # Ensure the lengths match
        min_len = min(len(importances), len(feature_names))
        importance_df = pd.DataFrame({
            'Feature': feature_names[:min_len],
            'Importance': importances[:min_len]
        }).sort_values('Importance', ascending=False)

        # Display top 15 features
        print(importance_df.head(15))

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title(f'Top 15 Feature Importance for {target_name}')
        plt.tight_layout()
        plt.show()

# Analyze feature importance
analyze_feature_importance(models, features)

# Save models
def save_models(models, preprocessor, file_prefix='marketing_campaign_model'):
    """Save the trained models and preprocessor"""
    # Save to Google Drive if mounted, otherwise save locally
    drive_path = '/content/drive/MyDrive/marketing_models/'
    local_path = '/content/'

    # Check if Drive is mounted
    try:
        import os
        if os.path.exists('/content/drive/MyDrive'):
            save_path = drive_path
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving models to Google Drive: {save_path}")
        else:
            save_path = local_path
            print(f"Saving models locally: {save_path}")
    except:
        save_path = local_path
        print(f"Saving models locally: {save_path}")

    # Save each model
    for target_name, model in models.items():
        model_path = f"{save_path}{file_prefix}_{target_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved {target_name} model to {model_path}")

    # Save the preprocessor
    preprocessor_path = f"{save_path}{file_prefix}_preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")

# Save the models
save_models(models, preprocessor)

# Campaign recommendation function
def recommend_campaign(models, preprocessor, campaign_info):
    """
    Recommend the best channel and target audience for a campaign based on inputs

    Parameters:
    campaign_info - dict with campaign attributes like Campaign_Type, Duration_Days, etc.

    Returns:
    DataFrame with recommendations
    """
    # Available channels and audiences
    channels = ['Google Ads', 'YouTube', 'Instagram', 'Website', 'Facebook', 'Email']
    audiences = ['Men 18-24', 'Men 25-34', 'Women 25-34', 'Women 35-44', 'All Ages']

    # Prepare a DataFrame to test different combinations
    test_combinations = []
    for channel in channels:
        for audience in audiences:
            # Create a copy of campaign_info
            test_case = campaign_info.copy()
            test_case['Channel_Used'] = channel
            test_case['Target_Audience'] = audience
            test_combinations.append(test_case)

    # Convert to DataFrame
    test_df = pd.DataFrame(test_combinations)

    # Make predictions for each target
    results = []
    for i, combo in test_df.iterrows():
        combo_df = pd.DataFrame([combo])

        pred_conversion = models['conversion_rate'].predict(combo_df)[0]
        pred_roi = models['roi'].predict(combo_df)[0]
        pred_engagement = models['engagement'].predict(combo_df)[0]

        # Calculate weighted score based on user's priorities
        conversion_weight = campaign_info.get('conversion_weight', 1.0)
        roi_weight = campaign_info.get('roi_weight', 1.0)
        engagement_weight = campaign_info.get('engagement_weight', 1.0)

        overall_score = (
            pred_conversion * conversion_weight +
            pred_roi * roi_weight +
            pred_engagement * engagement_weight
        )

        results.append({
            'Channel_Used': combo['Channel_Used'],
            'Target_Audience': combo['Target_Audience'],
            'Predicted_Conversion_Rate': pred_conversion,
            'Predicted_ROI': pred_roi,
            'Predicted_Engagement': pred_engagement,
            'Overall_Score': overall_score
        })

    # Convert to DataFrame and find the best combinations
    results_df = pd.DataFrame(results)

    # Sort by overall score (descending)
    results_df = results_df.sort_values('Overall_Score', ascending=False)

    return results_df

# Test the recommendation function with an example campaign
example_campaign = {
    'Campaign_Type': 'Social Media',
    'Customer_Segment': 'Tech Enthusiasts',
    'Location': 'New York',
    'Language': 'English',
    'Duration_Days': 30,
    'Month': datetime.now().month,  # Current month
    'Year': datetime.now().year,    # Current year
    'conversion_weight': 1.5,       # Higher priority on conversion
    'roi_weight': 1.0,              # Default priority on ROI
    'engagement_weight': 0.8        # Lower priority on engagement
}

# Get recommendations
recommendations = recommend_campaign(models, preprocessor, example_campaign)

# Display top 5 recommendations
print("\nTop 5 recommended configurations:")
print(recommendations.head(5))

# Visualize the recommendations
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Predicted_Conversion_Rate',
    y='Predicted_ROI',
    size='Predicted_Engagement',
    hue='Channel_Used',
    style='Target_Audience',
    sizes=(100, 300),
    data=recommendations.head(10)
)
plt.title('Top 10 Recommendations - Conversion Rate vs ROI')
plt.xlabel('Predicted Conversion Rate')
plt.ylabel('Predicted ROI')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Function to create a custom recommendation
def custom_recommendation():
    print("\nCustom Campaign Recommendation")
    print("==============================")

    # Get user inputs
    print("\nPlease enter campaign details:")

    # Campaign Type options
    campaign_types = ['Email', 'Influencer', 'Display', 'Search', 'Social Media']
    print("Campaign Type options:")
    for i, ct in enumerate(campaign_types):
        print(f"{i+1}. {ct}")
    campaign_type_idx = int(input("Enter Campaign Type number: ")) - 1
    campaign_type = campaign_types[campaign_type_idx]

    # Customer Segment options
    segments = ['Health & Wellness', 'Fashionistas', 'Outdoor Adventurers', 'Foodies', 'Tech Enthusiasts']
    print("\nCustomer Segment options:")
    for i, seg in enumerate(segments):
        print(f"{i+1}. {seg}")
    segment_idx = int(input("Enter Customer Segment number: ")) - 1
    customer_segment = segments[segment_idx]

    # Location options
    locations = ['New York', 'Los Angeles', 'Chicago', 'Miami', 'Houston']
    print("\nLocation options:")
    for i, loc in enumerate(locations):
        print(f"{i+1}. {loc}")
    location_idx = int(input("Enter Location number: ")) - 1
    location = locations[location_idx]

    # Language options
    languages = ['English', 'Spanish', 'French', 'German', 'Mandarin']
    print("\nLanguage options:")
    for i, lang in enumerate(languages):
        print(f"{i+1}. {lang}")
    language_idx = int(input("Enter Language number: ")) - 1
    language = languages[language_idx]

    # Duration
    duration_days = int(input("\nEnter campaign duration in days (7-90): "))

    # Weights for metrics
    print("\nSet priorities for optimization metrics (0.5-2.0):")
    conversion_weight = float(input("Conversion Rate importance (default 1.0): ") or "1.0")
    roi_weight = float(input("ROI importance (default 1.0): ") or "1.0")
    engagement_weight = float(input("Engagement importance (default 1.0): ") or "1.0")

    # Create campaign info dictionary
    campaign_info = {
        'Campaign_Type': campaign_type,
        'Customer_Segment': customer_segment,
        'Location': location,
        'Language': language,
        'Duration_Days': duration_days,
        'Month': datetime.now().month,
        'Year': datetime.now().year,
        'conversion_weight': conversion_weight,
        'roi_weight': roi_weight,
        'engagement_weight': engagement_weight
    }

    # Get recommendations
    recommendations = recommend_campaign(models, preprocessor, campaign_info)

    # Display top 5 recommendations
    print("\nTop 5 recommended configurations for your campaign:")
    recommendations_display = recommendations.head(5).copy()
    recommendations_display['Predicted_Conversion_Rate'] = recommendations_display['Predicted_Conversion_Rate'].map('{:.2%}'.format)
    recommendations_display['Predicted_ROI'] = recommendations_display['Predicted_ROI'].map('{:.2f}x'.format)
    recommendations_display['Predicted_Engagement'] = recommendations_display['Predicted_Engagement'].map('{:.1f}/10'.format)
    recommendations_display['Overall_Score'] = recommendations_display['Overall_Score'].map('{:.2f}'.format)

    print(recommendations_display[['Channel_Used', 'Target_Audience', 'Predicted_Conversion_Rate',
                            'Predicted_ROI', 'Predicted_Engagement', 'Overall_Score']])

    # Visualize the recommendations
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Predicted_Conversion_Rate',
        y='Predicted_ROI',
        size='Predicted_Engagement',
        hue='Channel_Used',
        style='Target_Audience',
        sizes=(100, 300),
        data=recommendations.head(10)
    )
    plt.title('Top 10 Recommendations - Conversion Rate vs ROI')
    plt.xlabel('Predicted Conversion Rate')
    plt.ylabel('Predicted ROI')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Save recommendations to CSV
    csv_filename = 'campaign_recommendations.csv'
    recommendations.to_csv(csv_filename, index=False)
    print(f"\nRecommendations saved to {csv_filename}")
    files.download(csv_filename)

    return recommendations

# Interactive recommendation
print("\nWould you like to get custom recommendations for a campaign?")
response = input("Enter 'y' for Yes, any other key to skip: ")

if response.lower() == 'y':
    custom_recommendations = custom_recommendation()

print("\nThank you for using the Marketing Campaign Optimizer!")
print("You can now download the trained models and use them in your applications.")
print("The models can be loaded using joblib.load() and used to make predictions for new campaigns.")