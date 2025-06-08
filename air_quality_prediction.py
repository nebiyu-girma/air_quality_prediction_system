import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create directory for plots
os.makedirs('plots', exist_ok=True)

# ================================
# 1. DATA GENERATION AND PREPROCESSING
# ================================

def generate_air_quality_data(n_samples=5000):
    """Generate synthetic air quality data based on real-world patterns"""
    data = {
        'temperature': np.random.normal(25, 10, n_samples),
        'humidity': np.random.uniform(20, 90, n_samples),
        'wind_speed': np.random.exponential(3, n_samples),
        'pressure': np.random.normal(1015, 15, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'traffic_density': np.random.uniform(0, 100, n_samples),
        'industrial_activity': np.random.uniform(0, 100, n_samples),
        'population_density': np.random.uniform(100, 10000, n_samples),
        'distance_to_industrial': np.random.exponential(5, n_samples),
        'green_space_ratio': np.random.uniform(0, 0.4, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic AQI
    aqi_base = (
        50 + 
        df['traffic_density'] * 0.8 +
        df['industrial_activity'] * 0.6 +
        (100 - df['humidity']) * 0.3 +
        (30 - df['temperature']) * 0.2 +
        (5 - df['wind_speed']) * 2 +
        df['population_density'] * 0.01 +
        (10 - df['distance_to_industrial']) * 3 +
        (0.3 - df['green_space_ratio']) * 100 +
        np.where(df['hour'].isin([7, 8, 17, 18, 19]), 20, 0) +
        np.where(df['day_of_week'] < 5, 10, -5) +
        np.where(df['month'].isin([12, 1, 2]), 15, 0)
    )
    
    df['aqi'] = np.clip(aqi_base + np.random.normal(0, 15, n_samples), 0, 500)
    
    df['aqi_category'] = pd.cut(df['aqi'], 
                               bins=[0, 50, 100, 150, 200, 300, 500],
                               labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 
                                      'Unhealthy', 'Very Unhealthy', 'Hazardous'])
    
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    # Feature engineering
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    df['traffic_industrial_combined'] = df['traffic_density'] * 0.7 + df['industrial_activity'] * 0.3
    df['is_rush_hour'] = df['hour'].isin([7, 8, 17, 18, 19]).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    return df

# ================================
# 2. EXPLORATORY DATA ANALYSIS
# ================================

def perform_eda(df):
    """Perform comprehensive exploratory data analysis"""
    print("=== AIR QUALITY DATA ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"AQI range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}")
    print("\nAQI Distribution by Category:")
    print(df['aqi_category'].value_counts())
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # AQI distribution
    axes[0,0].hist(df['aqi'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('AQI Distribution')
    
    # AQI by category
    df['aqi_category'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightcoral')
    axes[0,1].set_title('AQI Categories Distribution')
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=axes[0,2])
    axes[0,2].set_title('Feature Correlation Matrix')
    
    # AQI vs Traffic Density
    axes[1,0].scatter(df['traffic_density'], df['aqi'], alpha=0.3, color='green')
    axes[1,0].set_title('AQI vs Traffic Density')
    
    # AQI by hour of day
    hourly_aqi = df.groupby('hour')['aqi'].mean()
    axes[1,1].plot(hourly_aqi.index, hourly_aqi.values, marker='o', color='purple')
    axes[1,1].set_title('Average AQI by Hour of Day')
    axes[1,1].grid(True, alpha=0.3)
    
    # AQI vs Wind Speed
    axes[1,2].scatter(df['wind_speed'], df['aqi'], alpha=0.3, color='orange')
    axes[1,2].set_title('AQI vs Wind Speed')
    
    plt.tight_layout()
    plt.savefig('plots/eda_analysis.png')
    plt.close()
    
    return corr_matrix

# ================================
# 3. MACHINE LEARNING MODELS
# ================================

class AirQualityPredictor:
    """Comprehensive Air Quality Prediction System"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42, max_depth=5),
            'Support Vector Regression': SVR(kernel='rbf', C=10)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.model_scores = {}
        
    def prepare_features(self, df):
        """Prepare features for modeling"""
        feature_cols = [
            'temperature', 'humidity', 'wind_speed', 'pressure',
            'hour', 'day_of_week', 'month',
            'traffic_density', 'industrial_activity', 'population_density',
            'distance_to_industrial', 'green_space_ratio',
            'temp_humidity_interaction', 'traffic_industrial_combined',
            'is_rush_hour', 'is_weekend', 'is_winter'
        ]
        
        X = df[feature_cols]
        y = df['aqi']
        
        return X, y, feature_cols
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate multiple models"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        print("\n=== MODEL TRAINING AND EVALUATION ===")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name == 'Support Vector Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            if name == 'Support Vector Regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R²: {r2:.3f}")
            print(f"CV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_model = results[best_model_name]['model']
        self.model_scores = results
        
        print(f"\n🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['r2']:.3f})")
        
        return results
    
    def analyze_feature_importance(self, X, feature_names):
        """Analyze feature importance for tree-based models"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
            plt.title('Top 10 Most Important Features for AQI Prediction')
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png')
            plt.close()
            
            return feature_importance
        else:
            print("Feature importance not available for this model type.")
            return None

# ================================
# 4. UNSUPERVISED LEARNING - CLUSTERING
# ================================

def perform_clustering_analysis(df):
    """Perform clustering analysis to identify pollution patterns"""
    print("\n=== CITY CLUSTERING ANALYSIS ===")
    
    # Select features for clustering
    cluster_features = ['aqi', 'traffic_density', 'industrial_activity', 
                       'population_density', 'green_space_ratio']
    
    X_cluster = df[cluster_features]
    
    # Standardize features
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    # Determine optimal number of clusters
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/clustering_elbow.png')
    plt.close()
    
    # Perform clustering
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Analyze clusters
    print(f"\nClustering Results (k={optimal_k}):")
    cluster_summary = df_clustered.groupby('cluster')[cluster_features].mean()
    print(cluster_summary)
    
    # Visualize clusters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # AQI vs Traffic Density by cluster
    for cluster in range(optimal_k):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        axes[0,0].scatter(cluster_data['traffic_density'], cluster_data['aqi'], 
                         alpha=0.5, label=f'Cluster {cluster}')
    axes[0,0].set_xlabel('Traffic Density')
    axes[0,0].set_ylabel('AQI')
    axes[0,0].legend()
    
    # AQI vs Industrial Activity by cluster
    for cluster in range(optimal_k):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        axes[0,1].scatter(cluster_data['industrial_activity'], cluster_data['aqi'], 
                         alpha=0.5, label=f'Cluster {cluster}')
    axes[0,1].set_xlabel('Industrial Activity')
    axes[0,1].set_ylabel('AQI')
    axes[0,1].legend()
    
    # Population Density vs Green Space by cluster
    for cluster in range(optimal_k):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        axes[1,0].scatter(cluster_data['population_density'], cluster_data['green_space_ratio'], 
                         alpha=0.5, label=f'Cluster {cluster}')
    axes[1,0].set_xlabel('Population Density')
    axes[1,0].set_ylabel('Green Space Ratio')
    axes[1,0].legend()
    
    # Cluster distribution
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    axes[1,1].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
    axes[1,1].set_xlabel('Cluster')
    axes[1,1].set_ylabel('Number of Areas')
    
    plt.tight_layout()
    plt.savefig('plots/cluster_analysis.png')
    plt.close()
    
    # Interpret clusters
    print("\n=== CLUSTER INTERPRETATION ===")
    for i in range(optimal_k):
        cluster_data = cluster_summary.iloc[i]
        print(f"\nCluster {i}:")
        print(f"  Average AQI: {cluster_data['aqi']:.1f}")
        print(f"  Traffic Density: {cluster_data['traffic_density']:.1f}")
        print(f"  Industrial Activity: {cluster_data['industrial_activity']:.1f}")
        print(f"  Population Density: {cluster_data['population_density']:.0f}")
        print(f"  Green Space Ratio: {cluster_data['green_space_ratio']:.2f}")
        
        if cluster_data['aqi'] > 150:
            print("  → HIGH POLLUTION ZONE: Immediate intervention needed")
        elif cluster_data['aqi'] > 100:
            print("  → MODERATE POLLUTION: Implement pollution controls")
        else:
            print("  → LOW POLLUTION: Maintain current practices")
    
    return df_clustered

# ================================
# 5. ETHICAL CONSIDERATIONS
# ================================

def analyze_ethical_implications(df, model_results):
    """Analyze ethical implications and potential biases"""
    print("\n=== ETHICAL ANALYSIS ===")
    
    # Data representation
    print("\n1. DATA REPRESENTATION:")
    print(f"   - Dataset size: {len(df)} samples")
    print(f"   - AQI range coverage: {df['aqi'].min():.1f} to {df['aqi'].max():.1f}")
    
    # Model fairness
    print("\n2. MODEL FAIRNESS:")
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['r2'])
    predictions = model_results[best_model_name]['predictions']
    
    # Add predictions for error analysis
    df_temp = df.copy()
    df_temp = df_temp.iloc[:len(predictions)]
    df_temp['predicted_aqi'] = predictions
    df_temp['prediction_error'] = abs(df_temp['aqi'] - df_temp['predicted_aqi'])
    
    # Analyze errors by AQI category
    error_by_category = df_temp.groupby('aqi_category')['prediction_error'].agg(['mean', 'std'])
    print("\nPrediction Error by AQI Category:")
    print(error_by_category)
    
    # Recommendations
    print("\n3. ETHICAL DEPLOYMENT RECOMMENDATIONS:")
    print("   ✅ Regular model retraining with local data")
    print("   ✅ Transparent communication of prediction uncertainty")
    print("   ✅ Continuous monitoring for bias and fairness")
    print("   ✅ Accessibility of predictions to all community members")

# ================================
# 6. SUSTAINABILITY IMPACT
# ================================

def assess_sustainability_impact():
    """Assess the sustainability impact of the AI solution"""
    print("\n=== SUSTAINABILITY IMPACT ASSESSMENT ===")
    print("🌍 DIRECT SDG CONTRIBUTIONS:")
    print("   SDG 3: Reduce health impacts of air pollution")
    print("   SDG 11: Enable data-driven urban planning")
    print("   SDG 13: Support climate action through emission reduction")
    
    print("\n🎯 EXPECTED OUTCOMES:")
    print("   • 15-25% reduction in pollution exposure through early warnings")
    print("   • Improved public health in urban areas")
    print("   • Evidence-based policy making")
    
    print("\n📊 SUCCESS METRICS:")
    print("   • Model accuracy (RMSE < 20 AQI points)")
    print("   • Public health impact (reduced respiratory illness rates)")
    print("   • Policy adoption rate by city governments")

# ================================
# 7. MAIN EXECUTION FUNCTION
# ================================

def main():
    """Main function to execute the complete project pipeline"""
    print("🌍 SDG 11: SUSTAINABLE CITIES - AIR QUALITY PREDICTION SYSTEM")
    print("=" * 60)
    
    # Step 1: Generate and preprocess data
    print("\n📊 Step 1: Data Generation and Preprocessing")
    df = generate_air_quality_data(n_samples=5000)
    df = preprocess_data(df)
    print(f"✅ Generated {len(df)} samples with {len(df.columns)} features")
    
    # Step 2: Exploratory Data Analysis
    print("\n🔍 Step 2: Exploratory Data Analysis")
    correlation_matrix = perform_eda(df)
    print("✅ EDA complete - check 'plots' directory for visualizations")
    
    # Step 3: Supervised Learning
    print("\n🤖 Step 3: Supervised Learning - AQI Prediction")
    predictor = AirQualityPredictor()
    X, y, feature_names = predictor.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    model_results = predictor.train_models(X_train, y_train, X_test, y_test)
    
    # Feature importance
    feature_importance = predictor.analyze_feature_importance(X, feature_names)
    
    # Step 4: Unsupervised Learning
    print("\n🎯 Step 4: Unsupervised Learning - Urban Area Clustering")
    df_clustered = perform_clustering_analysis(df)
    print("✅ Clustering complete - check 'plots' directory for results")
    
    # Step 5: Ethical Analysis
    print("\n⚖️  Step 5: Ethical Considerations")
    analyze_ethical_implications(df, model_results)
    
    # Step 6: Sustainability Impact
    print("\n🌱 Step 6: Sustainability Impact Assessment")
    assess_sustainability_impact()
    
    # Step 7: Demo and Summary
    print("\n🚀 PROJECT SUMMARY:")
    best_model = max(model_results.keys(), key=lambda k: model_results[k]['r2'])
    best_r2 = model_results[best_model]['r2']
    print(f"✅ Best model: {best_model} (R² = {best_r2:.3f})")
    print("✅ Full pipeline executed successfully")
    print("✅ Visualizations saved to 'plots' directory")
    
    print("\n📋 NEXT STEPS:")
    print("   1. Deploy model as web application")
    print("   2. Integrate real-time data feeds")
    print("   3. Partner with city governments for implementation")
    
    return predictor, feature_names

# ================================
# 8. DEMO FUNCTION
# ================================

def run_prediction_demo(predictor, feature_names):
    """Interactive demonstration of the prediction model"""
    print("\n🎪 INTERACTIVE PREDICTION DEMO")
    print("=" * 40)
    
    # Create sample scenarios
    scenarios = [
        {
            'name': 'Rush Hour - High Traffic',
            'values': [22, 65, 2.5, 1012, 8, 1, 6, 85, 45, 5000, 2, 0.15, 44.5, 130, 1, 0, 0]
        },
        {
            'name': 'Weekend - Low Activity',
            'values': [25, 55, 5.0, 1018, 14, 6, 6, 30, 20, 2000, 8, 0.25, 36.25, 50, 0, 1, 0]
        },
        {
            'name': 'Industrial Zone - High Pollution',
            'values': [20, 70, 1.5, 1010, 12, 3, 1, 95, 80, 8000, 0.5, 0.10, 35.0, 175, 0, 0, 1]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📍 Scenario: {scenario['name']}")
        
        # Create prediction
        sample_data = pd.DataFrame([scenario['values']], columns=feature_names)
        predicted_aqi = predictor.best_model.predict(sample_data)[0]
        
        print(f"   Predicted AQI: {predicted_aqi:.1f}")
        
        # Interpret result
        if predicted_aqi <= 50:
            category = "Good"
            advice = "Air quality is satisfactory"
        elif predicted_aqi <= 100:
            category = "Moderate"
            advice = "Sensitive individuals should limit outdoor activities"
        elif predicted_aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
            advice = "Sensitive groups should avoid outdoor activities"
        elif predicted_aqi <= 200:
            category = "Unhealthy"
            advice = "Everyone should limit outdoor activities"
        else:
            category = "Very Unhealthy"
            advice = "Health warnings - avoid outdoor activities"
        
        print(f"   Category: {category}")
        print(f"   Recommendation: {advice}")

if __name__ == "__main__":
    predictor, feature_names = main()
    run_prediction_demo(predictor, feature_names)
    
