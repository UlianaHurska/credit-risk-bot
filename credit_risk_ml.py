import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.impute import SimpleImputer

# ==============================================================================
# КОНФІГУРАЦІЯ
# ==============================================================================


DATA_PATH = 'combined_loan_data.csv'
TARGET_COLUMN = 'TARGET' 

# ==============================================================================
# ОЗНАКИ
# ==============================================================================    
CATEGORICAL_FEATURES = [
    'CODE_GENDER',          
    'FLAG_OWN_CAR',         
    'FLAG_OWN_REALTY',      
    'NAME_INCOME_TYPE',     
    'NAME_EDUCATION_TYPE',  
    'NAME_FAMILY_STATUS',   
    'NAME_HOUSING_TYPE',    
    'OCCUPATION_TYPE'       
] 

# Фінальний список числових ознак після трансформації DAYS_BIRTH та DAYS_EMPLOYED
NUMERICAL_FEATURES_FINAL = ['AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AGE', 'YEARS_EMPLOYED']

# ==============================================================================
# ОСНОВНА ЛОГІКА ЗАВАНТАЖЕННЯ ТА ТРЕНУВАННЯ
# ==============================================================================

def load_data(path):
    """
    Завантажує дані та виконує feature engineering
    для перетворення днів на роки.
    """
    try:
        df = pd.read_csv(path)
        print(f"Дані завантажено з {path}. Кількість записів: {len(df)}")
        
        COLS_TO_DROP = ['ID', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL', 'MONTHS_BALANCE', 'STATUS']
        df = df.drop(columns=COLS_TO_DROP, errors='ignore')
        
        # Вік (DAYS_BIRTH)
        if 'DAYS_BIRTH' in df.columns:
            df['AGE'] = df['DAYS_BIRTH'].abs() / 365.25
            
        # Робочий стаж (DAYS_EMPLOYED)
        if 'DAYS_EMPLOYED' in df.columns:
            
            df['YEARS_EMPLOYED'] = np.where(
                df['DAYS_EMPLOYED'] < 0, 
                df['DAYS_EMPLOYED'].abs() / 365.25, 
                0
            )
    
        df = df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED'], errors='ignore')
        
        return df
    
    except FileNotFoundError:
        print(f"ПОМИЛКА: Файл '{path}' не знайдено.")
        print("Будь ласка, запустіть data_merger.py, щоб створити цей файл.")
        return pd.DataFrame()
    
def create_pipeline(X, y):
    """Створює ML-Pipeline для препроцесингу та моделювання."""
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), NUMERICAL_FEATURES_FINAL),
                
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
        
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
        
    ml_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
    ])

    return ml_pipeline

def train_and_evaluate(df):
    """Тренує та оцінює модель."""
    
    # ПЕРЕВІРКА: Чи присутні всі фінальні ознаки
    required_cols = NUMERICAL_FEATURES_FINAL + CATEGORICAL_FEATURES
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"КРИТИЧНА ПОМИЛКА: Вхідні дані не містять обов'язкові ознаки: {missing_cols}")
        return None, None
    
    if TARGET_COLUMN not in df.columns or df.empty:
        print(f"ПОМИЛКА: Цільова змінна '{TARGET_COLUMN}' відсутня в даних.")
        return None, None
    
    X = df[required_cols]
    y = df[TARGET_COLUMN]

    # Розділення на тренувальний та тестовий набори зі стратифікацією (зберігаємо пропорції класів)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = create_pipeline(X_train, y_train)
    
    # --- ТРЕНУВАННЯ ---
    print("\n--- Запуск тренування ML Pipeline ---")
    pipeline.fit(X_train, y_train)
    
    # Оцінка
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print("\n==================================")
    print(f"ML Pipeline навчено (Logistic Regression).")
    print(f"МЕТРИКИ (Оцінка якості прогнозу ризику):")
    print(f"ROC AUC (чим ближче до 1, тим краще): {auc:.4f}")
    print(f"F1-Score (баланс точності та повноти): {f1:.4f}")
    print("==================================")
    
    # Збереження навченого Pipeline у файл
    joblib.dump(pipeline, 'credit_risk_predictor.pkl')
    print("Pipeline успішно збережено як 'credit_risk_predictor.pkl'")
    
    return pipeline, X_train 

if __name__ == '__main__':
    data = load_data(DATA_PATH)
    if not data.empty:
        pipeline, X_train_full = train_and_evaluate(data)
