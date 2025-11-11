import pandas as pd
import numpy as np

# ==============================================================================
# КОНФІГУРАЦІЯ ШЛЯХІВ 
# ==============================================================================

APPLICATION_PATH = 'application_record.csv'
CREDIT_PATH = 'credit_record.csv'
OUTPUT_PATH = 'combined_loan_data.csv'
TARGET_COLUMN = 'TARGET'

# ==============================================================================
# ЛОГІКА СТВОРЕННЯ ЦІЛЬОВОЇ ЗМІННОЇ
# ==============================================================================

def derive_target_variable(credit_df: pd.DataFrame) -> pd.Series:
    """
    Створює цільову змінну (TARGET) на основі кредитної історії.
    TARGET = 1 (Високий Ризик / Дефолт): Якщо клієнт мав статус 2, 3, 4, або 5.
    TARGET = 0 (Низький Ризик): В іншому випадку.
    """
    RISK_STATUSES = ['2', '3', '4', '5']
    credit_df['RISK_FLAG'] = credit_df['STATUS'].apply(lambda x: 1 if str(x) in RISK_STATUSES else 0)
    
    # Групування за ID клієнта для визначення, чи мав він коли-небудь ризиковий статус
    target_series = credit_df.groupby('ID')['RISK_FLAG'].max().rename(TARGET_COLUMN)
    
    print(f"Цільова змінна створена для {len(target_series)} унікальних клієнтів.")
    
    return target_series

def merge_and_prepare_data(app_df: pd.DataFrame, target_series: pd.Series) -> pd.DataFrame:
    """
    Об'єднує дані заявки з цільовою змінною.
    """
    # Використовуємо inner join, щоб залишити лише тих клієнтів, 
    # які присутні в обох файлах і мають кредитну історію.
    combined_df = app_df.merge(target_series, on='ID', how='inner')
    
    # Видаляємо дублікати (якщо клієнт подавав кілька заявок у application_record) 
    combined_df = combined_df.drop_duplicates(subset=['ID'])
    
    print(f"\nРезультат об'єднання: {len(combined_df)} кінцевих записів.")
    print(f"Розподіл цільової змінної:\n{combined_df[TARGET_COLUMN].value_counts(normalize=True)}")
    
    return combined_df


# ==============================================================================
# ОСНОВНА ФУНКЦІЯ
# ==============================================================================
 
if __name__ == '__main__':
    
    try:
        application_df = pd.read_csv(APPLICATION_PATH)
        credit_df = pd.read_csv(CREDIT_PATH)
        print("--- Завантаження даних ---")
        
    except FileNotFoundError:
        print("--- Файли не знайдені ---")
        print("Будь ласка, переконайтеся, що 'application_record.csv' та 'credit_record.csv' знаходяться в робочій директорії.")
        exit()

    target_series = derive_target_variable(credit_df)
    
    # Об'єднання даних
    final_df = merge_and_prepare_data(application_df, target_series)
    
    # Збереження результату
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nГотовий об'єднаний файл збережено як '{OUTPUT_PATH}'.")