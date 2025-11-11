import os
import re
import joblib
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv
import unicodedata 
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, 
    ConversationHandler, ContextTypes
)
from messages import MESSAGES, MAPPINGS, REPLY_KEYBOARDS

# ==============================================================================
# КОНФІГУРАЦІЯ
# ==============================================================================
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
MODEL_PATH = 'credit_risk_predictor.pkl'
LOGGING_LEVEL = logging.INFO
USER_DATA_KEY = 'user_input'
ML_MODEL_KEY = 'ml_pipeline'

# ==============================================================================
# НАЛАШТУВАННЯ ЛОГУВАННЯ
# ==============================================================================

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Функція для обробки будь-яких помилок
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Логування помилок і повідомлення користувачу."""
    logger.error("Помилка під час обробки апдейту:", exc_info=context.error)

    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(MESSAGES["bot_error"],
                parse_mode='HTML'
            )
    except Exception as e:
        logger.error(f"Помилка під час відправки повідомлення користувачу: {e}")
        
# ==============================================================================
# ЕТАПИ ДІАЛОГУ 
# ==============================================================================

# 10 ключових ознак, які збираємо
AGE, GENDER, CAR, REALTY, INCOME, INCOME_TYPE, EDUCATION, MARITAL, HOUSING, CHILDREN, OCCUPATION, FAMILY_MEMBERS, EMPLOYED_DAYS = range(13)
END_CONVERSATION = ConversationHandler.END

# Фінальні списки ознак для правильної трансформації даних
CATEGORICAL_FEATURES = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
]
NUMERICAL_FEATURES_FINAL = ['AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AGE', 'YEARS_EMPLOYED']
FEATURE_ORDER_FOR_PREDICTION = NUMERICAL_FEATURES_FINAL + CATEGORICAL_FEATURES 

# ==============================================================================
# ML ЛОГІКА: ЗАВАНТАЖЕННЯ ТА ПРОГНОЗ
# ==============================================================================

def load_ml_assets():
    """Завантажує навчений Pipeline."""
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Файл моделі не знайдено: {MODEL_PATH}")
        return None

    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info(f"ML Pipeline '{MODEL_PATH}' успішно завантажено.")
        return pipeline

    except Exception as e:
        logger.error(f"Помилка завантаження ML-активів: {e}")
        return None
    
def prepare_input_df(user_data: dict) -> pd.DataFrame:
    """Перетворює словник даних користувача у DataFrame для прогнозування."""
    input_df = pd.DataFrame([user_data])
    
    required_cols = NUMERICAL_FEATURES_FINAL + CATEGORICAL_FEATURES
    
    # Перевіряємо, щоб усі необхідні колонки були в DataFrame, 
    # заповнюючи відсутні NaN (якщо такі є).
    for col in required_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan
            
    return input_df[required_cols]


def make_prediction_and_explain(pipeline, user_df):
    """Робить прогноз."""
    
    proba_risk = pipeline.predict_proba(user_df)[:, 1][0]
    
    risk_percent = proba_risk * 100
    
    if risk_percent > 30:
        conclusion = f"‼️ <b>ВИСОКИЙ РИЗИК НЕПОВЕРНЕННЯ ({risk_percent:.1f}%)</b> ‼️\n"
        conclusion += "Наша модель оцінює заявку як високоризиковану. Рекомендовано додаткову перевірку."
    elif risk_percent > 10:
        conclusion = f"⚠️ <b>СЕРЕДНІЙ РИЗИК НЕПОВЕРНЕННЯ ({risk_percent:.1f}%)</b> ⚠️\n"
        conclusion += "Модель бачить певні фактори ризику. Рекомендовано переглянути умови позики."
    else:
        conclusion = f"✅ <b>НИЗЬКИЙ РИЗИК НЕПОВЕРНЕННЯ ({risk_percent:.1f}%)</b> ✅\n"
        conclusion += "Модель оцінює заявку як надійну."
        
    return conclusion

async def run_prediction_pipeline(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Винесена логіка: запускає підготовку даних, прогноз та надсилання результату."""
    
    pipeline = context.application.bot_data[ML_MODEL_KEY]
    
    # 1. Підготовка даних
    user_df = prepare_input_df(context.user_data[USER_DATA_KEY])

    # 2. Отримання прогнозу
    result_text = make_prediction_and_explain(pipeline, user_df)
    
    await update.message.reply_text(
        f"<b>📊 РЕЗУЛЬТАТ ОЦІНКИ РИЗИКУ 📊</b>\n\n{result_text}", 
        parse_mode='HTML', 
        reply_markup=ReplyKeyboardRemove()
    )
    
    return END_CONVERSATION

# ==============================================================================
# ФУНКЦІЯ ДЛЯ ОЧИЩЕННЯ ЦІЛИХ ЧИСЕЛ
# ==============================================================================

def clean_and_convert_int(raw_text: str) -> int:
    """Використовує Unicode нормалізацію та regex для вилучення чистих цифр."""
    if raw_text is None:
        raise ValueError("Input text is None.")
    
    # 1. Нормалізація Unicode (видаляє невидимі символи та конвертує подібні цифри)
    text = unicodedata.normalize("NFKC", str(raw_text))
    
    # 2) Видалити відомі невидимі / проблемні символи
    for ch in ['\u200b', '\u200c', '\u200d', '\ufeff', '\xa0', '\u2060', '\u200e', '\u200f']:
        text = text.replace(ch, '')
    
    text = text.strip()
    
    if not text:
        raise ValueError("Empty input after cleaning")
    
    # 3) Обробка знака
    sign = 1
    if text[0] in ['+', '-']:
        if text[0] == '-':
            sign = -1
        text = text[1:].lstrip()

    if not text:
        raise ValueError("No digits found")
    
    digits = []
    for i, ch in enumerate(text):
        if '0' <= ch <= '9':
            digits.append(ch)
        else:
            # Якщо зустріли нецифровий символ — припиняємо і вважаємо ввід некоректним
            raise ValueError(f"Invalid character in input: {ch!r}")
        
        if not digits:
            raise ValueError("No numeric characters")
        
    return sign * int(''.join(digits))

# ==============================================================================
# КОМАНДИ ТА ДІАЛОГОВІ ХЕНДЛЕРИ
# ==============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Запускає діалог, скидає дані та запитує вік."""
    
    if ML_MODEL_KEY not in context.application.bot_data:
        await update.message.reply_text(MESSAGES["invalid_ml_model"], parse_mode='HTML')
        return END_CONVERSATION
        
    context.user_data[USER_DATA_KEY] = {}

    await update.message.reply_text(MESSAGES["welcome_text"], reply_markup=ReplyKeyboardRemove(), parse_mode='HTML')
    return AGE

async def get_age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє вік (ЗБЕРІГАЄ В РОКАХ) та запитує стать."""
    try:
        global age
        age = clean_and_convert_int(update.message.text)
        if age < 0:
            await update.message.reply_text(
                MESSAGES["age_negative"], parse_mode='HTML')
            return AGE
        
        elif age < 18:
            await update.message.reply_text(
                MESSAGES["age_too_young"], parse_mode='HTML')
            return END_CONVERSATION
        
        elif age > 100:
            await update.message.reply_text(
                MESSAGES["age_too_high"], parse_mode='HTML')
            return AGE
        
        context.user_data[USER_DATA_KEY]['AGE'] = age
        
        reply_keyboard = REPLY_KEYBOARDS['CODE_GENDER']
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)
        
        await update.message.reply_text("Оберіть вашу стать:", reply_markup=markup)
        return GENDER
        
    except ValueError:
        await update.message.reply_text(MESSAGES["invalid_age"], parse_mode='HTML')
        return AGE

async def get_gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє стать та запитує наявність авто."""
    gender = update.message.text
    
    if gender not in MAPPINGS["gender"]:
        await update.message.reply_text(MESSAGES["gender_invalid"], parse_mode='HTML')
        return GENDER
        
    context.user_data[USER_DATA_KEY]['CODE_GENDER'] = MAPPINGS["gender"][gender]
    
    reply_keyboard = REPLY_KEYBOARDS['FLAG_OWN_CAR']
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

    await update.message.reply_text(MESSAGES["car_prompt"], reply_markup=markup, parse_mode='HTML')
    return CAR


async def get_car(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє наявність авто та запитує нерухомість."""
    car = update.message.text
    
    if car not in MAPPINGS["car"]:
        await update.message.reply_text(MESSAGES["car_invalid"], parse_mode='HTML')
        return CAR
        
    context.user_data[USER_DATA_KEY]['FLAG_OWN_CAR'] = MAPPINGS["car"][car]
    
    reply_keyboard = REPLY_KEYBOARDS['FLAG_OWN_REALTY']
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

    await update.message.reply_text(MESSAGES["realty_prompt"], reply_markup=markup, parse_mode='HTML')
    return REALTY


async def get_realty(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє нерухомість та запитує річний дохід."""
    realty = update.message.text
    if realty not in MAPPINGS["realty"]:
        await update.message.reply_text(MESSAGES["realty_invalid"], parse_mode='HTML')
        return REALTY

    context.user_data[USER_DATA_KEY]['FLAG_OWN_REALTY'] = MAPPINGS["realty"][realty]

    await update.message.reply_text(MESSAGES["income_prompt"], reply_markup=ReplyKeyboardRemove(), parse_mode='HTML')
    return INCOME

async def get_income(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє дохід та запитує тип доходу."""
    try:
        income = clean_and_convert_int(update.message.text)
        
        if income == 0:
            await update.message.reply_text(MESSAGES["income_zero"], parse_mode='HTML')
            return INCOME
        
        elif income < 0:
            await update.message.reply_text(MESSAGES["income_minus"], parse_mode='HTML')
            return INCOME
            
        context.user_data[USER_DATA_KEY]['AMT_INCOME_TOTAL'] = income
        
        reply_keyboard = REPLY_KEYBOARDS['NAME_INCOME_TYPE']
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

        await update.message.reply_text(MESSAGES["income_type_prompt"], reply_markup=markup, parse_mode='HTML')
        return INCOME_TYPE
        
    except ValueError:
        await update.message.reply_text(MESSAGES["income_invalid"], parse_mode='HTML')
        return INCOME

async def get_income_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє тип доходу та запитує освіту."""
    income_type = update.message.text
    valid_options = [item for sublist in REPLY_KEYBOARDS['NAME_INCOME_TYPE'] for item in sublist]
    if income_type not in valid_options:
        await update.message.reply_text(MESSAGES["buttons_massage"], parse_mode='HTML')
        return INCOME_TYPE
        
    context.user_data[USER_DATA_KEY]['NAME_INCOME_TYPE'] = income_type
    
    reply_keyboard = REPLY_KEYBOARDS['NAME_EDUCATION_TYPE']
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

    await update.message.reply_text(MESSAGES["education_prompt"], reply_markup=markup, parse_mode='HTML')
    return EDUCATION

async def get_education(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє освіту та запитує сімейний стан."""
    education = update.message.text
    
    valid_options = [item for sublist in REPLY_KEYBOARDS['NAME_EDUCATION_TYPE'] for item in sublist]
    if education not in valid_options:
        await update.message.reply_text(MESSAGES["buttons_massage"], parse_mode='HTML')
        return EDUCATION
        
    context.user_data[USER_DATA_KEY]['NAME_EDUCATION_TYPE'] = education
    
    reply_keyboard = REPLY_KEYBOARDS['NAME_FAMILY_STATUS']
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

    await update.message.reply_text(MESSAGES["marital_prompt"], reply_markup=markup, parse_mode='HTML')
    return MARITAL


async def get_marital(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє сімейний стан та запитує тип житла."""
    marital = update.message.text
    valid_options = [item for sublist in REPLY_KEYBOARDS['NAME_FAMILY_STATUS'] for item in sublist]
    if marital not in valid_options:
        await update.message.reply_text(MESSAGES["buttons_massage"], parse_mode='HTML')
        return MARITAL
        
    context.user_data[USER_DATA_KEY]['NAME_FAMILY_STATUS'] = marital
    
    reply_keyboard = REPLY_KEYBOARDS['NAME_HOUSING_TYPE']
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

    await update.message.reply_text(MESSAGES["housing_prompt"], reply_markup=markup, parse_mode='HTML')
    return HOUSING


async def get_housing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє тип житла та запитує кількість дітей."""
    housing = update.message.text
    valid_options = [item for sublist in REPLY_KEYBOARDS['NAME_HOUSING_TYPE'] for item in sublist]
    if housing not in valid_options:
        await update.message.reply_text(MESSAGES["buttons_massage"], parse_mode='HTML')
        return HOUSING
        
    context.user_data[USER_DATA_KEY]['NAME_HOUSING_TYPE'] = housing

    await update.message.reply_text(MESSAGES["children_prompt"], reply_markup=ReplyKeyboardRemove(), parse_mode='HTML')                 
    return CHILDREN


async def get_children(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє кількість дітей та запитує професію."""
    try:
        children = clean_and_convert_int(update.message.text)
        
        if children < 0 or children > 20:
            await update.message.reply_text(MESSAGES["children_number"], parse_mode='HTML')
            return CHILDREN
            
        context.user_data[USER_DATA_KEY]['CNT_CHILDREN'] = children
        
        reply_keyboard = REPLY_KEYBOARDS['OCCUPATION_TYPE']
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)

        await update.message.reply_text(MESSAGES["occupation_prompt"], reply_markup=markup, parse_mode='HTML')
        return OCCUPATION
        
    except ValueError:
        await update.message.reply_text(MESSAGES["children_invalid"], parse_mode='HTML')
        return CHILDREN


async def get_occupation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє професію та запитує кількість членів сім'ї."""
    occupation = update.message.text.strip() 
    valid_options = [item for sublist in REPLY_KEYBOARDS['OCCUPATION_TYPE'] for item in sublist]
    if occupation not in valid_options:
        await update.message.reply_text(MESSAGES["buttons_massage"], parse_mode='HTML')
        return OCCUPATION
        
    context.user_data[USER_DATA_KEY]['OCCUPATION_TYPE'] = occupation

    await update.message.reply_text(MESSAGES["family_members_prompt"], parse_mode='HTML')       
    return FAMILY_MEMBERS


async def get_family_members(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє кількість членів сім'ї та запитує стаж роботи."""
    try:
        fam_members = clean_and_convert_int(update.message.text)
        
        if fam_members < 1 or fam_members > 15: 
            await update.message.reply_text(MESSAGES["family_members_number"], parse_mode='HTML')
            return FAMILY_MEMBERS
            
        context.user_data[USER_DATA_KEY]['CNT_FAM_MEMBERS'] = fam_members

        await update.message.reply_text(MESSAGES["employed_years_prompt"], parse_mode='HTML')
        return EMPLOYED_DAYS

    except ValueError:
        await update.message.reply_text(MESSAGES["family_members_invalid"], parse_mode='HTML')
        return FAMILY_MEMBERS


async def get_employed_days(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє стаж роботи та запускає прогноз."""
    try:
        years = clean_and_convert_int(update.message.text)
        max_possible_experience = age - 16
        print(max_possible_experience)
        if years < 0:
             await update.message.reply_text(MESSAGES["employed_years_negative"], parse_mode='HTML')
             return EMPLOYED_DAYS
         
        elif years > 80:
            await update.message.reply_text(MESSAGES["employed_years_too_high"], parse_mode='HTML')
            return EMPLOYED_DAYS
        
        elif years > max_possible_experience:
            await update.message.reply_text(MESSAGES["employed_years_exceed_age"].format(max_possible_experience=max_possible_experience), parse_mode='HTML')
            return EMPLOYED_DAYS
            
        context.user_data[USER_DATA_KEY]['YEARS_EMPLOYED'] = years
        return await run_prediction_pipeline(update, context)

    except ValueError:
        await update.message.reply_text(MESSAGES["employed_years_invalid"], parse_mode='HTML')
        return EMPLOYED_DAYS


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Завершує діалог."""
    await update.message.reply_text(MESSAGES["start_over"], reply_markup=ReplyKeyboardRemove(), parse_mode='HTML')
    
    return END_CONVERSATION


def main() -> None:
    """Запускає бота."""
    
    # 1. Завантаження ML-активів (моделі)
    pipeline = load_ml_assets() 
    if pipeline is None:
        logger.critical("Бот не може запуститися, оскільки ML-модель не завантажена. Перевірте PATH.")
        return

    # 2. Ініціалізація бота
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Зберігаємо модель у bot_data
    application.bot_data[ML_MODEL_KEY] = pipeline

    # 3. Налаштування діалогу (ConversationHandler)
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        
        states={
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_age)],
            GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_gender)],
            CAR: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_car)],
            REALTY: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_realty)],
            INCOME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_income)],
            INCOME_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_income_type)],
            EDUCATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_education)],
            MARITAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_marital)],
            HOUSING: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_housing)],
            CHILDREN: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_children)],
            OCCUPATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_occupation)],
            FAMILY_MEMBERS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_family_members)],
            EMPLOYED_DAYS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_employed_days)],
        },
        
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    application.add_handler(conv_handler)
    application.add_error_handler(error_handler)

    # 4. Запуск бота
    logger.info("Бот успішно запущений. Очікую на команди...")
    application.run_polling(poll_interval=1)
    


if __name__ == '__main__':
    # Фінальні списки ознак для коректного запуску
    # Ці списки гарантують, що підготовка даних буде відповідати тренувальним
    NUMERICAL_FEATURES_FINAL = ['AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AGE', 'YEARS_EMPLOYED']
    CATEGORICAL_FEATURES = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
    ]
    main()
