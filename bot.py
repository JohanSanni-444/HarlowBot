import telebot
import os
from telebot import types
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import logging

# --- Логирование ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- Конец логирования ---

# Замените 'YOUR_BOT_TOKEN' на токен вашего бота
BOT_TOKEN = os.environ.get("BOT_TOKEN", "7834448164:AAHzmGRhivgkvrUYK-a8DhkV8N9JFZpveh8")
bot = telebot.TeleBot(BOT_TOKEN)

# Папка для сохранения аудиофайлов
UPLOAD_FOLDER = 'audio_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Имя файла примера
EXAMPLE_AUDIO_FILE = "audio_uploads/nokia.mp3"  # Замените на имя вашего файла примера

# --- Эффекты ---
def trim_silence(y, sr, threshold_db=-60, frame_length=2048, hop_length=512):
    """Обрезка тишины в начале и конце аудиосигнала."""
    try:
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        threshold = librosa.db_to_amplitude(threshold_db)
        frames = np.where(rms > threshold)[0]
        if frames.size == 0:
            return np.array([])  # Возвращаем пустой массив, если тишина
        start_frame = frames[0]
        end_frame = frames[-1]
        start_sample = start_frame * hop_length
        end_sample = min(len(y), (end_frame + 1) * hop_length)
        return y[start_sample:end_sample]
    except Exception as e:
        logger.error(f"Ошибка при обрезке тишины: {e}")
        return y  # Возвращаем исходный сигнал в случае ошибки


def apply_distortion(audio_path, output_path, gain=10.0, threshold=0.5):
    """Применение эффекта distortion."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_amplified = y * gain
        y_clipped = np.clip(y_amplified, -threshold, threshold)
        max_value = np.max(np.abs(y_clipped))

        if max_value > 0:
            y_normalized = y_clipped / max_value
        else:
            y_normalized = y_clipped

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None
    except Exception as e:
        logger.error(f"Ошибка при применении distortion: {e}")
        return False, str(e)


def apply_overdrive(audio_path, output_path, gain=6.0, curve_amount=0.5):
    """Применение эффекта overdrive."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_amplified = y * gain
        y_overdriven = (1 + curve_amount) * y_amplified / (1 + curve_amount * np.abs(y_amplified))
        max_value = np.max(np.abs(y_overdriven))

        if max_value > 0:
            y_normalized = y_overdriven / max_value
        else:
            y_normalized = y_overdriven

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None
    except Exception as e:
        logger.error(f"Ошибка при применении overdrive: {e}")
        return False, str(e)


def apply_fuzz(audio_path, output_path, gain=20.0, clipping_level=0.8, feedback=0.2):
    """Применение эффекта fuzz."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_amplified = y * gain
        y_clipped = np.clip(y_amplified, -clipping_level, clipping_level)
        y_fuzzed = y_clipped + feedback * np.clip(y_clipped, -clipping_level, clipping_level)
        y_fuzzed = np.clip(y_fuzzed, -clipping_level, clipping_level)
        max_value = np.max(np.abs(y_fuzzed))

        if max_value > 0:
            y_normalized = y_fuzzed / max_value
        else:
            y_normalized = y_fuzzed

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None
    except Exception as e:
        logger.error(f"Ошибка при применении fuzz: {e}")
        return False, str(e)


def apply_reverb(audio_path, output_path, reverb_time=0.5, decay=0.5):
    """Применение эффекта reverb."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        impulse_response = np.zeros(int(reverb_time * sr))
        impulse_response[0] = 1.0
        decay_factor = decay ** (1 / len(impulse_response))
        for i in range(1, len(impulse_response)):
            impulse_response[i] = impulse_response[i - 1] * decay_factor
        y_reverbed = signal.convolve(y, impulse_response, mode='full')
        max_value = np.max(np.abs(y_reverbed))

        if max_value > 0:
            y_normalized = y_reverbed / max_value
        else:
            y_normalized = y_reverbed

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None
    except Exception as e:
        logger.error(f"Ошибка при применении reverb: {e}")
        return False, str(e)


def apply_delay(audio_path, output_path, delay_time=0.3, feedback=0.4, dry_wet=0.6):
    """Применение эффекта delay."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        delay_samples = int(delay_time * sr)
        delay_buffer = np.zeros(len(y) + delay_samples)
        delay_buffer[:len(y)] = y
        y_delayed = np.zeros_like(delay_buffer)
        for i in range(len(y)):
            y_delayed[i] += delay_buffer[i] * (1 - dry_wet)
            y_delayed[i + delay_samples] += delay_buffer[i] * dry_wet
            delay_buffer[i + delay_samples] += delay_buffer[i] * dry_wet * feedback
        max_value = np.max(np.abs(y_delayed))

        if max_value > 0:
            y_normalized = y_delayed / max_value
        else:
            y_normalized = y_delayed

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None
    except Exception as e:
        logger.error(f"Ошибка при применении delay: {e}")
        return False, str(e)


def apply_chorus(audio_path, output_path, delay=20.0, depth=3.0, rate=0.5, dry_wet=0.5):
    """Применение эффекта chorus."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        delay_samples = int((delay / 1000) * sr)
        depth_samples = int((depth / 1000) * sr)
        lfo = np.sin(2 * np.pi * rate * np.arange(len(y)) / sr) * depth_samples
        delayed_signal = np.zeros_like(y)
        for i in range(len(y)):
            current_delay = int(delay_samples + lfo[i])
            if i - current_delay >= 0:
                delayed_signal[i] = y[i - current_delay]
        y_chorus = (1 - dry_wet) * y + dry_wet * delayed_signal
        max_value = np.max(np.abs(y_chorus))

        if max_value > 0:
            y_normalized = y_chorus / max_value
        else:
            y_normalized = y_chorus

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None
    except Exception as e:
        logger.error(f"Ошибка при применении chorus: {e}")
        return False, str(e)


def apply_flanger(audio_path, output_path, delay=5.0, depth=2.0, rate=0.2, dry_wet=0.5, feedback=0.0):
    """Применение эффекта flanger."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        delay_samples = int((delay / 1000) * sr)
        depth_samples = int((depth / 1000) * sr)
        lfo = np.sin(2 * np.pi * rate * np.arange(len(y)) / sr) * depth_samples
        delayed_signal = np.zeros_like(y)
        feedback_buffer = np.zeros_like(y)
        for i in range(len(y)):
            current_delay = int(delay_samples + lfo[i])
            if i - current_delay >= 0 and i - current_delay < len(y):
                delayed_signal[i] = y[i - current_delay]
            else:
                delayed_signal[i] = 0.0
            feedback_buffer[i] = delayed_signal[i] + feedback * delayed_signal[i]
        y_flanger = (1 - dry_wet) * y + dry_wet * feedback_buffer
        max_value = np.max(np.abs(y_flanger))

        if max_value > 0:
            y_normalized = y_flanger / max_value
        else:
            y_normalized = y_flanger

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None
    except Exception as e:
        logger.error(f"Ошибка при применении flanger: {e}")
        return False, str(e)


def apply_phaser(audio_path, output_path, rate=0.8, depth=0.7, stages=4, dry_wet=0.5):
    """Применение эффекта phaser."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        lfo = np.sin(2 * np.pi * rate * np.arange(len(y)) / sr)
        f_c = 500 + lfo * 400
        y_phaser = y.copy()
        for i in range(stages):
            g = (np.tan(np.pi * f_c / sr) - 1) / (np.tan(np.pi * f_c / sr) + 1)
            g = np.clip(g, -1, 1)
            delayed = np.zeros_like(y_phaser)
            for n in range(1, len(y_phaser)):
                delayed[n] = g[n] * y_phaser[n] + y[n - 1] - g[n] * delayed[n - 1]
            y_phaser = delayed
        y_phaser = (1 - dry_wet) * y + dry_wet * y_phaser
        max_value = np.max(np.abs(y_phaser))

        if max_value > 0:
            y_normalized = y_phaser / max_value
        else:
            y_normalized = y_phaser

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None
    except Exception as e:
        logger.error(f"Ошибка при применении phaser: {e}")
        return False, str(e)


def apply_tremolo(audio_path, output_path, rate=6.0, depth=0.8):
    """Применение эффекта tremolo."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        lfo = np.sin(2 * np.pi * rate * np.arange(len(y)) / sr)
        y_tremolo = y * (1 - depth + depth * lfo)
        max_value = np.max(np.abs(y_tremolo))

        if max_value > 0:
            y_normalized = y_tremolo / max_value
        else:
            y_normalized = y_tremolo

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None
    except Exception as e:
        logger.error(f"Ошибка при применении tremolo: {e}")
        return False, str(e)


def apply_wahwah(audio_path, output_path, rate=1.0, depth=0.7, Q=1.0):
    """Применение эффекта wahwah."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        lfo = np.sin(2 * np.pi * rate * np.arange(len(y)) / sr)
        f_center_min = 300
        f_center_max = 2000
        f_center = f_center_min + depth * (f_center_max - f_center_min) * (lfo + 1) / 2
        nyquist = sr / 2
        order = 2  # Порядок фильтра
        b = np.zeros((len(y), order + 1))  # Числитель (order + 1 coefficients)
        a = np.zeros((len(y), order + 1))  # Знаменатель (order + 1 coefficients)

        for i in range(len(y)):
            low = f_center[i] / Q / nyquist
            high = f_center[i] * Q / nyquist

            if low > 0 and high < 1:
                b_coeffs, a_coeffs = signal.butter(order, [low, high], btype='band')
                if len(b_coeffs) == order + 1 and len(a_coeffs) == order + 1:  # Проверка длины коэффициентов
                    b[i, :] = b_coeffs
                    a[i, :] = a_coeffs
                else:
                    logger.warning(f"Неверная длина коэффициентов фильтра для i={i}. Пропускаем фильтрацию.")
                    b[i, :] = np.zeros(order + 1)  # Обнуляем коэффициенты
                    a[i, :] = np.array([1] + [0] * order)  # a[0] = 1, остальные нули
            else:
                logger.warning(f"Недопустимые границы полосы для i={i}. Пропускаем фильтрацию.")
                b[i, :] = np.zeros(order + 1)
                a[i, :] = np.array([1] + [0] * order)  # a[0] = 1, остальные нули

        y_wahwah = np.zeros_like(y)
        zi = np.zeros(order)  # Initial conditions for the filter (length order)
        for i in range(len(y)):
            # Проверка на стабильность и корректность коэффициентов (избегаем NaN)
            if np.isnan(b[i, :]).any() or np.isnan(a[i, :]).any() or np.any(np.abs(a[i, :]) > 1e10) or np.any(np.abs(b[i, :]) > 1e10):
                y_wahwah[i] = y[i]  # Если фильтр нестабилен, пропускаем исходный сигнал
            else:
                y_wahwah[i], zi = signal.lfilter(b[i, :], a[i, :], [y[i]], zi=zi)  # Применяем фильтр поэлементно

        max_value = np.max(np.abs(y_wahwah))
        if max_value > 0:
            y_normalized = y_wahwah / max_value
        else:
            y_normalized = y_wahwah

        y_trimmed = trim_silence(y_normalized, sr)
        sf.write(output_path, y_trimmed, sr)
        return True, None

    except Exception as e:
        logger.error(f"Ошибка при применении вау-вау: {e}")
        return False, str(e)

# --- Остальной код бота (без изменений) ---

# --- Клавиатура для регулировки параметров ---
def create_parameter_keyboard(effect):
    keyboard = types.InlineKeyboardMarkup(row_width=4)

    for param, value in effect_params[effect].items():
        keyboard.add(types.InlineKeyboardButton(f"{param}: {value:.2f}", callback_data="noop"))
        keyboard.add(types.InlineKeyboardButton("-1", callback_data=f"param_down_{effect}_{param}_1"))
        keyboard.add(types.InlineKeyboardButton("-0.1", callback_data=f"param_down_{effect}_{param}_0.1"))
        keyboard.add(types.InlineKeyboardButton("+0.1", callback_data=f"param_up_{effect}_{param}_0.1"))
        keyboard.add(types.InlineKeyboardButton("+1", callback_data=f"param_up_{effect}_{param}_1"))
    keyboard.add(types.InlineKeyboardButton("✅ Применить", callback_data=f"apply_effect_{effect}"))
    return keyboard

# --- Выбор эффекта (Обновленная клавиатура) ---
def create_effect_keyboard():
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    keyboard.add(types.InlineKeyboardButton("Distortion", callback_data='distortion'))
    keyboard.add(types.InlineKeyboardButton("Overdrive", callback_data='overdrive'))
    keyboard.add(types.InlineKeyboardButton("Fuzz", callback_data='fuzz'))
    keyboard.add(types.InlineKeyboardButton("Reverb", callback_data='reverb'))
    keyboard.add(types.InlineKeyboardButton("Delay", callback_data='delay'))
    keyboard.add(types.InlineKeyboardButton("Chorus", callback_data='chorus'))
    keyboard.add(types.InlineKeyboardButton("Flanger", callback_data='flanger'))
    keyboard.add(types.InlineKeyboardButton("Phaser", callback_data='phaser'))
    keyboard.add(types.InlineKeyboardButton("Tremolo", callback_data='tremolo'))
    keyboard.add(types.InlineKeyboardButton("WahWah", callback_data='wahwah'))
    return keyboard

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start(message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button_upload = types.KeyboardButton("Загрузить аудио")
    button_apply_again = types.KeyboardButton("Применить эффект к обработанному")
    keyboard.add(button_upload, button_apply_again)
    bot.send_message(message.chat.id, "👋 Привет! Я бот для обработки аудио. 🎶\nЗагрузите аудио 🎧, чтобы начать, или нажмите 'Применить эффект к обработанному' для работы с предыдущим результатом.", reply_markup=keyboard)

    # Отправка примера аудио сразу после запуска бота
    if os.path.exists(EXAMPLE_AUDIO_FILE):
        try:
            with open(EXAMPLE_AUDIO_FILE, 'rb') as audio:
                bot.send_audio(message.chat.id, audio, caption="Пример аудио для обработки")
        except Exception as e:
            logger.error(f"Ошибка при отправке примера аудио: {e}")
            bot.send_message(message.chat.id, "Не удалось отправить пример аудио.")
    else:
        logger.warning("Файл примера аудио не найден.")
        bot.send_message(message.chat.id, "Файл примера аудио не найден.")


# Обработчик команды "Применить эффект к обработанному"
@bot.message_handler(func=lambda message: message.text == "Применить эффект к обработанному")
def handle_apply_again_request(message):
    chat_id = message.chat.id
    user_data = database.get_user_data(chat_id) # Get User Data from DB
    if not user_data.file_path:
        bot.send_message(chat_id, "Сначала загрузите аудиофайл и примените эффект.")
        return
    bot.send_message(chat_id, "Выберите эффект для повторного применения:", reply_markup=keyboards.create_effect_keyboard())

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: message.text == "Загрузить аудио")
def handle_upload_request(message):
    bot.send_message(message.chat.id, "Пожалуйста, отправьте аудиофайл.")

# Обработчик входящих аудиофайлов
@bot.message_handler(content_types=['audio'])
def handle_audio(message):
    try:
        logger.info(f"Получено аудио от {message.chat.id}")
        file_info = bot.get_file(message.audio.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        file_name = message.audio.file_name if message.audio.file_name else f"audio_{message.audio.file_id}.mp3"
        file_path = os.path.join(config.UPLOAD_FOLDER, file_name)

        with open(file_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        database.update_user_data(message.chat.id, file_path=file_path)
        bot.send_message(message.chat.id, f"Аудиофайл '{file_name}' успешно загружен.  Выберите эффект:", reply_markup=keyboards.create_effect_keyboard())

    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка при обработке файла: {e}")
        logger.error(f"Ошибка при обработке аудио: {e}")

# Обработчик callback-запросов
@bot.callback_query_handler(func=lambda call: True)
def handle_query(call):
    try:
        data = call.data
        chat_id = call.message.chat.id
        user_data = database.get_user_data(chat_id) # Get User Data from DB

        if data in ['distortion', 'overdrive', 'fuzz', 'reverb', 'delay', 'chorus', 'flanger', 'phaser', 'tremolo', 'wahwah']: # Если это выбор эффекта
            database.update_user_data(chat_id, effect=data)
            bot.edit_message_text(f"Вы выбрали эффект *{data}*. ⚙️\nНастройте параметры:",
                                  chat_id,
                                  call.message.message_id,
                                  parse_mode="Markdown",
                                  reply_markup=keyboards.create_parameter_keyboard(data))

        elif data.startswith("param_up_") or data.startswith("param_down_"):
            parts = data.split("_")
            _, direction, effect, param, step = parts
            change = float(step) if direction == "up" else -float(step)

            # Update parameter in database
            param_name = f"{effect}_{param}" # distortion_gain
            #Update param in DB
            database.update_user_data(chat_id, **{param_name: getattr(user_data, param_name) + change})

            # Reload User Data
            user_data = database.get_user_data(chat_id)

            bot.edit_message_text(f"Настройте параметры для *{effect}*:",
                                  chat_id,
                                  call.message.message_id,
                                  parse_mode="Markdown",
                                  reply_markup=keyboards.create_parameter_keyboard(effect))

        elif data.startswith("apply_effect_"): # Apply effect button
            effect = data.split("_")[2]
            if not user_data.file_path:
                bot.send_message(chat_id, "Сначала загрузите аудиофайл.")
                return

            input_file = user_data.file_path
            output_file = os.path.join(config.UPLOAD_FOLDER, f"processed_{os.path.basename(input_file)}")

            success = False
            error_message = None

            # Apply effect
            if effect == 'distortion':
                success, error_message = effects.apply_distortion(input_file, output_file, gain=user_data.distortion_gain, threshold=user_data.distortion_threshold)
            elif effect == 'overdrive':
                success, error_message = effects.apply_overdrive(input_file, output_file, gain=user_data.overdrive_gain, curve_amount=user_data.overdrive_curve_amount)
            elif effect == 'fuzz':
                success, error_message = effects.apply_fuzz(input_file, output_file, gain=user_data.fuzz_gain, clipping_level=user_data.fuzz_clipping_level, feedback=user_data.fuzz_feedback)
            elif effect == 'reverb':
                success, error_message = effects.apply_reverb(input_file, output_file, reverb_time=user_data.reverb_reverb_time, decay=user_data.reverb_decay)
            elif effect == 'delay':
                success, error_message = effects.apply_delay(input_file, output_file, delay_time=user_data.delay_delay_time, feedback=user_data.delay_feedback, dry_wet=user_data.delay_dry_wet)
            elif effect == 'chorus':
                success, error_message = effects.apply_chorus(input_file, output_file, delay=user_data.chorus_delay, depth=user_data.chorus_depth, rate=user_data.chorus_rate, dry_wet=user_data.chorus_dry_wet)
            elif effect == 'flanger':
                success, error_message = effects.apply_flanger(input_file, output_file, delay=user_data.flanger_delay, depth=user_data.flanger_depth, rate=user_data.flanger_rate, dry_wet=user_data.flanger_dry_wet, feedback=user_data.flanger_feedback)
            elif effect == 'phaser':
                success, error_message = effects.apply_phaser(input_file, output_file, rate=user_data.phaser_rate, depth=user_data.phaser_depth, stages=user_data.phaser_stages, dry_wet=user_data.phaser_dry_wet)
            elif effect == 'tremolo':
                success, error_message = effects.apply_tremolo(input_file, output_file, rate=user_data.tremolo_rate, depth=user_data.tremolo_depth)
            elif effect == 'wahwah':
                success, error_message = effects.apply_wahwah(input_file, output_file, rate=user_data.wahwah_rate, depth=user_data.wahwah_depth, Q=user_data.wahwah_Q)
            else:
                bot.send_message(chat_id, "Неизвестный эффект.")
                return

            if success:
                with open(output_file, 'rb') as audio:
                    bot.send_audio(chat_id, audio, caption=f"🎶  Обработанный аудио с эффектом *{effect}*.  ✅", parse_mode="Markdown")
                #Store new file path to db
                database.update_user_data(chat_id, file_path=output_file)
            else:
                bot.send_message(chat_id, f"❌ Произошла ошибка при применении эффекта: {error_message}")
    except Exception as e:
        bot.send_message(call.message.chat.id, f"Произошла непредвиденная ошибка: {e}")
        logger.error(f"Произошла непредвиденная ошибка: {e}")

# Обработчик неизвестных команд
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "Я не понимаю эту команду. Попробуйте /start")

# Запуск бота
if __name__ == '__main__':
    print("Бот запущен...")
    bot.infinity_polling()