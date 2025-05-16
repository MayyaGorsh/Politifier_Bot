import asyncio
import logging
from datetime import datetime, date

from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command, CommandObject
import uuid
from Politify_function import Politify
from token import token


logging.basicConfig(level=logging.INFO)
bot = Bot(token=token)
dp = Dispatcher()
dp["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
BOT_USERNAME = '@politifier_bot'

# Словарь для отслеживания, кому мы уже показывали инструкцию сегодня
last_shown = {}  # user_id: date


# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message, command: CommandObject):
    if command.args == "help":
        await message.answer(
            "ℹ️ Чтобы воспользоваться ботом:\n\n"
            "1. В любом чате напишите `@politifier_bot` и следом сообщение.\n"
            "2. Бот покажет вежливую версию.\n"
            "3. Выберите ее и отправьте в чат.\n\n"
        )
    else:
        await message.answer("Здравствуйте! Я помогу сделать ваши сообщения вежливее.")


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(
        "ℹ️ Чтобы воспользоваться ботом:\n\n"
        "1. В любом чате напишите `@politifier_bot` и следом сообщение.\n"
        "2. Бот покажет вежливую версию.\n"
        "3. Выберите ее и отправьте в чат.\n\n"
    )


@dp.inline_query()
async def inline_query_handler(inline_query: types.InlineQuery):
    query = inline_query.query.strip()
    results = []

    user_id = inline_query.from_user.id
    today = date.today()

    show_help_button = False
    if user_id not in last_shown or last_shown[user_id] != today:
        last_shown[user_id] = today
        show_help_button = True
    if show_help_button:
        print('help', inline_query.id)
        await inline_query.answer(
            results,
            cache_time=1,
            switch_pm_text="ℹ️ Как пользоваться ботом?",
            switch_pm_parameter="help"
        )
    else:
        if query:
            # Create a single article result that, when chosen, sends 'query' as the message
            polite_version = Politify(query)

            results.append(
                types.InlineQueryResultArticle(
                    id=str(uuid.uuid4()),
                    title="Нажмите, чтобы отправить вежливую версию сообщения.",
                    description=polite_version,
                    input_message_content=types.InputTextMessageContent(message_text=polite_version)
                )
            )

        # Answer the inline query with the results
        print(inline_query.id)
        await inline_query.answer(results, cache_time=3)  # Set cache_time=1 for testing; can be higher in production


async def main():
    # Запускаем бота и пропускаем все накопленные входящие
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
