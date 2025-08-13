from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import requests

BACKEND_URL = "https://b022264d0b10.ngrok-free.app"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sender_id = str(update.effective_chat.id)
    res = requests.get(f"{BACKEND_URL}/start_chat", params={"sender_id": sender_id}).json()
    await update.message.reply_text(res["reply"])

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sender_id = str(update.effective_chat.id)
    user_message = update.message.text
    res = requests.post(f"{BACKEND_URL}/chat", json={
        "message": user_message,
        "sender_id": sender_id
    }).json()
    await update.message.reply_text(res["reply"])


app = ApplicationBuilder().token("8213335245:AAEDnboQLo6PHiIo0sYPjeVoe3aaPVnaiY8").build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

app.run_polling()