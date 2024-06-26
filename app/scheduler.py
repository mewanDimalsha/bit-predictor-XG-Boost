from apscheduler.schedulers.background import BackgroundScheduler
from model.model import train_and_save_model

scheduler = BackgroundScheduler()
scheduler.add_job(train_and_save_model, "cron", hour=0, minute=0)
scheduler.start()
