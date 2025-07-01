import datetime

def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log_focus(status, file="focus_log.csv"):
    with open(file, "a") as f:
        f.write(f"{get_time()},{status}\n")
