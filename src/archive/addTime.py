def time2sec(hh, mm, ss):
    return hh * 3600 + mm * 60 + ss

def sec2time(sec):
    ss = sec % 60
    mm = (sec // 60) % 60
    hh = sec // 3600
    return hh, mm, ss

start_str = input("Enter video start time (HH:MM:SS):")
start_hh = int(start_str.split(":")[0])
start_mm = int(start_str.split(":")[1])
start_ss = int(start_str.split(":")[2])

while True:
    added_str = input("Enter your time to add (HH:MM:SS):")
    if len(added_str.split(":")) != 3:
        continue
    added_hh = int(added_str.split(":")[0])
    added_mm = int(added_str.split(":")[1])
    added_ss = int(added_str.split(":")[2])
    sec_start = time2sec(start_hh, start_mm, start_ss)
    sec_added = time2sec(added_hh, added_mm, added_ss)
    added_hh, added_mm, added_ss = sec2time(sec_added + sec_start)
    print(f"Added time is: {added_hh}:{added_mm}:{added_ss}")
