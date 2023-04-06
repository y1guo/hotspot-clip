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
    asking_str = input("Enter your asking time (HH:MM:SS):")
    if len(asking_str.split(":")) != 3:
        continue
    asking_hh = int(asking_str.split(":")[0])
    asking_mm = int(asking_str.split(":")[1])
    asking_ss = int(asking_str.split(":")[2])
    sec_start = time2sec(start_hh, start_mm, start_ss)
    sec_asking = time2sec(asking_hh, asking_mm, asking_ss)
    diff_hh, diff_mm, diff_ss = sec2time(sec_asking - sec_start)
    print(f"Asked time in the video is: {diff_hh}:{diff_mm}:{diff_ss}")
