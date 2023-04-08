import os
import sys
import datetime
import pytz
import ffmpeg
import json
import lxml.etree as ET
import pandas as pd
from collections import Counter


class DanmakuPool:
    def __init__(self) -> None:
        # initialize empty data frame
        self.df = pd.DataFrame(
            columns=[
                "time",
                "offset",
                "roomid",
                "uid",
                "uname",
                "color",
                "price",
                "text",
            ]
        )

    # def add_danmaku(self, danmaku: dict) -> None:
    #     """Add a danmaku to the pool.

    #     Parameters:
    #         danmaku: A dictionary of danmaku attributes:

    #             time: Danmaku timestamp, timestamp, example: 1680514935.055

    #             offset: display time offset w.r.t. the timestamp, float, example: 1.214, -2.089

    #             roomid: room id, string, example: "92613"

    #             uid: user id, string, example: "74354088"

    #             uname: user name, string, example: "ymyyg"

    #             color: color, int, example: 16777215

    #             price: price in CNY if superchat, int, example: 0

    #             text: danmaku text, example: "233333"

    #     Returns:
    #         None
    #     """
    #     self.df = pd.concat([self.df, pd.DataFrame([danmaku])], ignore_index=True)

    def add_danmaku_from_xml(self, file_path: str) -> None:
        """Add danmaku to the pool from xml file.

        The xml file is expected to have a <BililiveRecorderRecordInfo> tag with metadata, a <d> tag for each danmaku and
        a <sc> tag for each superchat.

        The <BiliLiveRecorderRecordInfo> tag is expected to have the following attributes:

            roomid: room id

            shortid: short room id

            name: streamer username

            title: stream title

            areanameparent: area the live belongs to, example: "PC/Console Games", "Online Games", "Entertainment"

            areanamechild: subarea the live belongs to, example: "FPS", "Mobile Games", "Variety"

            start_time: start time of the recording, example: "2023-04-03T02:42:13.8005156-07:00"

        The <d> tag is expected to have a "p" attribute with the following format, separated by commas:

            example: p="3.090,1,25,4546550,1658318972567,0,81658411,0"

            p[0] = 3.090: video time offset

            p[1] = 1: player mode (danmaku location, 1=right to left, 3=top, 4=bottom)

            p[2] = 25: font size (seems to be always 25)

            p[3] = 4546550: color

            p[4] = 1658318972567: timestamp

            p[5] = 0: unknown (seems to be always zero)

            p[6] = 81658411: uid

            p[7] = 0: unknown (seems to be always zero)

        a "user" attribute with the username and the text is the danmaku.

        The <sc> tag is expected to have the following attributes:

            ts: video time offset

            user: username

            uid: uid

            price: superchat price

            time: superchat duration

        And the text is the danmaku.

        Parameters:
            file_path: Path to the file.
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            info = root.find("BililiveRecorderRecordInfo")
            roomid = info.attrib["roomid"]
            start_time = parse_Bilirec_time(info.attrib["start_time"]).timestamp()
            danmaku = {
                "time": [],
                "offset": [],
                "roomid": [],
                "uid": [],
                "uname": [],
                "color": [],
                "price": [],
                "text": [],
            }
            for child in root.findall("d"):
                p = child.attrib["p"].split(",")
                danmaku["time"].append(float(p[4]) * 1e-3)
                danmaku["offset"].append(start_time + float(p[0]) - float(p[4]) * 1e-3)
                danmaku["roomid"].append(roomid)
                danmaku["uid"].append(p[6])
                danmaku["uname"].append(child.attrib["user"])
                danmaku["color"].append(int(p[3]))
                danmaku["price"].append(0)
                danmaku["text"].append(child.text)
            for child in root.findall("sc"):
                danmaku["time"].append(start_time + float(p[0]))
                danmaku["offset"].append(0)
                danmaku["roomid"].append(roomid)
                danmaku["uid"].append(child.attrib["uid"])
                danmaku["uname"].append(child.attrib["user"])
                danmaku["color"].append(16772431)
                danmaku["price"].append(int(child.attrib["price"]))
                danmaku["text"].append(child.text)
            self.df = pd.concat([self.df, pd.DataFrame(danmaku)], ignore_index=True)
        except:
            print("Error: load_danmaku failed, file =", file_path, file=sys.stderr)

    def blacklist_filter(self, blacklist: dict[str, list]) -> None:
        """Filter danmaku by blacklist.

        Parameters:

            blacklist: Dictionary of blacklists. The key is the room id, and the value is a list of banned danmaku.
            Example: {"123456": ["banned danmaku 1", "banned danmaku 2"], "654321": ["banned danmaku 3"]}
        """
        for roomid in blacklist:
            self.df = self.df[~self.df["roomid"].eq(roomid) | ~self.df["text"].isin(blacklist[roomid])]


def parse_Bilirec_time(time_str: str) -> datetime.datetime:
    """Parse Bilirec comment time string to datetime.

    Bilirec uses ISO format but with 7 digits of microseconds,
    while datetime only supports 6 digits.

    Example: 2022-07-20T05:09:30.0049679-07:00"""
    # remove the last digit of microseconds
    time_str_iso = time_str[:26] + time_str[27:]
    return datetime.datetime.fromisoformat(time_str_iso)


def get_video_creation_time(video_path: str) -> datetime.datetime or None:
    """Get video creation time using ffmpeg.

    Assuming the comment field of the video has a line like: "录制时间: 2021-07-20T05:09:30.0049679-07:00"
    """
    try:
        output = ffmpeg.probe(video_path)
        tags = output["format"]["tags"]
        if "comment" in tags:
            comment: str = tags["comment"]
        else:
            comment: str = tags["Comment"]
        for line in comment.splitlines():
            if line.startswith("录制时间"):
                time_str = line.split(" ")[1]
                creation_time = parse_Bilirec_time(time_str)
                break
        return creation_time
    except:
        print("Error: get_video_creation_time failed, file =", video_path, file=sys.stderr)
        return None


def get_xml_creation_time(xml_path: str) -> datetime.datetime or None:
    """Get xml file creation time using lxml.

    Assuming there's an attribute like start_time="2022-07-20T05:09:30.0049679-07:00" in the
    <BililiveRecorderRecordInfo> tag.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        time_str = root.find("BililiveRecorderRecordInfo").attrib["start_time"]
        creation_time = parse_Bilirec_time(time_str)
        return creation_time
    except:
        print(
            "Error: get_danmaku_creation_time failed, file =",
            xml_path,
            file=sys.stderr,
        )
        return None


def get_file_modification_time(file_path: str) -> datetime.datetime:
    """Get file modification time using os.stat."""
    try:
        return datetime.datetime.fromtimestamp(os.stat(file_path).st_mtime, tz=TIMEZONE)
    except:
        print(
            "Error: get_file_modification_time failed, file =",
            file_path,
            file=sys.stderr,
        )
        return None


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffmpeg.

    Returns:
        Duration in seconds."""
    try:
        output = ffmpeg.probe(video_path)
        duration = float(output["format"]["duration"])
        return duration
    except:
        print("Error: video_duration failed, file =", video_path, file=sys.stderr)
        return 0


def get_duration_inconsistency(video_path: str) -> tuple[float, float]:
    """Get time difference in seconds between video duration and the recording time.

    Parameters:
        file_path: Path to the video file.

    Returns:
        Tuple of time differences in seconds. The first element is the difference between video duration and xml
        recording time. The second element is the difference between video recording time and xml recording time as a
        sanity check.

    If the video file is not the original flv file, the video recording duration would be incorrect. The xml recording
    duration in general would be a few seconds longer than the video recording duration, since danmaku after the stream
    ends while the live has not ended yet would be recorded.
    """
    xml_path = video_path[:-4] + ".xml"
    video_creation_time = get_video_creation_time(video_path)
    xml_creation_time = get_xml_creation_time(xml_path)
    video_modification_time = get_file_modification_time(video_path)
    xml_modification_time = get_file_modification_time(xml_path)
    video_duration = get_video_duration(video_path)
    if (
        video_creation_time is None
        or xml_creation_time is None
        or video_modification_time is None
        or xml_modification_time is None
        or video_duration == 0
    ):
        print("Error: get_time_inconsistency failed, file =", video_path, file=sys.stderr)
        return 0, 0
    video_recording_duration = (video_modification_time - video_creation_time).total_seconds()
    xml_recording_duration = (xml_modification_time - xml_creation_time).total_seconds()
    return (
        video_duration - xml_recording_duration,
        video_recording_duration - xml_recording_duration,
    )


def examine_danmaku(pool: DanmakuPool, top_num: int = 50) -> None:
    """Show the most frequent danmaku in the pool.

    Parameters:

        df: DataFrame containing the danmaku pool.

        top_num: Number of danmaku to show. Default is 50.
    """
    stats = {}
    for i in range(len(pool.df)):
        danmaku = pool.df.iloc[i]
        roomid = danmaku["roomid"]
        text = danmaku["text"]
        if roomid not in stats:
            stats[roomid] = Counter()
        stats[roomid].update({text: 1})
    for roomid, counter in stats.items():
        print(f"Room ID: {roomid}")
        for text, count in counter.most_common(top_num):
            print(f"{count:>6} {text}")


# load config
with open("config.json", "r") as f:
    config = json.load(f)
    TIMEZONE = pytz.timezone(config["timezone"])
    BLACKLIST = config["blacklist"]
    AVERAGE_DURATION = config["average_duration"]
