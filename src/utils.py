import os
import sys
import datetime
import pytz
import ffmpeg
import json
import lxml.etree as ET
import pandas as pd


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

    def add_danmaku(self, danmaku: dict) -> None:
        """Add a danmaku to the pool.

        Parameters:
            danmaku: A dictionary of danmaku attributes:

                time: Danmaku timestamp, example: 1680514935.055

                offset: display time offset w.r.t. the timestamp, example: 1.214, -2.089

                roomid: room id, example: 92613

                uid: user id, example: 74354088

                uname: user name, example: "ymyyg"

                color: color, example: 16777215

                price: price in CNY if superchat, example: 0

                text: danmaku text, example: "233333"

        Returns:
            None
        """
        self.df.loc[len(self.df)] = danmaku


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


def load_danmaku(file_path: str, pool: DanmakuPool) -> DanmakuPool:
    """Load danmaku from the xml file.

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
        pool: DanmakuPool to be loaded into.

    Returns:
        DanmakuPool.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        info = root.find("BililiveRecorderRecordInfo")
        roomid = int(info.attrib["roomid"])
        start_time = parse_Bilirec_time(info.attrib["start_time"]).timestamp()
        for child in root.findall("d"):
            danmaku = {}
            p = child.attrib["p"].split(",")
            danmaku["time"] = float(p[4]) * 1e-3
            danmaku["offset"] = start_time + float(p[0]) - danmaku["time"]
            danmaku["roomid"] = roomid
            danmaku["uid"] = int(p[6])
            danmaku["uname"] = child.attrib["user"]
            danmaku["color"] = int(p[3])
            danmaku["price"] = 0
            danmaku["text"] = child.text
            pool.add_danmaku(danmaku)
        for child in root.findall("sc"):
            danmaku = {}
            danmaku["time"] = start_time + float(p[0])
            danmaku["offset"] = 0
            danmaku["roomid"] = roomid
            danmaku["uid"] = int(child.attrib["uid"])
            danmaku["uname"] = child.attrib["user"]
            danmaku["color"] = 16772431
            danmaku["price"] = int(child.attrib["price"])
            danmaku["text"] = child.text
            pool.add_danmaku(danmaku)
    except:
        print("Error: load_danmaku failed, file =", file_path, file=sys.stderr)
        return pool


def blacklist_filter(pool: DanmakuPool) -> DanmakuPool:
    # to do
    return pool


# load config
with open("config.json", "r") as f:
    config = json.load(f)
    TIMEZONE = pytz.timezone(config["timezone"])
    BLACKLIST = config["blacklist"]
