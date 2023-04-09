from __future__ import annotations
import os
import sys
import datetime
import pytz
import ffmpeg
import json
import math
import lxml.etree as ET
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
from copy import deepcopy
from scipy.optimize import minimize


class DanmakuPool:
    def __init__(self, df: pd.DataFrame = None) -> None:
        # initialize empty data frame
        if df is None:
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
        else:
            self.df = deepcopy(df)

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

    def show_common_danmaku(self, top_num: int = 50) -> None:
        """Show the most frequent danmaku in the pool.

        Parameters:

            top_num: Number of danmaku to show. Default is 50.
        """
        for roomid in self.df["roomid"].unique():
            print(f"Room ID: {roomid}")
            df = self.df[self.df["roomid"].eq(roomid)]
            counter = Counter(df["text"])
            for text, count in counter.most_common(top_num):
                print(f"{count:>6} {text}")

    def get_danmaku_density_flat(self, roomid: str, smoothing_window: int = 0) -> tuple(np.ndarray, np.ndarray):
        """`deprecated`
            Get danmaku density for a room using a flat kernel.

        Parameters:

            roomid: Room ID.

            smoothing_window: The size of the smoothing window in seconds. Default is 0, which means no smoothing
            (kernel size = 1 second).

        Returns:

            A tuple of numpy array. The first array is the timestamp. The second array is the danmaku density, with
            each element representing the number of danmaku in the corresponding second.
        """
        df = self.df[self.df["roomid"].eq(roomid)]
        start_time = df["time"].min()
        end_time = df["time"].max()
        duration = math.ceil(end_time - start_time)
        time = start_time + np.arange(duration)
        density = np.zeros(duration)
        for _time in df["time"]:
            density[int(_time - start_time)] += 1
        if smoothing_window > 0:
            density = np.convolve(density, np.ones(smoothing_window), "same") / smoothing_window
        return time, density

    def get_danmaku_density_gaussian(self, roomid: str, kernel_sigma: float) -> tuple(np.ndarray, np.ndarray):
        """Get danmaku density for a room using Gaussian kernel.

        Parameters:

            roomid: Room ID.

            kernel_sigma: Sigma of the Gaussian kernel that's used to smooth the density. In seconds.

        Returns:

            A tuple of numpy array. The first array is the timestamp. The second array is the danmaku density, with
            each element representing the number of danmaku in the corresponding second.
        """
        df = self.df[self.df["roomid"].eq(roomid)]
        start_time = df["time"].min()
        end_time = df["time"].max()
        duration = math.ceil(end_time - start_time)
        time = start_time + np.arange(duration)
        density = np.zeros(duration)
        for _time in df["time"]:
            left = max(int(_time - start_time - 3 * kernel_sigma), 0)
            right = min(int(_time - start_time + 3 * kernel_sigma), duration)
            density[left:right] += (
                1
                / (kernel_sigma * math.sqrt(2 * math.pi))
                * np.exp(-0.5 * ((time[left:right] - _time) / kernel_sigma) ** 2)
            )
        return time, density


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


def blacklist_filter(pool: DanmakuPool, blacklist: dict[str, list]) -> DanmakuPool:
    """Filter danmaku by blacklist.

    Parameters:

        pool: DanmakuPool to be filtered.

        blacklist: Dictionary of blacklists. The key is the room id, and the value is a list of banned danmaku.
        Example: {"123456": ["banned danmaku 1", "banned danmaku 2"], "654321": ["banned danmaku 3"]}

    Returns:

        DanmakuPool: Filtered DanmakuPool. The original DanmakuPool is not modified.
    """
    new_pool = DanmakuPool(pool.df)
    for roomid in blacklist:
        new_pool.df = new_pool.df[~new_pool.df["roomid"].eq(roomid) | ~new_pool.df["text"].isin(blacklist[roomid])]
    return new_pool


def get_clips_by_threshold(time: np.ndarray, density: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    """`deprecated`
        Get clips by threshold.

    Parameters:

        time: Time array.

        density: Density array.

        threshold: Threshold.

    Returns:

        List of tuples. Each tuple is a clip. The first element is the start time of the clip, and the second element is
        the end time of the clip.
    """
    clips = []
    for t, d in zip(time, density):
        if d > threshold:
            start = max(0, t - TIME_BACKWARD)
            end = min(time[-1], t + TIME_AFTERWARD)
            if len(clips) > 0 and start <= clips[-1][1]:
                clips[-1] = (clips[-1][0], end)
            else:
                clips.append((start, end))
    return clips


def get_hotspots(
    time: np.ndarray,
    density: np.ndarray,
    kernel_sigma: float,
    max_hotspots: int = 1000,
    show_plot: bool = False,
    show_progress: bool = False,
) -> list[tuple[float, float, float]]:
    """Detect hotspots by assuming that danmaku reactions are spread gaussian in time with respect to the climax event.

    Parameters:

        time: Time array.

        density: Density array.

        kernel_sigma: Sigma of the gaussian kernel used to smooth the density array.

        max_hotspots: Max number of hotspots to be detected. Default is 1000. Detection stops when the new hotspot
        causes the squared sum to increase.

    Returns:

        List of hotspots. Each hotspot is a tuple of time, amplitude and sigma (gaussian stdev).
    """
    MAX_SIGMA = 60
    if max_hotspots == 0:
        return []
    # find the value and time of the maximum density
    _amplitude = max(density)
    _time = time[np.argmax(density)]
    _sigma = kernel_sigma

    # define gaussian function
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

    # define chisquare function
    def chisquare(density):
        # does not allow negative density
        return sum(density[density > 0] ** 2) + sum(density[density < 0] ** 2) * 1e3

    # define the objective function, trim to the neighborhood of the hotspot to speed up the optimization
    def objective(x, idx):
        new_density = density[idx] - x[1] * gaussian(time[idx], x[0], x[2])
        return chisquare(new_density)

    # optimize time, amplitude and sigma
    left = max(0, _time - 5 * MAX_SIGMA)
    right = min(time[-1], _time + 5 * MAX_SIGMA)
    idx = (time >= left) & (time <= right)
    res = minimize(
        objective,
        [_time, _amplitude, _sigma],
        bounds=[(_time - 0.5 * _sigma, _time + 0.5 * _sigma), (_amplitude, _amplitude), (kernel_sigma, MAX_SIGMA)],
        args=(idx,),
        method="Nelder-Mead",
    )
    _time = res.x[0]
    _amplitude = res.x[1]
    _sigma = res.x[2]
    # calculate the new density curve
    new_density = density - _amplitude * gaussian(time, _time, _sigma)
    # if adding the hotspot does not improve the fit, stop
    if res.fun > chisquare(density[idx]):
        return []
    # show progress
    if show_progress:
        print("Residue:", chisquare(density))
    # return hotspots
    hotspots = [(_time, _amplitude, _sigma)]
    hotspots += get_hotspots(time, new_density, kernel_sigma, max_hotspots - 1, show_progress=show_progress)
    # if ended, show the figure using plotly
    if show_plot:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[datetime.datetime.fromtimestamp(_).isoformat() for _ in time],
                y=density,
                mode="lines",
                name="danmaku density",
                line=dict(color="dodgerblue"),
            )
        )
        for _time, _amplitude, _sigma in hotspots:
            idx = (time >= _time - 3 * _sigma) & (time <= _time + 3 * _sigma)
            # set the color of the hotspot to orange and hide the labels
            fig.add_trace(
                go.Scatter(
                    x=[datetime.datetime.fromtimestamp(_).isoformat() for _ in time[idx]],
                    y=gaussian(time[idx], _time, _sigma) * _amplitude,
                    mode="lines",
                    name="hotspot",
                    line=dict(color="orange"),
                    showlegend=False,
                )
            )
        _density = deepcopy(density)
        for _time, _amplitude, _sigma in hotspots:
            _density -= gaussian(time, _time, _sigma) * _amplitude
        fig.add_trace(
            go.Scatter(
                x=[datetime.datetime.fromtimestamp(_).isoformat() for _ in time],
                y=_density,
                mode="lines",
                name="residue",
                line=dict(color="tomato"),
            )
        )
        fig.show()
    return hotspots


# load config
with open("config.json", "r") as f:
    config = json.load(f)
    TIMEZONE = pytz.timezone(config["timezone"])
    BLACKLIST = config["blacklist"]
    SMOOTHING_WINDOW = config["smoothing_window"]
    TIME_BACKWARD = config["time_backward"]
    TIME_AFTERWARD = config["time_afterward"]
    KERNEL_SIGMA = config["kernel_sigma"]
