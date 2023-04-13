from __future__ import annotations
import os
import sys
import datetime
import pytz
import ffmpeg
import json
import math
import multiprocessing as mp
import lxml.etree as ET
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from copy import deepcopy
from scipy.optimize import minimize


class Keyword:
    def __init__(self, exact: list[str] = [], fuzzy: list[str] = []) -> None:
        self.exact = exact
        self.fuzzy = fuzzy

    def match(self, text: str) -> bool:
        if text in self.exact:
            return True
        for word in self.fuzzy:
            if word in text:
                return True
        return False


class DanmakuPool:
    """A pool of danmaku.

    Attributes:

        df: A pandas DataFrame with the following columns:

            time: Danmaku timestamp, timestamp, example: 1680514935.055

            offset: display time offset w.r.t. the timestamp, float, example: 1.214, -2.089

            roomid: room id, string, example: "92613"

            uid: user id, string, example: "74354088"

            uname: user name, string, example: "ymyyg"

            color: color, int, example: 16777215

            price: price in CNY if superchat, int, example: 0

            text: danmaku text, example: "233333"

            weight: weight of the danmaku, example: 1
    """

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
                    "weight",
                ]
            )
        else:
            self.df = deepcopy(df)

    def add_danmaku_from_xml(self, file_path: str, verbose: bool = False) -> None:
        """Add danmaku to the pool from xml file.

        The xml file is expected to have a <BililiveRecorderRecordInfo> tag with metadata, a <d> tag for each danmaku
        and a <sc> tag for each superchat.

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
            # if not Bilirec danmaku file, skip
            if info is None:
                return
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
                "weight": [],
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
                danmaku["text"].append(child.text if child.text else "")
                danmaku["weight"].append(1)
            for child in root.findall("sc"):
                danmaku["time"].append(start_time + float(child.attrib["ts"]))
                danmaku["offset"].append(0)
                danmaku["roomid"].append(roomid)
                danmaku["uid"].append(child.attrib["uid"])
                danmaku["uname"].append(child.attrib["user"])
                danmaku["color"].append(16772431)
                danmaku["price"].append(int(child.attrib["price"]))
                danmaku["text"].append(child.text if child.text else "")
                danmaku["weight"].append(1)
            self.df = pd.concat([self.df, pd.DataFrame(danmaku)], ignore_index=True)
            # print message if verbose
            if verbose:
                print(f"Loaded danmaku from {file_path}")
        except:
            print("Error: load_danmaku failed, file =", file_path, file=sys.stderr)

    def show_common_danmaku(self, num: int = 50) -> None:
        """Show the most frequent danmaku in the pool.

        Parameters:

            num: Number of danmaku to show. Default is 50.
        """
        for roomid in self.df["roomid"].unique():
            print(f"Room ID: {roomid}")
            df = self.df[self.df["roomid"].eq(roomid)]
            counter = Counter(df["text"])
            for text, count in counter.most_common(num):
                print(f"{count:>6} {text}")

    def filter(
        self, whitelist: list[Keyword] = None, blacklist: list[Keyword] = None, roomid: str = None
    ) -> DanmakuPool:
        """Filter danmaku in the pool.

        Parameters:

            whitelist: A list of allowed Keywords. If None, no danmaku will be filtered by the whitelist.
            Default is None.

            blacklist: A list of forbidden Keywords. If None, no danmaku will be filtered by the blacklist.

            roomid: Room ID. If None, all rooms are included. Default is None.

        Returns:

            A new DanmakuPool object. The original DanmakuPool object is not modified.
        """
        df = deepcopy(self.df)
        if roomid is not None:
            df = df[df["roomid"].eq(roomid)]
        if whitelist is not None:
            df = df[df["text"].apply(lambda x: any([keyword.match(x) for keyword in whitelist]))]
        if blacklist is not None:
            df = df[df["text"].apply(lambda x: not any([keyword.match(x) for keyword in blacklist]))]
        return DanmakuPool(df)

    def segments(self) -> list[DanmakuPool]:
        """Split the pool into a list of DanmakuPool objects, each containing danmaku from a single room within a
        single stream. Streams are considered to be separated by a gap of more than 1 hour.

        Returns:

            A list of DanmakuPool objects.
        """
        seg = []
        for roomid in self.df["roomid"].unique():
            df = self.filter(roomid=roomid).df.sort_values(by="time")
            time = df["time"].values
            left, right = 0, 1
            while right < len(df):
                if time[right] - time[right - 1] > 3600:
                    seg.append(DanmakuPool(df.iloc[left:right]))
                    left = right
                right += 1
            seg.append(DanmakuPool(df.iloc[left:right]))
        return seg

    def get_danmaku_density(self, roomid: str, kernel_sigma: float) -> tuple(np.ndarray, np.ndarray):
        """Get danmaku density for a room using Gaussian kernel.

        Parameters:

            roomid: Room ID.

            kernel_sigma: Sigma of the Gaussian kernel that's used to smooth the density. In seconds.

        Returns:

            A tuple of numpy array. The first array is the timestamp. The second array is the danmaku density, with
            each element representing the number of danmaku in the corresponding second.
        """
        df = self.df[self.df["roomid"].eq(roomid)]
        if df.empty:
            print(f"Warning: get_danmaku_density_gaussian: no danmaku found, roomid = {roomid}", file=sys.stderr)
            return np.array([]), np.array([])
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

    def export_xml(self, roomid: str, start_time: float, end_time: float, file_path: str) -> None:
        """Export danmaku to an XML file.

        Parameters:

            roomid: Room ID.

            start_time: Start time of the danmaku. Timestamp.

            end_time: End time of the danmaku. Timestamp.

            file_path: Path to the XML file.
        """
        df = self.df[self.df["roomid"].eq(roomid)]
        df = df[df["time"].between(start_time, end_time)]
        root = ET.Element("i")
        for _, row in df.iterrows():
            d = ET.SubElement(root, "d")
            d.text = row["text"]
            mode = "1"
            if row["price"] > 0:
                mode = "3"
                d.text = f"【SC¥{row['price']}】" + row["uname"] + ":" + d.text
            d.set(
                "p",
                "{:.3f},{},25,{},{},0,{},0".format(
                    row["time"] - start_time,
                    mode,
                    row["color"],
                    int(row["time"] * 1000),
                    row["uid"],
                ),
            )
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True, pretty_print=True)


class VideoPool:
    def __init__(self) -> None:
        self.df = pd.DataFrame(
            columns=[
                "path",
                "roomid",
                "creation_time",
                "duration",
            ]
        )

    def add_video(self, video_path: str) -> None:
        """Add a video to the pool.

        Parameters:

            video_path: Path to the video.
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
                elif line.startswith("B站直播间"):
                    roomid = line.split(" ")[1]
            duration = float(output["format"]["duration"])
            # try to correct creation time for videos that were split afterwards
            try:
                time_str = video_path.split(roomid + "_")[1][:15]
                creation_time_from_filename = datetime.datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                creation_time_from_filename = TIMEZONE.localize(creation_time_from_filename)
                if abs((creation_time_from_filename - creation_time).total_seconds()) > 1:
                    creation_time = creation_time_from_filename
                    print("Warning: add_video: corrected creation time, file =", video_path, file=sys.stderr)
            except:
                pass
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        {
                            "path": [video_path],
                            "roomid": [roomid],
                            "creation_time": [creation_time],
                            "duration": [duration],
                        }
                    ),
                ],
                ignore_index=True,
            )
        except:
            print("Error: add_video failed, file =", video_path, file=sys.stderr)

    def generate_clips(
        self, roomid: str, clips: pd.DataFrame, out_dir: str, danmaku_pool: DanmakuPool = None, num_threads: int = 1
    ) -> None:
        """Generate clips from the videos in the pool according to the clips information.

        Parameters:

            roomid: Room ID.

            clips: A pandas DataFrame containing the clips information. The columns should be "start", "end",
            "amplitude".

            out_dir: Output directory.

            danmaku_pool: A DanmakuPool object. If not None, xml files will be added along with the clips.
            Default is None.

            num_threads: Number of threads to use. Default is 1.

        Note: This function uses multiprocessing to generate clips.
        """
        # if out_dir does not exist or  is not a directory, stop
        if not os.path.exists(out_dir) or not os.path.isdir(out_dir):
            print(f"Error: generate_clips: out_dir does not exist, out_dir = {out_dir}", file=sys.stderr)
            return
        # find videos
        df = self.df[self.df["roomid"].eq(roomid)]
        if df.empty:
            print(f"Warning: generate_clips: no video found, roomid = {roomid}", file=sys.stderr)
            return
        # generate clips
        args = []
        summary = pd.DataFrame(columns=["file", "amplitude"])
        for _, row in clips.iterrows():
            start = row["start"]
            end = row["end"]
            for _, video_row in df.iterrows():
                video_path = video_row["path"]
                video_start = video_row["creation_time"].timestamp()
                video_end = video_start + video_row["duration"]
                if start < video_end and end > video_start:
                    actual_start = max(start, video_start)
                    actual_end = min(end, video_end)
                    ss = actual_start - video_start
                    t = actual_end - actual_start
                    out_file = "{}_{}_{:02d}{:02d}{:02d}.mp4".format(
                        roomid,
                        video_row["creation_time"].strftime("%Y%m%d_%H%M%S"),
                        int(ss) // 3600,
                        (int(ss) % 3600) // 60,
                        int(ss) % 60,
                    )
                    out_path = os.path.join(out_dir, out_file)
                    args.append((video_path, ss, t, out_path))
                    # output danmaku xml file
                    if danmaku_pool is not None:
                        print(f"Generating danmaku xml file: {out_path[:-4] + '.xml'}", flush=True)
                        danmaku_pool.export_xml(roomid, actual_start, actual_end, out_path[:-4] + ".xml")
                    # update summary
                    summary = pd.concat(
                        [
                            summary,
                            pd.DataFrame(
                                {
                                    "file": [out_file],
                                    "amplitude": [row["amplitude"]],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
        with mp.Pool(num_threads) as pool:
            pool.starmap(self._generate_clip_mp, args)
        # output summary
        summary = summary.sort_values(by="amplitude", ascending=False).reset_index(drop=True)
        summary.to_csv(os.path.join(out_dir, f"{roomid}_summary.csv"), index=False)

    def _generate_clip_mp(self, video_path: str, ss: float, t: float, out_path: str) -> None:
        """Generate a clip using ffmpeg.

        Parameters:

            video_path: Path to the video.

            ss: Start time in seconds.

            t: Duration in seconds.

            out_path: Output path.
        """
        print(f"Generating clip: {out_path}", flush=True)
        # generate the clip without re-encoding, overwrite
        ffmpeg.input(video_path, ss=ss, t=t).output(out_path, vcodec="copy", acodec="copy").run_async(
            overwrite_output=True,
            quiet=True,
        )


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


def get_hotspots(
    time: np.ndarray,
    density: np.ndarray,
    kernel_sigma: float,
    max_hotspots: int = 1000000000,
    show_plot: bool = False,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Detect hotspots by assuming that danmaku reactions are spread gaussian in time with respect to the climax event.
    Fitting the density curve with amplitude * exp(- (time - time_of_climax) ** 2 / (2 * sigma ** 2)).

    Parameters:

        time: Time array.

        density: Density array.

        kernel_sigma: Sigma of the gaussian kernel used to smooth the density array.

        max_hotspots: Max number of hotspots to be detected. Default is 1000. Detection stops when the new hotspot
        causes the squared sum to increase.

        show_plot: Whether to show the plot. Default is False.

        show_progress: Whether to show the progress of regression. Default is False.

    Returns:

        A DataFrame with columns "time", "amplitude", "sigma".
    """
    MAX_SIGMA = 60
    # show progress
    if show_progress:
        print("Residue:", chisquare(density))
    # recursion stop condition
    if max_hotspots == 0:
        return pd.DataFrame(columns=["time", "amplitude", "sigma"])
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
    left = max(time[0], _time - 5 * MAX_SIGMA)
    right = min(time[-1], _time + 5 * MAX_SIGMA)
    idx = (time >= left) & (time <= right)
    res = minimize(
        objective,
        [_time, _amplitude, _sigma],
        bounds=[(_time - 0.5 * _sigma, _time + 0.5 * _sigma), (_amplitude, _amplitude), (kernel_sigma, MAX_SIGMA)],
        args=(idx,),
        method="Nelder-Mead",
    )
    _time, _amplitude, _sigma = res.x
    # if adding the hotspot does not improve the fit, stop
    if res.fun > chisquare(density[idx]):
        return pd.DataFrame(columns=["time", "amplitude", "sigma"])
    # calculate the new density curve
    new_density = density - _amplitude * gaussian(time, _time, _sigma)
    # return hotspots
    hotspots = pd.DataFrame([[_time, _amplitude, _sigma]], columns=["time", "amplitude", "sigma"])
    hotspots = pd.concat(
        [hotspots, get_hotspots(time, new_density, kernel_sigma, max_hotspots - 1, show_progress=show_progress)],
        ignore_index=True,
    )
    # if ended, show the figure using plotly
    if show_plot:
        # subsample density to speed up plotting
        if len(density) < 2e5:
            subsample = 4
        else:
            subsample = 16
        density = density[::subsample]
        time = time[::subsample]
        # remove unncessary points where density is zero
        idx = density > 0
        time = time[idx]
        density = density[idx]
        # create a figure with two subplots, let the right one 1/4 of the width of the left one
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[9, 1], horizontal_spacing=0)
        fig.add_trace(
            go.Scatter(
                x=[datetime.datetime.fromtimestamp(_).isoformat() for _ in time],
                y=density,
                mode="lines",
                name="danmaku density",
                line=dict(color="dodgerblue"),
            ),
            row=1,
            col=1,
        )
        for _, row in hotspots.iterrows():
            idx = (time >= row["time"] - 3 * row["sigma"]) & (time <= row["time"] + 3 * row["sigma"])
            # set the color of the hotspot to orange and hide the labels
            fig.add_trace(
                go.Scatter(
                    x=[datetime.datetime.fromtimestamp(_).isoformat() for _ in time[idx]],
                    y=gaussian(time[idx], row["time"], row["sigma"]) * row["amplitude"],
                    mode="lines",
                    name="hotspot",
                    line=dict(color="orange"),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        _density = deepcopy(density)
        for _, row in hotspots.iterrows():
            _density -= gaussian(time, row["time"], row["sigma"]) * row["amplitude"]
        fig.add_trace(
            go.Scatter(
                x=[datetime.datetime.fromtimestamp(_).isoformat() for _ in time],
                y=_density,
                mode="lines",
                name="residue",
                line=dict(color="tomato"),
            ),
            row=1,
            col=1,
        )
        # show the amplitudes of the hotspots as a histogram, y-axis is the amplitude, x-axis is the number of hotspots
        fig.add_trace(
            go.Histogram(
                y=hotspots["amplitude"],
                name="amplitude",
                marker_color="dodgerblue",
                orientation="h",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        # add vertical lines at the median, 3/4 quantiles, with different colors
        fig.add_shape(
            type="line",
            x0=0,
            y0=hotspots["amplitude"].median(),
            x1=hotspots.shape[0],
            y1=hotspots["amplitude"].median(),
            line=dict(color="tomato", dash="dash"),
            row=1,
            col=2,
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=hotspots["amplitude"].quantile(0.75),
            x1=hotspots.shape[0],
            y1=hotspots["amplitude"].quantile(0.75),
            line=dict(color="tomato", dash="dash"),
            row=1,
            col=2,
        )
        # move the legend on top of the figure, in the center, show in horizontal direction
        fig.update_layout(legend=dict(xanchor="center", yanchor="bottom", x=0.5, y=1, orientation="h"))
        # reduce the margin of the figure
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        # show the figure
        fig.show()
    return hotspots


def get_clips(hotspots: pd.DataFrame, percentile: float = 0, threshold: float = 0, num: int = 0) -> pd.DataFrame:
    """Get the start and end time of the clips.

    Parameters:

        hotspots: DataFrame of hotspots.

        percentile: the percentile of the amplitudes of the hotspots to return. 0 <= percentile <= 1. If 0, return all
        clips. If non-zero, return the clips with the amplitudes higher than the percentile. Default is 0.

        threshold: the threshold of the amplitudes of the hotspots to return. If 0, return all clips. If non-zero,
        return the clips with the amplitudes higher than the threshold. Default is 0.

        num: number of clips to return. If 0, return all clips. If non-zero, return the top-N clips with the
        highest amplitudes. Default is 0. Note that the number of returned clips might be smaller if clips merge.

    Returns:

        A DataFrame with columns "start", "end", "amplitude".
    """
    # select the hotspots with the amplitudes higher than the percentile
    if percentile > 0:
        hotspots = hotspots[hotspots["amplitude"] >= hotspots["amplitude"].quantile(percentile)]
    # select the hotspots with the amplitudes higher than the threshold
    if threshold > 0:
        hotspots = hotspots[hotspots["amplitude"] >= threshold]
    # select the top-N hotspots with the highest amplitudes
    if num > 0 and num < hotspots.shape[0]:
        hotspots = hotspots.sort_values(by="amplitude", ascending=False).iloc[:num]
    # sort the hotspots by the time
    hotspots = hotspots.sort_values(by="time")
    # get the start and end time of the clips
    clips = pd.DataFrame(columns=["start", "end", "amplitude"])
    for _, row in hotspots.iterrows():
        start = row["time"] - 2 * row["sigma"] - TIME_BACKWARD
        end = row["time"] + 2 * row["sigma"] + TIME_AFTERWARD
        amplitude = row["amplitude"]
        if clips.empty:
            clips = pd.DataFrame([[start, end, amplitude]], columns=["start", "end", "amplitude"])
        else:
            if start <= clips.iloc[-1]["end"]:
                clips.iloc[-1]["end"] = end
                clips.iloc[-1]["amplitude"] = max(clips.iloc[-1]["amplitude"], amplitude)
            else:
                clips = pd.concat(
                    [clips, pd.DataFrame([[start, end, amplitude]], columns=["start", "end", "amplitude"])],
                    ignore_index=True,
                )
    # sammary message
    print(
        "Total {} clips. Duration: {}. Lowest amplitude: {:.2f}.".format(
            clips.shape[0],
            datetime.timedelta(seconds=clips["end"].sum() - clips["start"].sum()),
            hotspots["amplitude"].min(),
        ),
    )
    return clips


# load config
with open("config.json", "r") as f:
    config = json.load(f)
    TIMEZONE = pytz.timezone(config["timezone"])
    BLACKLIST = config["blacklist"]
    SMOOTHING_WINDOW = config["smoothing_window"]
    TIME_BACKWARD = config["time_backward"]
    TIME_AFTERWARD = config["time_afterward"]
    KERNEL_SIGMA = config["kernel_sigma"]
    KEYWORDS = config["keywords"]
