# hotspot-clips

Using danmaku to find hotspot clips

## Requirements

This tool is to be used with [BililiveRecorder](https://github.com/BililiveRecorder/BililiveRecorder).

`Save Live Chat`: must be turned on

`Save Super Chat`: optional

`Split on Detection of Possible Data Loss`: turn on to get minimal frame-drop (which causes the video/danmaku
out-of-sync problem)

`Recording Splitting`: optional

## To-do:

- Maybe fourier transform the danmaku frequency to pick the peaks.

- Can use simple neural network to filter danmaku and classify the clips based on the danmaku text and sender user id.

## Video-Danmaku out of Sync Problem

The video will get out of sync with the danmaku whenever frames are lost. Assuming frame lost happens uniformly in
time, we recover the sync by scaling the danmaku. The workflow from raw recordings to clips is as follows:

- Encode the start and end time of the video in the filename. The end time can be found from the modified time of the
  flv file. Assuming the video and the danmaku files share the same last modificed time (might not be true), we can
  scale danmaku using the factor ( video duration / actual time ).

- Danmaku file start time encoded in the `<BililiveRecorderRecordInfo>` tag. Filename follows the associated video
  file. Each danmaku has attributes of a relative time and an absolute time. We use the absolute time. SC do not have
  absolute time, transform them into regular danmaku in advance.

- When splitting recordings, we split the video (update its filename as well) but keep the same danmaku files
  (duplicate) for all splitted videos.

- When we want to generate ass files or show the danmaku on the video player, we use the absolute time and the
  start/end time of the video, the scaling factor, to calculate the relative time.

I also want to try the other way: use danmaku time (actual time) overall, convert the time when dealing with video
clips. Segmenting streams more often can help with this issue, e.g. new part every 30 /60 mins.

Updates: Problem solved by turning on `Split on Detection of Possible Data Loss`. This feature reduces the amount of
time inconsistency to less than 10 seconds mostly. Old recordings can be fixed by splitting the video according to the
associated `txt` file.
