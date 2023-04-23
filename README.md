# hotspot-clip

Using danmaku to find hotspot clips

## Requirements

This tool is to be used with [BililiveRecorder](https://github.com/BililiveRecorder/BililiveRecorder).

`Save Live Chat`: must be turned on

`Save Super Chat`: optional

`Split on Detection of Possible Data Loss`: turn on to get minimal frame-drop (which causes the video/danmaku
out-of-sync problem)

`Recording Splitting`: optional

## Danmaku Weight Evaluation Design

- Find correlations between danmaku. / Discover clasifications of danmaku.

- Get weight of danmaku from text, roomid, uid

## Updates

- Changed hotspot detection from plain threshold method to gaussian signal fitting.

## To-do:

- Can use simple neural network to filter danmaku and classify the clips based on the danmaku text and sender user id.

## Video-Danmaku out of Sync Problem

The video will get out of sync with the danmaku whenever frames are lost. This can be solved by turning on
`Split on Detection of Possible Data Loss`. This feature reduces the amount of time inconsistency to less than 10
seconds mostly. Old recordings can be fixed by splitting the video according to the associated `txt` file.
