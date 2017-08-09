package test.app;

import android.util.Log;
import android.view.Surface;
import android.net.Uri;
import android.content.Context;
import android.media.*;
import java.util.LinkedList;
import java.io.IOException;

class Decoder {
    static String TAG = "TESTAPP_Decoder";

    MediaCodec codec;
    MediaExtractor extract;

    Surface surface;

    LinkedList<FrameInfo> fqueue = new LinkedList<FrameInfo>();
    long prev_time, prev_pts;
    boolean first_frame;

    static class FrameInfo {
        int index;
        MediaCodec.BufferInfo info;

        FrameInfo(int index, MediaCodec.BufferInfo info) {
            this.index = index;
            this.info = info;
        }
    }

    boolean process() {
        long curr_time = System.nanoTime() / 1000;

        while (fqueue.peek() != null) {
            FrameInfo fi = fqueue.peek();
            long delta = fi.info.presentationTimeUs - prev_pts;
            if (delta < 0)
                delta = 0;
            if (delta > 1000000) // max 1 second
                delta = 1000000;

            if (curr_time - prev_time < delta && !first_frame)
                break;

            fqueue.remove();
            prev_time = curr_time;
            prev_pts = fi.info.presentationTimeUs;
            first_frame = false;

            codec.releaseOutputBuffer(fi.index, true);
            if ((fi.info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0)
                return true;
        }
        return false;
    }

    final MediaCodec.Callback callback = new MediaCodec.Callback() {
        public void onError(MediaCodec codec, MediaCodec.CodecException e) {
        }

        public void onInputBufferAvailable(MediaCodec codec, int index) {
            if (extract == null)
                return;

            int size = extract.readSampleData(codec.getInputBuffer(index), 0);
            long timestamp = extract.getSampleTime();
            if (extract.advance() && size > 0) {
                codec.queueInputBuffer(index, 0, size, timestamp, 0);
            } else {
                codec.queueInputBuffer(index, 0, 0, 0,
                                       MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                extract.release();
                extract = null;
            }
        }

        public void onOutputBufferAvailable(MediaCodec codec, int index,
                                            MediaCodec.BufferInfo info) {
            fqueue.offer(new FrameInfo(index, info));
        }

        public void onOutputFormatChanged(MediaCodec codec, MediaFormat format) {
        }
    };

    Decoder(Surface surface) {
        this.surface = surface;
    }

    boolean open(Context context, Uri uri) {
        extract = new MediaExtractor();

        try {
            extract.setDataSource(context, uri, null);
        } catch (IOException e) {
            Log.e(TAG, "", e);
            return false;
        }

        for (int track = 0; track < extract.getTrackCount(); track++) {
            MediaFormat format = extract.getTrackFormat(track);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime.startsWith("video/")) {
                extract.selectTrack(track);
                try {
                    codec = MediaCodec.createDecoderByType(mime);
                } catch (IOException e) {
                    Log.e(TAG, "", e);
                    continue;
                }
                codec.configure(format, surface, null, 0);
                codec.setCallback(callback, null);
                codec.start();
                first_frame = true;
                return true;
            }
        }
        //failure
        extract.release();
        extract = null;
        return false;
    }

    void close() {
        if (extract != null) {
            extract.release();
            extract = null;
        }

        if (codec == null)
            return;

        codec.stop();
        codec.release();
        codec = null;

        while (fqueue.peek() != null)
            fqueue.remove();
    }
}
