package test.app;

import android.util.Log;
import android.view.Surface;
import android.media.*;
import java.nio.ByteBuffer;
import java.io.IOException;

class Encoder {
    static String TAG = "TESTAPP_Encoder";

    MediaCodec codec;
    Surface surface;

    static MediaFormat format(int width, int height) {
        MediaFormat format = MediaFormat.createVideoFormat("video/avc", width, height);
        format.setInteger(MediaFormat.KEY_BIT_RATE, 8000000);
		format.setInteger(MediaFormat.KEY_FRAME_RATE, 15);
		format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 0);
		format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                          MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
		format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2);
		return format;
    }

    final MediaCodec.Callback callback = new MediaCodec.Callback() {
        public void onError(MediaCodec codec, MediaCodec.CodecException e) {
        }

        public void onInputBufferAvailable(MediaCodec codec, int index) {
        }

        public void onOutputBufferAvailable(MediaCodec codec, int index,
                                            MediaCodec.BufferInfo info) {
            ByteBuffer data = codec.getOutputBuffer(index);
            MainActivity.addbuffer(data, info.offset, info.size,
                                   info.presentationTimeUs, info.flags);
            codec.releaseOutputBuffer(index, false);
        }

        public void onOutputFormatChanged(MediaCodec codec, MediaFormat format) {
        }
    };

    Encoder() {
    }

    void open(int width, int height) {
        close();

        try {
            codec = MediaCodec.createEncoderByType("video/avc");
		} catch (IOException e) {
            Log.e(TAG, "", e);
		}

		codec.configure(format(width, height), null, null,
                        MediaCodec.CONFIGURE_FLAG_ENCODE);
        surface = codec.createInputSurface();
        codec.setCallback(callback, null);
        codec.start();
    }

    void close() {
        if (codec != null) {
            codec.stop();
            surface.release();
            codec = null;
        }
    }
}
