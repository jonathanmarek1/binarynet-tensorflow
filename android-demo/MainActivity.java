// TODO
// camera matrix
// dont call setCamera twice on init
// user control over codec parameters
// setting: clip location
// open clip and play

//detect cameras and resolutions

package test.app;

import android.view.*;
import android.widget.*;
import android.util.*;
import android.graphics.*;
import android.hardware.camera2.*;
import android.os.*;
import android.content.*;
import android.media.*;
import android.app.*;
import android.preference.*;
import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.*;
import java.util.Arrays;
import java.util.LinkedList;
import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends Activity implements SurfaceHolder.Callback2,
    SharedPreferences.OnSharedPreferenceChangeListener {
    static {
        System.loadLibrary("hello-jni");
    }

    static native int init(Surface surface);
    static native int exit();
    static native void draw(int encode);

    static native void setcodecsurface(Surface surface);
    static native void created(Surface surface);
    static native void changed(Surface surface, int fmt, int width, int height);
    static native void destroyed();

    static native void addbuffer(ByteBuffer buffer, int offset, int size,
                                 long timestamp, int flags);
    static native void writemux(String path);

    static native String getoverlay();


    static String TAG = "MainActivityTESTAPP";
    static String VERSION = "0";

    MediaCodec codec, decoder;
    MediaFormat format;

    MediaExtractor extract;

    Surface enc_surface = null;

    String camera_id;
    int num_opened = 0;
    CameraDevice camera;
    Surface surface;
    SurfaceTexture surface_texture;

    SurfaceView mView;
    TextView overlay;

    SharedPreferences prefs;

    static void printf(String format, Object... arguments) {
        Log.e(TAG, String.format(format, arguments));
    }

    CaptureRequest build_request() throws CameraAccessException {
        CaptureRequest.Builder req;

        req = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
        req.addTarget(surface);
        req.set(CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
        return req.build();
    }

    final CameraDevice.StateCallback state_callback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice cam) {
            /* if multiple cameras opened, only keep the last one
                TODO: does this method always work?
            */
            printf("ONOPENED %d\n", num_opened);
            if (num_opened > 1) {
                num_opened--;
                cam.close();
                return;
            }

            try {
                if (camera != null)
                    throw new RuntimeException("camera is not null");
                camera = cam;
                camera.createCaptureSession(Arrays.asList(surface),
                                            capture_state_callback, null);
            } catch (CameraAccessException e) {
                Log.e(TAG, "", e);
            }
        }

        /* is cam.close() necessary? */
        @Override
        public void onDisconnected(CameraDevice cam) {
            printf("DISCONNECTED\n");
            num_opened--;
            if (camera == cam)
                camera = null;
            cam.close();
        }

        @Override
        public void onError(CameraDevice cam, int error) {
            onDisconnected(cam);
        }
    };

    final CameraCaptureSession.StateCallback capture_state_callback =
        new CameraCaptureSession.StateCallback() {
        @Override
        public void onConfigured(CameraCaptureSession session) {
            if (camera == null)
                return;

            try {
                session.setRepeatingRequest(build_request(), capture_callback, null);
            } catch (CameraAccessException e) {
                Log.e(TAG, "", e);
            }
        }

        @Override
        public void onConfigureFailed(CameraCaptureSession session) {
            printf("onConfigureFailed");
        }
    };

    final CameraCaptureSession.CaptureCallback capture_callback =
        new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureProgressed(CameraCaptureSession session, CaptureRequest request, CaptureResult result) {
        }

        @Override
        public void onCaptureCompleted(CameraCaptureSession session, CaptureRequest request, TotalCaptureResult result) {
            surface_texture.updateTexImage();
            //printf("get frame");
            draw(1);
        }
    };

    public void surfaceCreated(SurfaceHolder holder) {
        created(holder.getSurface());
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        printf("%d %d %d\n", format, width, height);
        changed(holder.getSurface(), format, width, height);
    }

    public void surfaceRedrawNeeded(SurfaceHolder holder) {
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        destroyed();
    }

    final MediaCodec.Callback codec_callback = new MediaCodec.Callback() {
        public void onError(MediaCodec codec, MediaCodec.CodecException e) {
            printf("onError\n");
        }

        public void onInputBufferAvailable(MediaCodec codec, int index) {
            printf("onInputBufferAvailable\n");
        }

        public void onOutputBufferAvailable(MediaCodec codec, int index,
                                            MediaCodec.BufferInfo info) {
            ByteBuffer data = codec.getOutputBuffer(index);
            addbuffer(data,info.offset,info.size,info.presentationTimeUs,info.flags);
            codec.releaseOutputBuffer(index, false);
        }

        public void onOutputFormatChanged(MediaCodec codec, MediaFormat format) {
            printf("onOutputFormatChanged\n");
        }
    };

    static class FrameInfo {
        int index;
        MediaCodec.BufferInfo info;

        FrameInfo(int index, MediaCodec.BufferInfo info) {
            this.index = index;
            this.info = info;
        }
    }

    LinkedList<FrameInfo> fqueue = new LinkedList<FrameInfo>();
    long prev_time, prev_pts;
    boolean first_frame;

    final MediaCodec.Callback decode_callback = new MediaCodec.Callback() {
        public void onError(MediaCodec codec, MediaCodec.CodecException e) {
            printf("onError\n");
        }

        public void onInputBufferAvailable(MediaCodec codec, int index) {
            //printf("onInputBufferAvailable\n");
            if (extract == null)
                return;

            int size = extract.readSampleData(decoder.getInputBuffer(index), 0);
            long timestamp = extract.getSampleTime();
            if (extract.advance() && size > 0) {
                decoder.queueInputBuffer(index, 0, size, timestamp, 0);
            } else {
                decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                extract.release();
                extract = null;
            }
        }

        public void onOutputBufferAvailable(MediaCodec codec, int index,
                                            MediaCodec.BufferInfo info) {
            //printf("onOutputBufferAvailable\n");
            fqueue.offer(new FrameInfo(index, info));
        }

        public void onOutputFormatChanged(MediaCodec codec, MediaFormat format) {
            printf("onOutputFormatChanged\n");
        }
    };

    void release_and_handle_eos(int index, MediaCodec.BufferInfo info) {
        decoder.releaseOutputBuffer(index, true);
        if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
            printf("done\n");
            decoder.stop();
            decoder.release();
            decoder = null;
        }
    }

    final Runnable decode_draw = new Runnable() {
        public void run() {
            if (decoder == null) { // when decoding was stopped
                while (fqueue.peek() != null)
                    fqueue.remove();
                openCamera();
                return;
            }

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

                release_and_handle_eos(fi.index, fi.info);
            }

            surface_texture.updateTexImage();
            draw(0);

            if (decoder != null)
                (new Handler()).post(decode_draw);
            else
                openCamera();
        }
    };

    static String[] keys = {
        "network_id", "camera_id", "camera_res", "area_select"
    };
    static CharSequence[] titles = {
        "Network Type", "Camera", "Camera Resolution", "Area Select"
    };
    static CharSequence[] networks = {"BWN", "XNORNET"};
    static CharSequence[] cameras = {"0", "1"};

    Preference pref[] = new Preference[4];


    final PreferenceFragment settings = new PreferenceFragment() {
        @Override
        public void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            Context context = getActivity();
            PreferenceScreen screen =
                getPreferenceManager().createPreferenceScreen(context);

            ListPreference lp;
            SwitchPreference sp;

            pref[0] = new ListPreference(context);
            pref[1] = new ListPreference(context);
            pref[2] = new ListPreference(context);
            pref[3] = new SwitchPreference(context);

            lp = (ListPreference) pref[0];
            lp.setEntries(networks);
            lp.setEntryValues(networks);

            lp = (ListPreference) pref[1];
            lp.setEntries(cameras);
            lp.setEntryValues(cameras);

            lp = (ListPreference) pref[2];
            lp.setEntries(networks);
            lp.setEntryValues(networks);


            for (int i = 0; i < pref.length; i++) {
                updateSummary(prefs, keys[i], i);
                pref[i].setKey(keys[i]);
                pref[i].setTitle(titles[i]);
                screen.addPreference(pref[i]);
            }

            Preference pr;

            pr = new Preference(context);
            pr.setTitle("Take Snap");
            pr.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener() {
                @Override
                public boolean onPreferenceClick(Preference preference) {
                    writemux("/sdcard/clip.mp4");
                    return true;
                }
            });
            screen.addPreference(pr);

            pr = new Preference(context);
            pr.setTitle("Play Snap");
            pr.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener() {
                @Override
                public boolean onPreferenceClick(Preference preference) {
                    if (decoder == null) {
                        openClip("/sdcard/clip.mp4");
                        preference.setTitle("Stop Snap");
                    } else {
                        if (extract != null) {
                            extract.release();
                        }
                        decoder.stop();
                        decoder.release();
                        decoder = null;

                        preference.setTitle("Play Snap");
                    }

                    return true;
                }
            });
            screen.addPreference(pr);

            setPreferenceScreen(screen);
        }
    };

    void updateSummary(SharedPreferences prefs, String key, int i) {
        String summary;
        if (i < 3)
            summary = prefs.getString(key, "");
        else {
            printf("KEY %s %d\n", key, i);
            summary = prefs.getBoolean(key, false) ? "On" : "Off";
        }
        pref[i].setSummary(summary);
    }

    void applySetting(SharedPreferences prefs, String key, int i) {
        if (i < 3) {
            String string = prefs.getString(key, "0");
            if (i == 0)
                ; //use_bwn = string == "BWN";
            else if (i == 1)
                setCameraName(string);
            else if (i == 2)
                setCameraResolution(string);
        } else {
            boolean bool = prefs.getBoolean(key, false);
            //use_area = bool;
        }
    }

    public void onSharedPreferenceChanged(SharedPreferences prefs, String key) {
        int i;
        for (i = 0; i < pref.length; i++)
            if (key == keys[i])
                break;
        if (i == pref.length)
            return;

        updateSummary(prefs, key, i);
        applySetting(prefs, key, i);
    }

    MediaFormat format(int width, int height) {
        format = MediaFormat.createVideoFormat("video/avc", width, height);
        format.setInteger(MediaFormat.KEY_BIT_RATE, 8000000);
		format.setInteger(MediaFormat.KEY_FRAME_RATE, 15);
		format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 0);
		format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                          MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
		format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2);
		return format;
    }

    void closeCamera() {
        if (camera != null) {
            num_opened--;
            camera.close();
            camera = null;
        }
    }

    void openCamera() {
        printf("DO SET CAMERA: %s\n", camera_id);

        if (decoder != null)
            return;

        closeCamera();

        int width = 640, height = 480;

        /* change codec resolution,
            is it really necessary to recreate the whole codec? */
        if (enc_surface != null) {
            codec.stop();
            enc_surface.release();
        }
        try {
            codec = MediaCodec.createEncoderByType("video/avc");
		} catch (IOException e) {
            Log.e(TAG, "", e);
            return;
		}
        codec.configure(format(width, height),
                        null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        enc_surface = codec.createInputSurface();
        codec.setCallback(codec_callback, null);
        codec.start();
        setcodecsurface(enc_surface);

        /* change capture resolution */
        surface_texture.setDefaultBufferSize(width, height);

        /* new camera open request */
        CameraManager manager=(CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            num_opened++;
            manager.openCamera(camera_id, state_callback, null);
        } catch (CameraAccessException e) {
            Log.e(TAG, "", e);
            return;
        }
    }

    void openClip(String path) {
        closeCamera();

        extract = new MediaExtractor();
        MediaFormat format;

        try {
            extract.setDataSource(path);
        } catch (IOException e) {
            Log.e(TAG, "", e);
            return;
        }

        // get id of video track
        int track;
        for (track = 0; track < extract.getTrackCount(); track++) {
            format = extract.getTrackFormat(track);
            if (format.getString(MediaFormat.KEY_MIME).startsWith("video/")) {
                extract.selectTrack(track);
                try {
                    decoder = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME));
                } catch (IOException e) {
                    Log.e(TAG, "", e);
                    continue;
                }
                decoder.configure(format, surface, null, 0);
                decoder.setCallback(decode_callback, null);
                decoder.start();
                 (new Handler()).post(decode_draw);
                first_frame = true;
                return;
            }
        }

        extract.release();
        extract = null;
    }

    void setCameraName(String name) {
        camera_id = name;
        openCamera();
    }

    void setCameraResolution(String name) {
        // openCamera();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        prefs = PreferenceManager.getDefaultSharedPreferences(this);
        if (prefs.getString("version", "NONE") == "NONE") {
            SharedPreferences.Editor edit = prefs.edit();
            edit.putString("version", VERSION);
            edit.putString(keys[0], "BWN");
            edit.putString(keys[1], "0");
            edit.putString(keys[2], "640x480");
            edit.commit();
        }
        prefs.registerOnSharedPreferenceChangeListener(this);


        overlay = new TextView(this);
        overlay.setTextColor(Color.WHITE);
        overlay.setShadowLayer(2.0f, 0.0f, 0.0f, Color.BLACK);
        overlay.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        overlay.setText("test text");
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        params.gravity = Gravity.TOP | Gravity.LEFT;

        mView = new SurfaceView(this);
        mView.getHolder().addCallback(this);

        setContentView(mView);
        addContentView(overlay, params);

        /*Button test = new Button(this);
        test.setText("hello button");
        test.setBackgroundResource(R.drawable.my_button);
        test.setOnClickListener(this);
        //test.setBackgroundColor(Color.TRANSPARENT);
        //test.setImageResource(android.R.drawable.btn_radio);

        addContentView(test, params);*/

        /* Toast.makeText(this, "hello", 0).show();

        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*//*");
        startActivityForResult(intent, 42); */

        /* the init function needs a surface for egl, get one from a dummy codec
           otherwise this codec creation code could be removed
           *there is a probably a better way to get a temporary surface
        */
        try {
            codec = MediaCodec.createEncoderByType("video/avc");
		} catch (IOException e) {
            throw new RuntimeException("createEncoderByType\n");
		}
        codec.configure(format(640, 480),
                        null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        enc_surface = codec.createInputSurface();
        codec.setCallback(codec_callback, null);
        codec.start();

        int texture_id = init(enc_surface);

        surface_texture = new SurfaceTexture(texture_id);
        surface = new Surface(surface_texture);

        for (int i = 0; i < pref.length; i++)
            applySetting(prefs, keys[i], i);

        final CountDownTimer timer = new CountDownTimer(30000000000l, 100) {
            public void onTick(long millisUntilFinished) {
                overlay.setText(getoverlay());
            }

            public void onFinish() {
                //mTextField.setText("done!");
            }
        };
        timer.start();
    }

    @Override
    protected void onPause() {
        super.onPause();
        closeCamera();
    }

    @Override
    protected void onResume() {
        super.onResume();
        openCamera();
    }

    /* use the menu button to toggle preferences
        is there a better way? */
    boolean toggle = false;

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        FragmentManager fm = getFragmentManager();
        if (!toggle)
            fm.beginTransaction().replace(android.R.id.content, settings).commit();
        else
            fm.beginTransaction().remove(settings).commit();

        overlay.setVisibility(toggle ? TextView.VISIBLE : TextView.INVISIBLE);

        toggle = !toggle;
        return false;
    }
}
