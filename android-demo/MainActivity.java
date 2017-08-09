/*
Java part of the app, manages configuration, camera, video codecs, etc.

two exclusive inputs: Camera and Decoder, both write to the same SurfaceTexture

SurfaceTexture is used to render to a SurfaceView using opengl es2.0 (native code)

rendering is done as often as possible (Vsync) while the SurfaceView is active:
    *when input is camera, the latest frame is displayed
    *when decoder is input, there is some logic to time frames correctly

when camera is input the input is also rendered to an encoder surface for recording
    *this is done in response to onCaptureCompleted events

the overlay text (class+probabilities) is updated every 100ms (CountDownTimer) to a
string obtained from the native code

encoded data is passed on to native code where it is stored in a circular buffer

when the "snapshot" button is pressed, a filename /sdcard/clipXXXXX.mp4 is generated,
where XXXXX is an incremental value stored using preferences, and passed on to native
code which creates the mp4 file

some TODOs:
-detect cameras and resolutions instead of hardcoded options
-way to stop clip (currently possible by pausing and resuming activity)
-add some user control over codec parameters
-split preferences stuff into seperate class
-some cleaning up
-NETWORK SELECT
*/

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
import android.database.Cursor;
import android.net.Uri;
import android.provider.MediaStore;
import javax.microedition.khronos.opengles.GL10;
import javax.microedition.khronos.egl.*;
import java.io.*;
import java.nio.ByteBuffer;

public class MainActivity extends Activity implements SurfaceHolder.Callback2,
    SharedPreferences.OnSharedPreferenceChangeListener {
    static {
        System.loadLibrary("hello-jni");
    }

    static native int init(Surface surface);
    static native int exit();

    static native void draw(float[] mtx);
    static native void encode(float[] mtx);

    static native void setcodecsurface(Surface surface);
    static native void created(Surface surface);
    static native void changed(Surface surface, int fmt, int width, int height);
    static native void destroyed();

    static native void setnetwork(int id);

    static native void addbuffer(ByteBuffer buffer, int offset, int size,
                                 long timestamp, int flags);
    static native void writemux(String path);

    static native String getoverlay();

    static String TAG = "MainActivityTESTAPP";
    static String VERSION = "0";

    Camera camera;
    Decoder decoder;
    Encoder encoder;

    Surface surface;
    SurfaceTexture surface_texture;

    SurfaceView mView;
    boolean have_surface = false, draw_called = false;

    TextView overlay;
    ImageButton button;

    SharedPreferences prefs;

    String camera_id;
    int camera_width, camera_height;

    static void printf(String format, Object... arguments) {
        Log.e(TAG, String.format(format, arguments));
    }

    public void surfaceCreated(SurfaceHolder holder) {
        have_surface = true;
        created(holder.getSurface());

        // TODO
        if (decoder.codec == null)
            openCamera();

        if (!draw_called) {
            draw_called = true;
            (new Handler()).post(draw_call);
        }
    }

    public void surfaceChanged(SurfaceHolder holder, int fmt, int width, int height) {
        changed(holder.getSurface(), fmt, width, height);
    }

    public void surfaceRedrawNeeded(SurfaceHolder holder) {
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        have_surface = false;
        destroyed();
        camera.close();
        decoder.close();
    }

    final Runnable draw_call = new Runnable() {
        public void run() {
            if (!have_surface) {
                draw_called = false;
                return;
            }

            if (decoder.codec != null) {
                if (decoder.process()) { // TODO
                    decoder.close();
                    openCamera();
                }
                surface_texture.updateTexImage();
            }

            float mtx[] = new float[16];
            surface_texture.getTransformMatrix(mtx);
            draw(mtx);

            (new Handler()).post(draw_call);
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
    static CharSequence[] resolutions = {"640x480", "1920x1080"};

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
            lp.setEntries(resolutions);
            lp.setEntryValues(resolutions);


            for (int i = 0; i < pref.length; i++) {
                updateSummary(prefs, keys[i], i);
                pref[i].setKey(keys[i]);
                pref[i].setTitle(titles[i]);
                screen.addPreference(pref[i]);
            }

            Preference pr;

            pr = new Preference(context);
            pr.setTitle("Open Clip");
            pr.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener() {
                @Override
                public boolean onPreferenceClick(Preference preference) {
                    Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                    intent.addCategory(Intent.CATEGORY_OPENABLE);
                    intent.setType("*/*");
                    MainActivity.this.startActivityForResult(intent, 0);
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
            if (i == 0) {
                setnetwork(string == "BWN" ? 0 : 1);
            } else if (i == 1) {
                camera_id = string;
                // camera_width / camera_height reset
                openCamera();
            } else if (i == 2) {
                String[] str = string.split("x");
                camera_width = Integer.parseInt(str[0]);
                camera_height = Integer.parseInt(str[1]);
                openCamera();
            }
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

    void openCamera() {
        if (decoder.codec != null || !have_surface)
            return;

        camera.close();

        encoder.open(camera_width, camera_height);
        setcodecsurface(encoder.surface);

        /* change capture resolution */
        surface_texture.setDefaultBufferSize(camera_width, camera_height);
        camera.open((CameraManager)getSystemService(Context.CAMERA_SERVICE), camera_id);
    }

    final View.OnClickListener record_click = new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (decoder.codec != null) {
                Toast.makeText(MainActivity.this, "stopped clip", 0).show();
                return;
            }

            String path;
            int id;

            id = prefs.getInt("clip_id", 0);
            path = String.format("/sdcard/clip%05d.mp4", id);

            SharedPreferences.Editor edit = prefs.edit();
            edit.putInt("clip_id", id + 1);
            edit.commit();

            writemux(path);
            Toast.makeText(MainActivity.this, "saved as " + path, 0).show();
        }
    };

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
            edit.putInt("clip_id", 0);
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

        button = new ImageButton(this);
        button.setOnClickListener(record_click);
        button.setImageResource(android.R.drawable.btn_radio);

        FrameLayout.LayoutParams params2 = new FrameLayout.LayoutParams(RelativeLayout.LayoutParams.WRAP_CONTENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        params2.gravity = Gravity.BOTTOM | Gravity.CENTER;
        addContentView(button, params2);

        /* the init function needs a surface for egl, get one from encoder
           *there is a probably a better way to get a temporary surface
        */
        encoder = new Encoder();

        encoder.open(640, 480);
        int texture_id = init(encoder.surface);
        encoder.close();

        surface_texture = new SurfaceTexture(texture_id);
        surface = new Surface(surface_texture);

        camera = new Camera(surface, surface_texture);
        decoder = new Decoder(surface);

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
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        // can only be result for clip open

        if (resultCode != -1) // -1 = success
            return;

        if (decoder.open(this, data.getData()))
            camera.close();
    }

    /* use the menu button to toggle preferences
        this assumes there will be a menu button... */
    boolean toggle = false;

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        FragmentManager fm = getFragmentManager();
        if (!toggle)
            fm.beginTransaction().replace(android.R.id.content, settings).commit();
        else
            fm.beginTransaction().remove(settings).commit();

        overlay.setVisibility(toggle ? View.VISIBLE : View.INVISIBLE);
        button.setVisibility(toggle ? View.VISIBLE : View.INVISIBLE);

        toggle = !toggle;
        return false;
    }
}
