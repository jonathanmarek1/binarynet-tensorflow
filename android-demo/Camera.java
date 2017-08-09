package test.app;

import android.util.Log;
import android.view.Surface;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.*;
import java.util.Arrays;

class Camera {
    static String TAG = "TESTAPP_Camera";

    int num_opened = 0;
    CameraDevice camera = null;

    Surface surface;
    SurfaceTexture texture;

    CaptureRequest build_request() throws CameraAccessException {
        CaptureRequest.Builder req;

        req = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
        req.addTarget(surface);
        req.set(CaptureRequest.CONTROL_AF_MODE,
                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
        return req.build();
    }

    final CameraCaptureSession.CaptureCallback capture_callback =
        new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureProgressed(CameraCaptureSession session,
                                        CaptureRequest request,
                                        CaptureResult result) {
        }

        @Override
        public void onCaptureCompleted(CameraCaptureSession session,
                                       CaptureRequest request,
                                       TotalCaptureResult result) {
            float mtx[] = new float[16];

            texture.updateTexImage();
            texture.getTransformMatrix(mtx);
            MainActivity.encode(mtx);
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
        }
    };

    final CameraDevice.StateCallback state_callback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice cam) {
            /* if multiple cameras opened, only keep the last one
                TODO: does this method always work?
            */
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

    Camera(Surface surface, SurfaceTexture texture) {
        this.surface = surface;
        this.texture = texture;
    }

    void open(CameraManager manager, String name) {
        close();
        try {
            num_opened++;
            manager.openCamera(name, state_callback, null);
        } catch (CameraAccessException e) {
            Log.e(TAG, "", e);
            return;
        }
    }

    void close() {
        if (camera != null) {
            num_opened--;
            camera.close();
            camera = null;
        }
    }
}
