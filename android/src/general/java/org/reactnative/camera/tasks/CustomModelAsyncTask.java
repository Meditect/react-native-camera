package org.reactnative.camera.tasks;

import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.util.Log;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.uimanager.ThemedReactContext;

import com.google.android.cameraview.CameraView;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;

//Custom model imports

import android.widget.Toast;

import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Callback;
import com.google.android.gms.tasks.Continuation;
import com.google.android.gms.tasks.Task;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import android.app.Activity;
import android.graphics.BitmapFactory;
import android.os.SystemClock;
import androidx.annotation.NonNull;
import android.widget.Toast;

import android.graphics.Bitmap;
import android.graphics.Color;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.AbstractMap;
import java.util.ArrayList;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;

import org.reactnative.camera.utils.ImageDimensions;

import java.util.List;


public class CustomModelAsyncTask extends android.os.AsyncTask<Void, Void, Void> {

    private CustomModelAsyncTaskDelegate mDelegate;
    private ThemedReactContext mThemedReactContext;
    private byte[] mImageData;
    private int mWidth;
    private int mHeight;
    private int mRotation;
    private double mScaleX;
    private double mScaleY;
    private ImageDimensions mImageDimensions;
    private int mPaddingLeft;
    private int mPaddingTop;
    private String TAG = "RNCamera";

    //Custom Model variables

    private static final int RESULTS_TO_SHOW = 3;

    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int DIM_IMG_SIZE_X = 224; //368;
    private static final int DIM_IMG_SIZE_Y = 224;//368;

    private List<String> mLabelList;

    private final PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1,
                                           Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    private static ReactApplicationContext reactContext;

    private FirebaseModelInterpreter mInterpreter;

    private FirebaseModelInputOutputOptions mDataOptions;

    public CustomModelAsyncTask(
            CustomModelAsyncTaskDelegate delegate,
            ThemedReactContext themedReactContext,
            byte[] imageData,
            int width,
            int height,
            int rotation,
            float density,
            int facing,
            int viewWidth,
            int viewHeight,
            int viewPaddingLeft,
            int viewPaddingTop
    ) {
        mThemedReactContext = themedReactContext;
        mDelegate = delegate;
        mImageData = imageData;
        mWidth = width;
        mHeight = height;
        mRotation = rotation;
        mImageDimensions = new ImageDimensions(width, height, rotation, facing);
        mScaleX = (double) (viewWidth) / (mImageDimensions.getWidth() * density);
        mScaleY = (double) (viewHeight) / (mImageDimensions.getHeight() * density);
        mPaddingLeft = viewPaddingLeft;
        mPaddingTop = viewPaddingTop;
    }

    @Override
    protected Void doInBackground(Void... ignored) {

        System.out.println("rentré dans background");
        if (isCancelled() || mDelegate == null) {
            return null;
        }

        mLabelList = loadLabelList(mThemedReactContext);

        try {

            mDataOptions = new FirebaseModelInputOutputOptions.Builder()
                    .setInputFormat(0, FirebaseModelDataType.BYTE, new int[]{1, 224, 224, 3})
                    .setOutputFormat(0, FirebaseModelDataType.BYTE, new int[]{1, mLabelList.size()})
                    .build();

            FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder().setAssetFilePath("mobilenet_v1_1.0_224_quant.tflite").build();

            FirebaseModelInterpreterOptions modelOptions = new FirebaseModelInterpreterOptions.Builder(localModel).build();

            mInterpreter = FirebaseModelInterpreter.getInstance(modelOptions);

            //showToast("Model loaded");
            // custom model

            byte[] imgDataNV21 = YV12toNV21(mImageData, null, mWidth, mHeight);

            ByteBuffer imgData = convertByteArrayToByteBuffer(imgDataNV21, mWidth, mHeight);

            FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(imgData).build();
            // Here's where the magic happens!!
            mInterpreter
                    .run(inputs, mDataOptions)
                    .addOnSuccessListener(new OnSuccessListener<FirebaseModelOutputs>() {
                        @Override
                        public void onSuccess(FirebaseModelOutputs firebaseModelOutputs) {
                            Log.e("","ok");
                            byte[][] labelProbArray = firebaseModelOutputs.<byte[][]>getOutput(0);
                            WritableArray topLabels = getTopLabels(labelProbArray);
                            //WritableArray arrayTest = Arguments.createArray();
                            System.out.println(topLabels);
                            mDelegate.onCustomModel(topLabels);
                            mDelegate.onCustomModelTaskCompleted();
                        }
                    })
                    .addOnFailureListener(
                            new OnFailureListener() {
                                @Override
                                public void onFailure(Exception e) {
                                    Log.e(TAG, "Text recognition task failed" + e);
                                    mDelegate.onCustomModelTaskCompleted();
                                }
                            });

        } catch (FirebaseMLException | FileNotFoundException e) {
            //Toast.makeText(getReactApplicationContext(), "model load failed", 4).show();
            e.printStackTrace();
        }
        return null;
    }

    private synchronized WritableArray getTopLabels(byte[][] labelProbArray) {
        for (int i = 0; i < mLabelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(mLabelList.get(i), (labelProbArray[0][i] & 0xff) / 255.0f));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        WritableArray result = Arguments.createArray();
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            result.pushString(label.getKey() + ":" + label.getValue());
        }
        //Log.d("labels: " + result.toString());
        return result;
    }

    private List<String> loadLabelList(ThemedReactContext context) {
        List<String> labelList = new ArrayList<>();
        try (
                BufferedReader reader =
                        new BufferedReader(new InputStreamReader(context.getAssets().open("mobilenet.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labelList.add(line);
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to read label list.", e);
        }
        return labelList;
    }

    private synchronized ByteBuffer convertByteArrayToByteBuffer(byte[] mImgData, int mWidth, int mHeight) throws FileNotFoundException {

        int bytesPerChannel = 1;

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        YuvImage yuvImage = new YuvImage(mImgData, ImageFormat.NV21, mWidth, mHeight, null);
        yuvImage.compressToJpeg(new Rect(0, 0, mWidth, mHeight), 100, out);
        byte[] imageBytes = out.toByteArray();
        Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

        // Bitmap bitmap = Bitmap.createBitmap(mHeight, mWidth, Bitmap.Config.ALPHA_8);
        // ByteBuffer buffer = ByteBuffer.wrap(mImgData);
        // buffer.rewind();
        // bitmap.copyPixelsFromBuffer(buffer);

        //        Matrix matrixR = new Matrix();
        //        matrixR.postRotate(90.0f);
        //        Bitmap bitmapRaw = Bitmap.createBitmap(bitmap, 0, 0,  bitmap.getWidth(),  bitmap.getHeight(), matrixR, true);

        ByteBuffer imgData = ByteBuffer.allocateDirect(bytesPerChannel * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, true);
        imgData.rewind();

        /* Preallocated buffers for storing image data. */
        int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());
        // Convert the image to int points.
        long startTime = SystemClock.uptimeMillis();

        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];

                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));

            }
        }
        long endTime = SystemClock.uptimeMillis();
        return imgData;
    }

    private byte[] YV12toNV21(final byte[] input, byte[] output, final int width, final int height) {
        if (output == null) {
            output = new byte[input.length];
        }
        final int size = width * height;
        final int quarter = size / 4;
        final int u0 = size + quarter;

        System.arraycopy(input, 0, output, 0, size); // Y is same

        for (int v = size, u = u0, o = size; v < u0; u++, v++, o += 2) {
            output[o] = input[v]; // For NV21, V first
            output[o + 1] = input[u]; // For NV21, U second
        }
        return output;
    }

    // private void showToast(String message) {
    //   Toast.makeText(getReactApplicationContext(), message, Toast.LENGTH_SHORT).show();
    // }
}
