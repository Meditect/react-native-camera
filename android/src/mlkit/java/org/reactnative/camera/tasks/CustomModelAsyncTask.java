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
  private static final int DIM_IMG_SIZE_X = 400; //368;
  private static final int DIM_IMG_SIZE_Y = 400;//368;

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

    //mLabelList = loadLabelList(mThemedReactContext);

    try {

      mDataOptions = new FirebaseModelInputOutputOptions.Builder()
              .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, 3})
              .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 7})
              .build();

      FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder().setAssetFilePath("mnist_model.tflite").build();

      FirebaseModelInterpreterOptions modelOptions = new FirebaseModelInterpreterOptions.Builder(localModel).build();

      mInterpreter = FirebaseModelInterpreter.getInstance(modelOptions);

      // custom model

      float[][][][] imgData = convertByteArrayToByteBuffer(mImageData, mWidth, mHeight);

      FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(imgData).build();
      // Here's where the magic happens!!
      mInterpreter
              .run(inputs, mDataOptions)
              .addOnSuccessListener(new OnSuccessListener<FirebaseModelOutputs>() {
                @Override
                public void onSuccess(FirebaseModelOutputs firebaseModelOutputs) {
                  Log.e("","ok");
                  float[][] labelProbArray = firebaseModelOutputs.<float[][]>getOutput(0);

                  for (int i = 0; i < labelProbArray[0].length; i++) {
                    System.out.println(labelProbArray[0][i]);
                  }

                  // WritableArray topLabels = getTopLabels(labelProbArray);
                  // WritableArray arrayTest = Arguments.createArray();
                  // System.out.println(topLabels);
                  // mDelegate.onCustomModel(topLabels);
                   mDelegate.onCustomModelTaskCompleted();
                }
              })
              .addOnFailureListener(
                      new OnFailureListener() {
                        @Override
                        public void onFailure(Exception e) {
                          Log.e(TAG, "Custom model task failed" + e);
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

  private synchronized float[][][][] convertByteArrayToByteBuffer(byte[] mImgData, int mWidth, int mHeight) throws FileNotFoundException {

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    YuvImage yuvImage = new YuvImage(mImgData, ImageFormat.NV21, mWidth, mHeight, null);
    yuvImage.compressToJpeg(new Rect(0, 0, mWidth, mHeight), 100, out);
    byte[] imageBytes = out.toByteArray();
    Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    Bitmap scaledBitmap2 = Bitmap.createScaledBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, true);

    float[][][][] input = new float[1][DIM_IMG_SIZE_X][DIM_IMG_SIZE_Y][3];
    float min = 1;
    float max = 0;
    int max2 = 0;
    for (int x = 0; x < DIM_IMG_SIZE_X; x++) {
      for (int y = 0; y < DIM_IMG_SIZE_Y; y++) {
        int pixel = scaledBitmap2.getPixel(x, y);
        input[0][x][y][0] = (Color.red(pixel) );
        input[0][x][y][1] = (Color.green(pixel) );
        input[0][x][y][2] = (Color.blue(pixel) );
      }
    }
    return input;
  }
}
