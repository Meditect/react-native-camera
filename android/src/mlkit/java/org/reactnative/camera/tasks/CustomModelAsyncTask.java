
package org.reactnative.camera.tasks;

import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.util.Log;
import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.graphics.Color;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.uimanager.ThemedReactContext;
import com.facebook.react.bridge.ReactApplicationContext;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;

import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;

import org.reactnative.camera.utils.ImageDimensions;
import org.reactnative.mlcustom.MLCustomDetector;

public class CustomModelAsyncTask extends android.os.AsyncTask<Void, Void, Void> {

  private CustomModelAsyncTaskDelegate mDelegate;
  private ThemedReactContext mThemedReactContext;
  private MLCustomDetector mCustomDetector;
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

  private static ReactApplicationContext reactContext;

  private FirebaseModelInterpreter mInterpreter;

  private FirebaseModelInputOutputOptions mDataOptions;

  public CustomModelAsyncTask(
          CustomModelAsyncTaskDelegate delegate,
          ThemedReactContext themedReactContext,
          MLCustomDetector customDetector,
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
    mCustomDetector = customDetector;
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

    if (isCancelled() || mDelegate == null) {
      return null;
    }

    try {

        mInterpreter = mCustomDetector.getDetector();

        mDataOptions = mCustomDetector.getOptions();

        final float[][][][] imgData = convertByteArrayToByteBuffer(mImageData, mWidth, mHeight);

        FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(imgData).build();

        mInterpreter
                .run(inputs, mDataOptions)
                .addOnSuccessListener(new OnSuccessListener<FirebaseModelOutputs>() {
                    @Override
                    public void onSuccess(FirebaseModelOutputs firebaseModelOutputs) {

                    float[][][] labelProbArray = firebaseModelOutputs.<float[][][]>getOutput(0);

                    WritableArray result = Arguments.createArray();
                    String nomMedicament = "";

                    result.pushString(String.valueOf(labelProbArray[0][0][0]));

                    for (int i = 1; i < labelProbArray[0][0].length; i++) {
                      if (labelProbArray[0][0][i] != 0) {
                        nomMedicament += Character.toString((char) labelProbArray[0][0][i]);
                      }
                    } 

                    result.pushString(nomMedicament);

                    mDelegate.onCustomModel(result);
                    mDelegate.onCustomModelTaskCompleted();
                    }
                })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(Exception e) {
                            e.printStackTrace();
                            mDelegate.onCustomModelTaskCompleted();
                            }
                        });

    } catch (FirebaseMLException | FileNotFoundException e) {
      e.printStackTrace();
    }
    return null;
  }

  private synchronized float[][][][] convertByteArrayToByteBuffer(byte[] mImgData, int mWidth, int mHeight) throws FileNotFoundException {

    //Convert the YUV byte array into a bitmap

    Bitmap squareBitmap;

    ByteArrayOutputStream out = new ByteArrayOutputStream();

    YuvImage yuvImage = new YuvImage(mImgData, ImageFormat.NV21, mWidth, mHeight, null);
    yuvImage.compressToJpeg(new Rect(0, 0, mWidth, mHeight), 100, out);
    byte[] imageBytes = out.toByteArray();

    Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

    if (bitmap.getWidth() >= bitmap.getHeight()){

      squareBitmap = Bitmap.createBitmap(
         bitmap, 
         bitmap.getWidth()/2 - bitmap.getHeight()/2,
         0,
         bitmap.getHeight(), 
         bitmap.getHeight()
         );
    
    }else{
    
      squareBitmap = Bitmap.createBitmap(
         bitmap,
         0, 
         bitmap.getHeight()/2 - bitmap.getWidth()/2,
         bitmap.getWidth(),
         bitmap.getWidth() 
         );
    }

    // RenderScript rs = RenderScript.create(mThemedReactContext);
    // ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

    // Allocation in = Allocation.createSized(rs, Element.U8(rs), mImgData.length);

    // Bitmap bitmap = Bitmap.createBitmap(mWidth, mHeight, Bitmap.Config.ARGB_8888);

    // Allocation out = Allocation.createFromBitmap(rs,bitmap);

    // yuvToRgbIntrinsic.setInput(in);

    // in.copyFrom(mImgData);

    // yuvToRgbIntrinsic.forEach(out);

    // out.copyTo(bitmap);

    Bitmap scaledBitmap2 = Bitmap.createScaledBitmap(squareBitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, true);

    // AssetManager assetManager = mThemedReactContext.getAssets();
    // InputStream istr = null;
    // try {
    //     istr = assetManager.open("imagetestv2.jpg");
    // } catch (IOException e) {
    //     e.printStackTrace();
    // }
    // Bitmap bitmap2 = BitmapFactory.decodeStream(istr);
    // Bitmap scaledBitmap3 = bitmap2.createScaledBitmap(bitmap2, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, true);

    // Create a 4 dimension array with the reduced Bitmap informations
    float[][][][] input = new float[1][DIM_IMG_SIZE_X][DIM_IMG_SIZE_Y][3];

    for (int y = 0; y < DIM_IMG_SIZE_Y; y++) {
      for (int x = 0; x < DIM_IMG_SIZE_X; x++) {
        int pixel = scaledBitmap2.getPixel(x, y);
        input[0][y][x][0] = ( Color.red(pixel) / 255.0f );
        input[0][y][x][1] = ( Color.green(pixel) / 255.0f );
        input[0][y][x][2] = ( Color.blue(pixel) / 255.0f );
      }
    }

    return input;
  }

}