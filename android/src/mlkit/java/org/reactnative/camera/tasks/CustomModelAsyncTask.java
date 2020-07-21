package org.reactnative.camera.tasks;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
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

  // Dimensions required by the custom model
  private static final int DIM_BATCH_SIZE = 1;
  private static final int DIM_PIXEL_SIZE = 3;
  private static final int DIM_IMG_SIZE_X = 256; //368;
  private static final int DIM_IMG_SIZE_Y = 256;//368;

  private static ReactApplicationContext reactContext;

  private FirebaseModelInterpreter mInterpreter;

  private FirebaseModelInputOutputOptions mDataOptions;

  private String[] ref_filenames = new String[]{"Voltarène"
    ,"Vogalène_en"
    ,"Tetanea"
    ,"Primalan"
    ,"Nifluril_fr"
    ,"Nifluril_en"
    ,"Mag2"
    ,"Forlax_fr"
    ,"Forlax_en"
    ,"Flucazol"
    ,"Ery"
    ,"Efferalgan_suppositoires_600mg_fr"
    ,"Efferalgan_suppositoires_300mg_fr"
    ,"Efferalgan_suppositoires_150mg_fr"
    ,"Efferalgan_suppositoires_80mg_fr"
    ,"Efferalgan_poudreEffervescentePediatrique_en"
    ,"Efferalgan_effervescent_500mg_fr"
    ,"Efferalgan_effervescent_500mg_en"
    ,"Efferalgan_codeine_30mg_fr"
    ,"Doliprane"
    ,"Co-Arinate_fr"
    ,"Co-Arinate_en"
    ,"ChibroCadron"
    ,"Bimalaril"
    ,"Balsolène"
    ,"Augmentin"
    ,"Aspirine_vitamineC_330mg_fr"
    ,"Efferalgan_vitamineC_500mg_fr"
    ,"Efferalgan_suppositoires_300mg_en"
    ,"Efferalgan_suppositoires_80mg_en"
    ,"Efferalgan_poudreEffervescentePediatrique_fr"
    ,"Efferalgan_pediatrique_250mg_fr"
    ,"Efferalgan_effervescent_1000mg_en"
    ,"Efferalgan_comprimes_1000mg_fr"
    ,"Efferalgan_comprimes_500mg_fr"
    ,"Aspirine_1000mg_en"};

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

    if (isCancelled() || mDelegate == null) {
      return null;
    }

    try {

      mDataOptions = new FirebaseModelInputOutputOptions.Builder()
              .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, 3})
              .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 36})
              .build();

      // initialization of local and remote models

      final FirebaseCustomRemoteModel remoteModel = new FirebaseCustomRemoteModel.Builder("Drug-Detector").build();

      final FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder().setAssetFilePath("model_v3.tflite").build();

      FirebaseModelManager.getInstance().isModelDownloaded(remoteModel)
              .addOnSuccessListener(new OnSuccessListener<Boolean>() {
                @Override
                public void onSuccess(Boolean isDownloaded) {
                  FirebaseModelInterpreterOptions options;

                  // Create the interpreter for the local or remote model if downloaded

                  if (isDownloaded) {
                    System.out.println("utilise distant");
                    options = new FirebaseModelInterpreterOptions.Builder(remoteModel).build();
                  } else {
                    System.out.println("utilise local");
                    options = new FirebaseModelInterpreterOptions.Builder(localModel).build();
                  }
                  try {
                    mInterpreter = FirebaseModelInterpreter.getInstance(options);
                    float[][][][] imgData = convertByteArrayToByteBuffer(mImageData, mWidth, mHeight);

                    FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(imgData).build();

                    mInterpreter
                            .run(inputs, mDataOptions)
                            .addOnSuccessListener(new OnSuccessListener<FirebaseModelOutputs>() {
                              @Override
                              public void onSuccess(FirebaseModelOutputs firebaseModelOutputs) {

                                float[][] labelProbArray = firebaseModelOutputs.<float[][]>getOutput(0);

                                WritableArray result = Arguments.createArray();
                                WritableArray medicaments = Arguments.createArray();

                                float max = -10000;
                                float max2 = -10000;
                                float max3 = -10000;
                                int index = 0;
                                int index2 = 0;
                                int index3 = 0;

                                for (int i = 0; i < labelProbArray[0].length; i++) {
                                  result.pushString(String.valueOf(labelProbArray[0][i]));
                                  if(max < labelProbArray[0][i])
                                  {
                                    max = labelProbArray[0][i];
                                    index = i;
                                  }
                                }

                                for (int i = 0; i < labelProbArray[0].length; i++) {
                                  if(max2 < labelProbArray[0][i] && labelProbArray[0][i] != max )
                                  {
                                    max2 = labelProbArray[0][i];
                                    index2 = i;
                                  }
                                }

                                for (int i = 0; i < labelProbArray[0].length; i++) {
                                  if(max3 < labelProbArray[0][i] && labelProbArray[0][i] != max && labelProbArray[0][i] != max2 )
                                  {
                                    max3 = labelProbArray[0][i];
                                    index3 = i;
                                  }
                                }

                                medicaments.pushString(ref_filenames[index]);
                                medicaments.pushString(ref_filenames[index2]);
                                medicaments.pushString(ref_filenames[index3]);

                                mDelegate.onCustomModel(medicaments);
                                mDelegate.onCustomModelTaskCompleted();
                              }
                            })
                            .addOnFailureListener(
                                    new OnFailureListener() {
                                      @Override
                                      public void onFailure(Exception e) {
                                        mDelegate.onCustomModelTaskCompleted();
                                      }
                                    });
                  } catch (FirebaseMLException | FileNotFoundException e) {
                    e.printStackTrace();
                  }

                }
              });

    } catch (FirebaseMLException e) {
      e.printStackTrace();
    }
    return null;
  }

  private synchronized float[][][][] convertByteArrayToByteBuffer(byte[] mImgData, int mWidth, int mHeight) throws FileNotFoundException {

    // Convert the YUV byte array into a bitmap
    ByteArrayOutputStream out = new ByteArrayOutputStream();

    YuvImage yuvImage = new YuvImage(mImgData, ImageFormat.NV21, mWidth, mHeight, null);
    yuvImage.compressToJpeg(new Rect(0, 0, mWidth, mHeight), 100, out);
    byte[] imageBytes = out.toByteArray();

    Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    Bitmap scaledBitmap2 = Bitmap.createScaledBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, true);

    // Create a 4 dimension array with the reduced Bitmap informations
    float[][][][] input = new float[1][DIM_IMG_SIZE_X][DIM_IMG_SIZE_Y][3];

    for (int x = 0; x < DIM_IMG_SIZE_X; x++) {
      for (int y = 0; y < DIM_IMG_SIZE_Y; y++) {
        int pixel = scaledBitmap2.getPixel(x, y);
        input[0][x][y][0] = ( Color.red(pixel) / 255.0f );
        input[0][x][y][1] = ( Color.green(pixel) / 255.0f );
        input[0][x][y][2] = ( Color.blue(pixel) / 255.0f );
      }
    }
    return input;
  }

}
