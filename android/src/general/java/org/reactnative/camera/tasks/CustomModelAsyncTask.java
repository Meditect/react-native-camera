
package org.reactnative.camera.tasks;

import android.content.Context;
import android.content.ContextWrapper;
import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.uimanager.ThemedReactContext;
import com.facebook.react.bridge.ReactApplicationContext;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelOutputs;

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
        // On créer l'interpreter local ou distant
        mInterpreter = mCustomDetector.getDetector();

        // Récuperer les options
        mDataOptions = mCustomDetector.getOptions();

        // Convertir l'image reçu par la caméra en tableau à 4 dimensions puis en input
        final float[][][][] imgData = convertByteArrayToByteBuffer(mImageData, mWidth, mHeight);
        FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(imgData).build();

        mInterpreter
                .run(inputs, mDataOptions)
                .addOnSuccessListener(new OnSuccessListener<FirebaseModelOutputs>() {
                    @Override
                    public void onSuccess(FirebaseModelOutputs firebaseModelOutputs) {

                    // En sortie on a un array de dimensions [1][3][x] contenant les 3 médicaments les plus probables
                    float[][][] labelProbArray = firebaseModelOutputs.<float[][][]>getOutput(0);
                    // if boxOrNot[0][0] > 0 then it is a box
                    float[][] boxOrNot = firebaseModelOutputs.<float[][]>getOutput(1);

                    WritableArray result = Arguments.createArray();
                    String nomMedicament = "";

                    //La première valeure de chaque image contient la probabilité > 0 si le médicament est reconnu 0 sinon
                    result.pushDouble(labelProbArray[0][0][0]);
                    boolean medicamentProbaNotnull = false;

                    for (int p = 0; p < 2; p++) {
                    // Les valeurs suivantes représentent la chaine de caractère du médicament reconnu
                    // Cette chaine est encodé en Unicode il faut ensuite la décoder
                      if (labelProbArray[0][p][0] > 0) {
                        medicamentProbaNotnull = true;
                        for (int i = 1; i < labelProbArray[0][p].length; i++) {
                          if (labelProbArray[0][p][i] != 0) {
                            nomMedicament += Character.toString((char) labelProbArray[0][0][i]);
                          }
                        }
                        result.pushString(nomMedicament);
                        break;
                      }
                    }

                    // Si la probabilité est superieur à 0 le médicament est reconnu donc on le renvoie à react-native
                    if ((boxOrNot[0][0] > 0 && medicamentProbaNotnull)) {
                      mDelegate.onCustomModel(result);
                    }

                    // if (debugML) {
                    //   mDelegate.onCustoModelPrediction(all_results);
                    // }

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

    //Premiere méthode pour convertir mImgData en Bitmap

    // ByteArrayOutputStream out = new ByteArrayOutputStream();

    // YuvImage yuvImage = new YuvImage(mImgData, ImageFormat.NV21, mWidth, mHeight, null);
    // yuvImage.compressToJpeg(new Rect(0, 0, mWidth, mHeight), 100, out);
    // byte[] imageBytes = out.toByteArray();

    // Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

    //Deuxieme méthode pour convertir mImgData en Bitmap

    RenderScript rs = RenderScript.create(mThemedReactContext);
    ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

    Allocation in = Allocation.createSized(rs, Element.U8(rs), mImgData.length);

    Bitmap bitmap = Bitmap.createBitmap(mWidth, mHeight, Bitmap.Config.ARGB_8888);

    Allocation out = Allocation.createFromBitmap(rs,bitmap);

    yuvToRgbIntrinsic.setInput(in);

    in.copyFrom(mImgData);

    yuvToRgbIntrinsic.forEach(out);

    out.copyTo(bitmap);

    //Une fois la bitmap obtenu on la rogne pour obtenir un carré car sinon l'image serait trop déformée

    Bitmap squareBitmap;

    if (bitmap.getWidth() >= bitmap.getHeight()) {

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

    // Fonction pour obtenir une Bitmap depuis une image stockée dans Assets

    // AssetManager assetManager = mThemedReactContext.getAssets();
    // InputStream istr = null;
    // try {
    //     istr = assetManager.open("imagetest.jpeg");
    // } catch (IOException e) {
    //     e.printStackTrace();
    // }
    // Bitmap bitmapTest = BitmapFactory.decodeStream(istr);

    // On créer une bitmap à l'echelle du modèle [256][256]

    Bitmap scaledBitmap = Bitmap.createScaledBitmap(squareBitmap, 256, 256, true);

    // On créer un array à 4 dimensions pour accueillir la bitmap
    float[][][][] input = new float[1][256][256][3];

    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        int pixel = scaledBitmap.getPixel(x, y);
        input[0][y][x][0] = ( Color.red(pixel) / 255.0f );
        input[0][y][x][1] = ( Color.green(pixel) / 255.0f );
        input[0][y][x][2] = ( Color.blue(pixel) / 255.0f );
      }
    }

    return input;
  }

}
