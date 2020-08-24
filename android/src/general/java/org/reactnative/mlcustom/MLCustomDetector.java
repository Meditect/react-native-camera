package org.reactnative.mlcustom;

import android.content.Context;

import java.util.List;

import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;

public class MLCustomDetector {

    private FirebaseModelInterpreter mCustomDetector = null;
    private FirebaseModelInputOutputOptions mCustomOptions;
    private boolean mMode = false;
    private String mName = "";
    private int[] mDimensions;

    public MLCustomDetector(Context context) {

    }

    public FirebaseModelInterpreter getDetector() throws FirebaseMLException {

        createCustomDetector();
        return mCustomDetector;
    }

    public FirebaseModelInputOutputOptions getOptions() throws FirebaseMLException {

        createCustomOptions();
        return mCustomOptions;
    }

    public void setMode(Boolean mode) {
        mMode = mode;
    }

    public void setName(String name) {
        mName = name;
    }

    public void setDimensions(List<Integer> dimensions) {
        
        int[] dimensionsArray = new int[dimensions.size()];
        for(int i = 0;i < dimensionsArray.length;i++) {
            dimensionsArray[i] = dimensions.get(i);
        }
        mDimensions = dimensionsArray;
    }

    private void createCustomDetector() throws FirebaseMLException {
        if (mMode && mName != "" && mDimensions.length > 0) {
            FirebaseCustomRemoteModel remoteModel = new FirebaseCustomRemoteModel.Builder(mName).build();
            FirebaseModelInterpreterOptions options = new FirebaseModelInterpreterOptions.Builder(remoteModel).build();
            mCustomDetector = FirebaseModelInterpreter.getInstance(options);
        } else {
            FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder().setAssetFilePath("model_v8.tflite").build();
            FirebaseModelInterpreterOptions options = new FirebaseModelInterpreterOptions.Builder(localModel).build();
            mCustomDetector = FirebaseModelInterpreter.getInstance(options);
        }
    }

    private void createCustomOptions() throws FirebaseMLException {
        if (mMode && mName != "" && mDimensions.length > 0) {
            mCustomOptions = new FirebaseModelInputOutputOptions.Builder()
              .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 256, 256, 3})
              .setOutputFormat(0, FirebaseModelDataType.FLOAT32, mDimensions)
              .build();
        } else {
            mCustomOptions = new FirebaseModelInputOutputOptions.Builder()
              .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 256, 256, 3})
              .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 3, 73})
              .build();
        }
    }
}