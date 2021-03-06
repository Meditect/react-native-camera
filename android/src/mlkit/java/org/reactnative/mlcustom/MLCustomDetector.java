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

    //Definir si le modèle distant est téléchargé ou non
    public void setMode(Boolean mode) {
        mMode = mode;
    }

    //Definir le nom du modèle distant
    public void setName(String name) {
        mName = name;
    }

    //Definir les dimensions du modèle distant
    public void setDimensions(List<Integer> dimensions) {

        int[] dimensionsArray = new int[dimensions.size()];
        for(int i = 0;i < dimensionsArray.length;i++) {
            dimensionsArray[i] = dimensions.get(i);
        }
        mDimensions = dimensionsArray;
    }

    private void createCustomDetector() throws FirebaseMLException {

        // on vérifie si le modèle distant est téléchargé et a un nom et une dimensions de définit
        if (mMode && mName != "" && mDimensions.length > 0) {
            FirebaseCustomRemoteModel remoteModel = new FirebaseCustomRemoteModel.Builder(mName).build();
            FirebaseModelInterpreterOptions options = new FirebaseModelInterpreterOptions.Builder(remoteModel).build();
            mCustomDetector = FirebaseModelInterpreter.getInstance(options);
        // Sinon on utilise le modèle local
        } else {
            System.out.println("CUSTOM LOCAL");
            FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder().setAssetFilePath("model_v8.tflite").build();
            FirebaseModelInterpreterOptions options = new FirebaseModelInterpreterOptions.Builder(localModel).build();
            mCustomDetector = FirebaseModelInterpreter.getInstance(options);
        }
    }

    private void createCustomOptions() throws FirebaseMLException {
        // on vérifie si le modèle distant est téléchargé et a un nom et une dimensions de définit
        if (mMode && mName != "" && mDimensions.length > 0) {
            mCustomOptions = new FirebaseModelInputOutputOptions.Builder()
              .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 256, 256, 3})
              .setOutputFormat(0, FirebaseModelDataType.FLOAT32, mDimensions)
              .setOutputFormat(1, FirebaseModelDataType.FLOAT32, new int[]{1, 1})
              .build();
        // Sinon on définit les options pour le modèle local
        } else {
            mCustomOptions = new FirebaseModelInputOutputOptions.Builder()
              .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 256, 256, 3})
              .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 3, 73})
              .setOutputFormat(1, FirebaseModelDataType.FLOAT32, new int[]{1, 1})
              .build();
        }
    }
}
