package org.reactnative.camera.tasks;

import com.facebook.react.bridge.WritableArray;

public interface CustomModelAsyncTaskDelegate {
  void onCustomModel(WritableArray serializedData);
  void onCustomModelTaskCompleted();
}
