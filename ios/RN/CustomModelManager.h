#if __has_include(<FirebaseMLVision/FirebaseMLVision.h>)
  #import <FirebaseMLCommon/FirebaseMLCommon.h>
  #import <FirebaseMLVision/FirebaseMLVision.h>
  #import <FirebaseMLModelInterpreter/FirebaseMLModelInterpreter.h>

#endif
  @interface CustomModelManager : NSObject
  typedef void(^postRecognitionBlock)(NSArray *customs);

  - (instancetype)init;

  -(BOOL)isRealDetector;

  -(void)runCustomModel:(UIImage *)image scaleX:(float)scaleX scaleY:(float)scaleY completed:(postRecognitionBlock)completed;

  @end
