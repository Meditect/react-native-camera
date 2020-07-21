#import "CustomModelManager.h"


@interface CustomModelManager ()
@property(nonatomic, assign) float scaleX;
@property(nonatomic, assign) float scaleY;
@end

@implementation CustomModelManager

- (instancetype)init
{
  if (self = [super init]) {

  }
  return self;
}

-(BOOL)isRealDetector
{
  return true;
}

- (void)runCustomModel:(UIImage *)uiImage scaleX:(float)scaleX scaleY:(float) scaleY completed: (void (^)(NSArray * result)) completed
{

    NSArray *ref_filenames = @[@"Voltarène"
      ,@"Vogalène_en"
      ,@"Tetanea"
      ,@"Primalan"
      ,@"Nifluril_fr"
      ,@"Nifluril_en"
      ,@"Mag2"
      ,@"Forlax_fr"
      ,@"Forlax_en"
      ,@"Flucazol"
      ,@"Ery"
      ,@"Efferalgan_suppositoires_600mg_fr"
      ,@"Efferalgan_suppositoires_300mg_fr"
      ,@"Efferalgan_suppositoires_150mg_fr"
      ,@"Efferalgan_suppositoires_80mg_fr"
      ,@"Efferalgan_poudreEffervescentePediatrique_en"
      ,@"Efferalgan_effervescent_500mg_fr"
      ,@"Efferalgan_effervescent_500mg_en"
      ,@"Efferalgan_codeine_30mg_fr"
      ,@"Doliprane"
      ,@"Co-Arinate_fr"
      ,@"Co-Arinate_en"
      ,@"ChibroCadron"
      ,@"Bimalaril"
      ,@"Balsolène"
      ,@"Augmentin"
      ,@"Aspirine_vitamineC_330mg_fr"
      ,@"Efferalgan_vitamineC_500mg_fr"
      ,@"Efferalgan_suppositoires_300mg_en"
      ,@"Efferalgan_suppositoires_80mg_en"
      ,@"Efferalgan_poudreEffervescentePediatrique_fr"
      ,@"Efferalgan_pediatrique_250mg_fr"
      ,@"Efferalgan_effervescent_1000mg_en"
      ,@"Efferalgan_comprimes_1000mg_fr"
      ,@"Efferalgan_comprimes_500mg_fr"
      ,@"Aspirine_1000mg_en"];

    NSString *modelPath = [NSBundle.mainBundle pathForResource:@"model_v3"
                                                    ofType:@"tflite"];
    FIRCustomLocalModel *localModel =
        [[FIRCustomLocalModel alloc] initWithModelPath:modelPath];

    FIRCustomRemoteModel *remoteModel =
        [[FIRCustomRemoteModel alloc] initWithName:@"Drug-Detector"];

    FIRModelInterpreter *interpreter;
    if ([[FIRModelManager modelManager] isModelDownloaded:remoteModel]) {
      printf("utilise distant");
      interpreter = [FIRModelInterpreter modelInterpreterForRemoteModel:remoteModel];
    } else {
      printf("utilise local");
      interpreter = [FIRModelInterpreter modelInterpreterForLocalModel:localModel];
    }

    FIRModelInputOutputOptions *ioOptions = [[FIRModelInputOutputOptions alloc] init];

    NSError *error;
    [ioOptions setInputFormatForIndex:0
                                type:FIRModelElementTypeFloat32
                          dimensions:@[@1, @256, @256, @3]
                                error:&error];

    [ioOptions setOutputFormatForIndex:0
                                  type:FIRModelElementTypeFloat32
                            dimensions:@[@1, @36]
                                error:&error];

    CGImageRef image = uiImage.CGImage;
    long imageWidth = CGImageGetWidth(image);
    long imageHeight = CGImageGetHeight(image);
    CGContextRef context = CGBitmapContextCreate(nil,
                                                imageWidth, imageHeight,
                                                8,
                                                imageWidth * 4,
                                                CGColorSpaceCreateDeviceRGB(),
                                                kCGImageAlphaNoneSkipFirst);
    CGContextDrawImage(context, CGRectMake(0, 0, imageWidth, imageHeight), image);
    UInt8 *imageData = CGBitmapContextGetData(context);

    FIRModelInputs *inputs = [[FIRModelInputs alloc] init];
    NSMutableData *inputData = [[NSMutableData alloc] initWithCapacity:0];

    for (int row = 0; row < 400; row++) {
      for (int col = 0; col < 400; col++) {
        long offset = 4 * (col * imageWidth + row);

        Float32 red = imageData[offset+1] / 255.0f;
        Float32 green = imageData[offset+2] / 255.0f;
        Float32 blue = imageData[offset+3] / 255.0f;

        [inputData appendBytes:&red length:sizeof(red)];
        [inputData appendBytes:&green length:sizeof(green)];
        [inputData appendBytes:&blue length:sizeof(blue)];
      }
    }

    [inputs addInput:inputData error:&error];

    [interpreter runWithInputs:inputs
                   options:ioOptions
                completion:^(FIRModelOutputs * _Nullable outputs,
                             NSError * _Nullable error) {
        NSError *outputError;
        NSArray *distances = [outputs outputAtIndex:0 error:&outputError][0];
        NSMutableArray *medicaments = [NSMutableArray new];

        float max = -10000;
        float max2 = -10000;
        float max3 = -10000;
        int index = 0;
        int index2 = 0;
        int index3 = 0;

        for (int i = 0; i < [distances count]; i++) {
          if(max < [distances[i] floatValue])
          {
            max = [distances[i] floatValue];
            index = i;
          }
        }

        for (int i = 0; i < [distances count]; i++) {
          if(max2 < [distances[i] floatValue] && [distances[i] floatValue] != max )
          {
            max2 = [distances[i] floatValue];
            index2 = i;
          }
        }

        for (int i = 0; i < [distances count]; i++) {
          if(max3 < [distances[i] floatValue] && [distances[i] floatValue] != max && [distances[i] floatValue] != max2 )
          {
            max3 = [distances[i] floatValue];
            index3 = i;
          }
        }

        medicaments[0] = (ref_filenames[index]);
        medicaments[1] = (ref_filenames[index2]);
        medicaments[2] = (ref_filenames[index3]);

        completed(medicaments);
    }];

}

@end

