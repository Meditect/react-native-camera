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

    NSArray *ref_filenames = @[@"Voltarène/Ref.jpg", @"Flucazol/Ref.jpg", @"Primalan/Ref.jpg", @"Ery/Ref.jpg", @"Doliprane/Ref.jpg", @"Mag2/Ref.jpg", @"Co-Arinate_fr/Ref.jpg", @"Co-Arinate_en/Ref.jpg", @"Tetanea/Ref.jpg", @"Nifluril_fr/Ref.jpg", @"Efferalgan_effervescent_500mg_en/Ref.jpg", @"Efferalgan_effervescent_500mg_fr/Ref.jpg", @"Efferalgan_poudreEffervescentePediatrique_en/Ref.jpg", @"Efferalgan_codeine_30mg_fr/Ref.jpg", @"Efferalgan_suppositoires_150mg_fr/Ref.jpg", @"Efferalgan_suppositoires_80mg_fr/Ref.jpg", @"Efferalgan_suppositoires_300mg_fr/Ref.jpg", @"Aspirine_vitamineC_330mg_fr/Ref.jpg", @"Efferalgan_suppositoires_600mg_fr/Ref.jpg", @"Forlax_fr/Ref.jpg", @"Forlax_en/Ref.jpg", @"Vogalène_en/Ref.jpg", @"Efferalgan_effervescent_1000mg_en/Ref.jpg", @"Efferalgan_comprimes_500mg_fr/Ref.jpg", @"Efferalgan_comprimes_1000mg_fr/Ref.jpg", @"Efferalgan_suppositoires_80mg_en/Ref.jpg", @"Aspirine_1000mg_en/Ref.jpg", @"Efferalgan_poudreEffervescentePediatrique_fr/Ref.jpg", @"Efferalgan_pediatrique_250mg_fr/Ref.jpg", @"Efferalgan_suppositoires_300mg_en/Ref.jpg", @"Efferalgan_vitamineC_500mg_fr/Ref.jpg", @"Nifluril_en/Ref.jpg"];

    NSString *modelPath = [NSBundle.mainBundle pathForResource:@"model_v2_quant"
                                                    ofType:@"tflite"];
    FIRCustomLocalModel *localModel =
        [[FIRCustomLocalModel alloc] initWithModelPath:modelPath];

    FIRModelInterpreter *interpreter =
        [FIRModelInterpreter modelInterpreterForLocalModel:localModel];

    FIRModelInputOutputOptions *ioOptions = [[FIRModelInputOutputOptions alloc] init];

    NSError *error;
    [ioOptions setInputFormatForIndex:0
                                type:FIRModelElementTypeFloat32
                          dimensions:@[@1, @400, @400, @3]
                                error:&error];

    [ioOptions setOutputFormatForIndex:0
                                  type:FIRModelElementTypeFloat32
                            dimensions:@[@1, @32]
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

