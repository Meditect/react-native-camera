#import "CustomModelManager.h"
#import <React/RCTConvert.h>

@interface CustomModelManager ()
@property(nonatomic, assign) float scaleX;
@property(nonatomic, assign) float scaleY;
@property(nonatomic, assign) BOOL mMode;
@property(nonatomic, assign) NSString *mName;
@property(nonatomic, assign) NSMutableArray *mDimensions;
@end

@implementation CustomModelManager

- (instancetype)init
{
  if (self = [super init]) {
    self.mMode = false;
    self.mName = [[NSString alloc]initWithString:@"foo"];  
    self.mDimensions = [NSMutableArray new];
  }
  return self;
}

-(BOOL)isRealDetector
{
  return true;
}

// Permet de définir si le modèle distant est téléchargé (fonctionne)
- (void)setMode:(id)json queue:(dispatch_queue_t)sessionQueue 
{
  BOOL requestedValue = [RCTConvert BOOL:json];
  self.mMode = requestedValue;
}

// Permet de définir le nom du modèle distant (fonctionne pas)
- (void)setName:(id)json queue:(dispatch_queue_t)sessionQueue 
{
    NSMutableString *requestedValue = [RCTConvert NSString:json];
    self.mName = requestedValue;
}

// Permet de définir les dimensions du modèle distant (fonctionne pas)
- (void)setDimensions:(id)json queue:(dispatch_queue_t)sessionQueue 
{
    NSMutableArray *requestedValue = [RCTConvert NSArray:json];
    self.mDimensions = requestedValue;
}

- (void)runCustomModel:(UIImage *)uiImage scaleX:(float)scaleX scaleY:(float) scaleY completed: (void (^)(NSArray * result)) completed
{

    // Definition du modèle local
    NSString *modelPath = [NSBundle.mainBundle pathForResource:@"mnist_model_v13_quant"
                                                    ofType:@"tflite"];
    FIRCustomLocalModel *localModel =
        [[FIRCustomLocalModel alloc] initWithModelPath:modelPath];

    // Defintion du modèle distant (Remplacer Drug-Detector par le props mName)
    FIRCustomRemoteModel *remoteModel =
        [[FIRCustomRemoteModel alloc] initWithName:@"Drug-Detector"];

    FIRModelInterpreter *interpreter;

    // Verifie si le modèle distant est télécharge sinon utilise le modèle local (a implementer avec mMode et mName ou [[FIRModelManager modelManager] isModelDownloaded:remoteModel ] a la place de mMode )
//    if (mMode) {
//      interpreter = [FIRModelInterpreter modelInterpreterForRemoteModel:mName];
//    } else {
      interpreter = [FIRModelInterpreter modelInterpreterForLocalModel:localModel];
//    }

    FIRModelInputOutputOptions *ioOptions = [[FIRModelInputOutputOptions alloc] init];

    // Definition des paramètres d'entrée et de sortie
    NSError *error;
    [ioOptions setInputFormatForIndex:0
                                type:FIRModelElementTypeFloat32
                          dimensions:@[@1, @256, @256, @3]
                                error:&error];

    [ioOptions setOutputFormatForIndex:0
                                  type:FIRModelElementTypeFloat32
                            dimensions:@[@1, @3, @132]
                                error:&error];

    // Cette partie permet de récuperer la frame de la camera puis de la convertir en inputData pour le modèl
    // Source du code https://firebase.google.com/docs/ml/ios/use-custom-models
    // (à implementer rognage de l'image en carré pour éviter d'avoir une image déformée)
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

    for (int row = 0; row < 256; row++) {
      for (int col = 0; col < 256; col++) {
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

    // Ici on fait tourner le modèle sur inputData
    [interpreter runWithInputs:inputs
                   options:ioOptions
                completion:^(FIRModelOutputs * _Nullable outputs,
                             NSError * _Nullable error) {
        NSError *outputError;
        // Array de dimensions [3][x] contenant les informations sur les médicaments les plus probables
        NSArray *labelProbArray = [outputs outputAtIndex:0 error:&outputError][0];
        
        NSMutableArray *resultat = [NSMutableArray new];
        
        NSLog(@"%@",[error localizedDescription]);
        
        // On decode la chaine contenant les informations sur le médicament qui est encodé en Unicode
        NSMutableString *nomMedicament = [[NSMutableString alloc] init];

        for (int i = 1; i < [labelProbArray[0] count]; i++)
        {
            unichar hexa = (unsigned long)[labelProbArray[0][i] integerValue];
            NSString* charactere = [NSString stringWithCharacters:&hexa length:1];
            [nomMedicament appendString:charactere];
        }

        // La première valeur de l'array de sortie contient la probabilité du médicament 0 si le médicament n'est pas reconnu
        resultat[0] = labelProbArray[0][0];

        resultat[1] = nomMedicament;

        completed(resultat);
    }];

}

@end

