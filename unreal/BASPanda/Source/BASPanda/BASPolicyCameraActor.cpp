#include "BASPolicyCameraActor.h"

#include "Camera/CameraComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/PointLightComponent.h"
#include "Components/SceneComponent.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "Math/RotationMatrix.h"

ABASPolicyCameraActor::ABASPolicyCameraActor()
{
    PrimaryActorTick.bCanEverTick = false;
    CameraRoot = CreateDefaultSubobject<USceneComponent>(TEXT("CameraRoot"));
    RootComponent = CameraRoot;

    PreviewCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("PreviewCamera"));
    PreviewCamera->SetupAttachment(CameraRoot);
    PreviewCamera->SetAspectRatio(4.0f / 3.0f);
    PreviewCamera->bConstrainAspectRatio = true;

    ObservationCapture = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("ObservationCapture"));
    ObservationCapture->SetupAttachment(CameraRoot);
    ObservationCapture->bCaptureEveryFrame = false;
    ObservationCapture->bCaptureOnMovement = false;
    ObservationCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    CameraFillLight = CreateDefaultSubobject<UPointLightComponent>(TEXT("CameraFillLight"));
    CameraFillLight->SetupAttachment(CameraRoot);
    CameraFillLight->SetRelativeLocation(FVector(140.0, 0.0, 110.0));
    CameraFillLight->SetIntensity(3500.0f);
    CameraFillLight->SetAttenuationRadius(350.0f);
    CameraFillLight->SetCastShadows(false);
}

void ABASPolicyCameraActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);

    // Orbit pose as a MuJoCo free camera (azimuth 90: looking back at the robot along -x). UE uses cm and
    // its OBJ import reflects Y; camera and target both have Y=0 here.
    const FVector LookAtCm = LookAtM * 100.0;
    const float Elevation = FMath::DegreesToRadians(ElevationDeg);
    const FVector CameraCm = LookAtCm + 100.0 * DistanceM * FVector(FMath::Cos(Elevation), 0.0, FMath::Sin(Elevation));
    const FRotator CameraRotation = FRotationMatrix::MakeFromXZ(
        LookAtCm - CameraCm, FVector::UpVector).Rotator();

    // UE camera FOV is horizontal; the frame is 4:3.
    const float HorizontalFov = FMath::RadiansToDegrees(2.0f * FMath::Atan(
        (4.0f / 3.0f) * FMath::Tan(FMath::DegreesToRadians(VerticalFovDeg) / 2.0f)));
    PreviewCamera->SetRelativeLocationAndRotation(CameraCm, CameraRotation);
    PreviewCamera->SetFieldOfView(HorizontalFov);
    ObservationCapture->SetRelativeLocationAndRotation(CameraCm, CameraRotation);
    ObservationCapture->FOVAngle = HorizontalFov;
    ObservationCapture->TextureTarget = ObservationTarget;
}

bool ABASPolicyCameraActor::CaptureRGB(TArray<uint8>& OutRGB, FString& OutError)
{
    if (!ObservationTarget)
    {
        ObservationTarget = NewObject<UTextureRenderTarget2D>(this, TEXT("TransientPolicyObservation"));
        ObservationTarget->RenderTargetFormat = RTF_RGBA8;
        ObservationTarget->InitAutoFormat(320, 240);
        ObservationTarget->UpdateResourceImmediate(true);
    }
    if (ObservationTarget->SizeX != 320 || ObservationTarget->SizeY != 240)
    {
        OutError = TEXT("Observation target must be 320x240 pixels");
        return false;
    }
    ObservationCapture->TextureTarget = ObservationTarget;
    ObservationCapture->CaptureScene();

    TArray<FColor> Pixels;
    FTextureRenderTargetResource* Resource = ObservationTarget->GameThread_GetRenderTargetResource();
    if (!Resource || !Resource->ReadPixels(Pixels) || Pixels.Num() != 320 * 240)
    {
        OutError = TEXT("Scene capture readback failed");
        return false;
    }
    OutRGB.SetNumUninitialized(Pixels.Num() * 3);
    for (int32 Index = 0; Index < Pixels.Num(); ++Index)
    {
        OutRGB[Index * 3] = Pixels[Index].R;
        OutRGB[Index * 3 + 1] = Pixels[Index].G;
        OutRGB[Index * 3 + 2] = Pixels[Index].B;
    }
    return true;
}
