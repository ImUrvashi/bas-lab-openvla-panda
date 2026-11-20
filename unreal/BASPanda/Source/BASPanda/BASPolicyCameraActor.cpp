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

    // MuJoCo free camera: lookat (0.50, 0, 0.46) m, distance 0.75 m,
    // azimuth 90 deg, elevation -15 deg. UE uses cm and its OBJ import
    // reflects Y; both camera and target have Y=0 here.
    const FVector LookAtCm(50.0, 0.0, 46.0);
    const FVector CameraCm(50.0 + 75.0 * FMath::Cos(FMath::DegreesToRadians(15.0)),
        0.0, 46.0 + 75.0 * FMath::Sin(FMath::DegreesToRadians(15.0)));
    const FRotator CameraRotation = FRotationMatrix::MakeFromXZ(
        LookAtCm - CameraCm, FVector::UpVector).Rotator();

    // MuJoCo's scene uses 45 deg vertical FOV. UE camera FOV is horizontal.
    const float HorizontalFov = FMath::RadiansToDegrees(2.0f * FMath::Atan(
        (4.0f / 3.0f) * FMath::Tan(FMath::DegreesToRadians(45.0f) / 2.0f)));
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
        ObservationTarget->InitAutoFormat(224, 224);
        ObservationTarget->UpdateResourceImmediate(true);
    }
    if (ObservationTarget->SizeX != 224 || ObservationTarget->SizeY != 224)
    {
        OutError = TEXT("Observation target must be 224x224 pixels");
        return false;
    }
    ObservationCapture->TextureTarget = ObservationTarget;
    ObservationCapture->CaptureScene();

    TArray<FColor> Pixels;
    FTextureRenderTargetResource* Resource = ObservationTarget->GameThread_GetRenderTargetResource();
    if (!Resource || !Resource->ReadPixels(Pixels) || Pixels.Num() != 224 * 224)
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
