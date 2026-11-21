#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BASPolicyCameraActor.generated.h"

class UCameraComponent;
class UPointLightComponent;
class USceneCaptureComponent2D;
class USceneComponent;
class UTextureRenderTarget2D;

/** Policy camera framing the whole station bench (all five tasks' props). */
UCLASS(Blueprintable)
class BASPANDA_API ABASPolicyCameraActor : public AActor
{
    GENERATED_BODY()

public:
    ABASPolicyCameraActor();
    virtual void OnConstruction(const FTransform& Transform) override;

    /** Capture one 320x240 raw RGB frame, as MuJoCo renders; the gateway resizes it to 224x224. */
    bool CaptureRGB(TArray<uint8>& OutRGB, FString& OutError);

    /** Orbit camera in MuJoCo terms (metres, degrees): looks at LookAtM from DistanceM, ElevationDeg down, from +x. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Camera")
    FVector LookAtM = FVector(0.56, 0.0, 0.46);

    UPROPERTY(EditAnywhere, Category = "BAS Policy Camera")
    float DistanceM = 0.9f;

    UPROPERTY(EditAnywhere, Category = "BAS Policy Camera")
    float ElevationDeg = 40.0f;

    /** Vertical field of view; the 320x240 frame is squashed to 224x224 by the gateway, as in training. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Camera")
    float VerticalFovDeg = 45.0f;

    /** Optional 320x240 render target; a transient one is created if empty. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BAS Policy Camera")
    TObjectPtr<UTextureRenderTarget2D> ObservationTarget;

private:
    UPROPERTY(VisibleAnywhere, Category = "BAS Policy Camera")
    TObjectPtr<USceneComponent> CameraRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Policy Camera")
    TObjectPtr<UCameraComponent> PreviewCamera;

    UPROPERTY(VisibleAnywhere, Category = "BAS Policy Camera")
    TObjectPtr<USceneCaptureComponent2D> ObservationCapture;

    /** Soft camera-side fill approximates MuJoCo's ambient headlight in the enclosed lab. */
    UPROPERTY(VisibleAnywhere, Category = "BAS Policy Camera")
    TObjectPtr<UPointLightComponent> CameraFillLight;
};
