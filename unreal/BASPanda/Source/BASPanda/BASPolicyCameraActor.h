#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BASPolicyCameraActor.generated.h"

class UCameraComponent;
class UPointLightComponent;
class USceneCaptureComponent2D;
class USceneComponent;
class UTextureRenderTarget2D;

/** Camera matching closed_loop_panda.SIDE_CAMERA for the first OpenVLA observation prototype. */
UCLASS(Blueprintable)
class BASPANDA_API ABASPolicyCameraActor : public AActor
{
    GENERATED_BODY()

public:
    ABASPolicyCameraActor();
    virtual void OnConstruction(const FTransform& Transform) override;

    /** Capture one 224x224 raw RGB observation for the local policy protocol. */
    bool CaptureRGB(TArray<uint8>& OutRGB, FString& OutError);

    /** Assign a 320x240 render target when the observation bridge is ready. */
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
