#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "PandaRobotActor.generated.h"

class USceneComponent;

/** Kinematic Panda preview assembled from the imported MJCF visual links. */
UCLASS(Blueprintable)
class BASPANDA_API APandaRobotActor : public AActor
{
    GENERATED_BODY()

public:
    APandaRobotActor();

    virtual void OnConstruction(const FTransform& Transform) override;
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;

    /** Joint angles in radians, ordered joint1 through joint7. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Panda")
    TArray<float> ArmJointsRad;

    /** Each finger slides this far from closed; 4 cm is fully open. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Panda", meta = (ClampMin = "0.0", ClampMax = "4.0"))
    float FingerOpeningCm = 4.0f;

    /** Opt-in runtime check: sweep joint 1 by at most 0.1 rad, then restore its original pose. */
    UPROPERTY(EditAnywhere, Category = "Panda|Motion Test")
    bool bRunMotionTestOnBeginPlay = false;

    UPROPERTY(VisibleAnywhere, Category = "Panda|Motion Test")
    FString MotionTestStatus = TEXT("Not run");

    UFUNCTION(CallInEditor, BlueprintCallable, Category = "Panda")
    void ResetHome();

    /** Show which imported meshes are attached and their actual Unreal bounds. */
    UFUNCTION(CallInEditor, Category = "Panda|Diagnostics")
    void InspectImportedLinks();

    /** Use after changing ArmJointsRad or FingerOpeningCm at runtime. */
    UFUNCTION(BlueprintCallable, Category = "Panda")
    void ApplyJointPose();

    /** Accept one bounded absolute joint target from the local policy bridge. Runtime only. */
    bool CommandJointTargets(const TArray<float>& JointTargetsRad, float GripperOpenness);

    /** Approximate centre between the fingertip pads, in world coordinates. */
    UFUNCTION(BlueprintPure, Category = "Panda")
    FVector GetGraspCenterWorld() const;

    USceneComponent* GetGraspFrame() const { return GraspFrame; }

private:
    UPROPERTY(VisibleAnywhere, Category = "Panda")
    TObjectPtr<USceneComponent> PandaRoot;

    UPROPERTY(Transient)
    TArray<TObjectPtr<USceneComponent>> MotionFrames;

    UPROPERTY(VisibleAnywhere, Category = "Panda")
    TObjectPtr<USceneComponent> GraspFrame;

    TArray<float> MotionTestStartJoints;
    float MotionTestElapsed = 0.0f;
    bool bMotionTestActive = false;
    TArray<float> CommandTargetsRad;
    float CommandFingerTargetCm = 4.0f;
    bool bCommandActive = false;
};
