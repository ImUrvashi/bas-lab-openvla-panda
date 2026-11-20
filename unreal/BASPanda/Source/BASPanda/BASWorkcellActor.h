#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BASWorkcellActor.generated.h"

class UPointLightComponent;
class UPostProcessComponent;
class USceneComponent;
class UStaticMeshComponent;
class APandaRobotActor;

/** First BAS workcell preview, placed at the MuJoCo world origin. */
UCLASS(Blueprintable)
class BASPANDA_API ABASWorkcellActor : public AActor
{
    GENERATED_BODY()

public:
    ABASWorkcellActor();
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;
    virtual void OnConstruction(const FTransform& Transform) override;

    /** Return the cube to its source pose. During play, physics resumes after the reset. */
    UFUNCTION(CallInEditor, BlueprintCallable, Category = "BAS Workcell|Task")
    void ResetCube();

    /** True while the first-pass gripper proxy is carrying the cube. */
    UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "BAS Workcell|Task")
    bool bCubeHeld = false;

    UFUNCTION(BlueprintPure, Category = "BAS Workcell|Task")
    bool IsCubePlaced() const;

    /** Cube start and marker centre in MuJoCo metres (x forward, y left). */
    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task")
    FVector2D CubeStartXY = FVector2D(0.45, 0.0);

    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task")
    FVector2D MarkerXY = FVector2D(0.60, 0.0);

    /** Put the marker at MarkerXY and the cube back at CubeStartXY. */
    UFUNCTION(CallInEditor, BlueprintCallable, Category = "BAS Workcell|Task")
    void ResetTask();

    /** Draw a layout from the MuJoCo training ranges (cube/marker at least 13 cm apart). */
    void RandomizeLayout(FRandomStream& Stream);

    /** Cube centre in MuJoCo metres. */
    FVector GetCubePositionMujoco() const;

    /** Fixed-layout controller check; uses precomputed Panda IK poses, not OpenVLA. */
    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Scripted Test")
    bool bRunScriptedPickPlaceOnBeginPlay = false;

    UPROPERTY(VisibleInstanceOnly, Category = "BAS Workcell|Scripted Test")
    FString ScriptedTestStatus = TEXT("Not run");

private:
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<USceneComponent> WorkcellRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UStaticMeshComponent> RobotMount;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UStaticMeshComponent> MountPlate;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UStaticMeshComponent> BenchRack;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UStaticMeshComponent> BenchTop;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UStaticMeshComponent> TargetMarker;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UStaticMeshComponent> PickCube;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UPointLightComponent> WorkcellLight;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UPostProcessComponent> WorkcellExposure;

    UPROPERTY(EditInstanceOnly, Category = "BAS Workcell|Task")
    TObjectPtr<APandaRobotActor> PandaRobot;

    void GraspCube();
    void ReleaseCube();
    void AdvanceScriptedTest(float DeltaSeconds);

    int32 ScriptedPhase = -1;
    float ScriptedPhaseElapsed = 0.0f;
};
