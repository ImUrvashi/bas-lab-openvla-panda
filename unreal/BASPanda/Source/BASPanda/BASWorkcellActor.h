#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BASWorkcellActor.generated.h"

class UPointLightComponent;
class UPostProcessComponent;
class USceneComponent;
class UStaticMeshComponent;
class APandaRobotActor;

/** The five bench tasks; only the active task's props are visible. */
UENUM(BlueprintType)
enum class EBASTask : uint8
{
    CubeToPad       UMETA(DisplayName = "1 Cube to yellow pad"),
    CanisterToDock  UMETA(DisplayName = "2 Canister to analysis dock"),
    ToolCaseToTray  UMETA(DisplayName = "3 Tool case to storage tray"),
    SortSamples     UMETA(DisplayName = "4 Sort sample by colour"),
    PressButton     UMETA(DisplayName = "5 Press panel button"),
};

/** BAS workcell at the MuJoCo world origin: bench, task props, grasp proxy and success checks. */
UCLASS(Blueprintable)
class BASPANDA_API ABASWorkcellActor : public AActor
{
    GENERATED_BODY()

public:
    ABASWorkcellActor();
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;
    virtual void OnConstruction(const FTransform& Transform) override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BAS Workcell|Task")
    EBASTask Task = EBASTask::CubeToPad;

    /** Task 4: 0 red sample, 1 blue sample. Task 5: 0 green button, 1 red button. */
    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task", meta = (ClampMin = "0", ClampMax = "1"))
    int32 TargetVariant = 0;

    /** Object start and target centre in MuJoCo metres (x forward, y left). Task 4: red sample / red pad. Task 5: console. */
    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task")
    FVector2D CubeStartXY = FVector2D(0.45, 0.0);

    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task")
    FVector2D MarkerXY = FVector2D(0.60, 0.0);

    /** Task 4 only: blue sample start and blue pad centre. */
    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task")
    FVector2D OtherObjectXY = FVector2D(0.45, 0.10);

    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task")
    FVector2D OtherTargetXY = FVector2D(0.62, -0.10);

    /** Task 5: green button on the right (-y) instead of the left. */
    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task")
    bool bSwapButtons = false;

    /** True while the gripper proxy is carrying an object. */
    UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "BAS Workcell|Task")
    bool bObjectHeld = false;

    /** Show the active task's props, place them at the layout above and reset buttons. */
    UFUNCTION(CallInEditor, BlueprintCallable, Category = "BAS Workcell|Task")
    void ResetTask();

    /** Random layout (and target variant) for the active task, inside the reachable, visible bench area. */
    void RandomizeLayout(FRandomStream& Stream);

    FString GetTaskInstruction() const;

    /** Object the task acts on (button top for task 5), MuJoCo metres. */
    FVector GetActiveObjectMujoco() const;

    /** Target centre at its top surface, MuJoCo metres. */
    FVector GetTargetMujoco() const;

    /** Task complete: object released at the target (or the right button pressed and the hand clear). */
    UFUNCTION(BlueprintPure, Category = "BAS Workcell|Task")
    bool IsTaskSuccess() const;

    UFUNCTION(BlueprintPure, Category = "BAS Workcell|Task")
    bool IsCubePlaced() const;

    /** Fixed-layout controller check for task 1; uses precomputed Panda IK poses, not OpenVLA. */
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

    // Task 1
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 1")
    TObjectPtr<UStaticMeshComponent> TargetMarker;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 1")
    TObjectPtr<UStaticMeshComponent> PickCube;

    // Task 2
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 2")
    TObjectPtr<UStaticMeshComponent> Canister;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 2")
    TObjectPtr<UStaticMeshComponent> Dock;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 2")
    TObjectPtr<UStaticMeshComponent> DockCentre;

    // Task 3
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 3")
    TObjectPtr<UStaticMeshComponent> ToolCase;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 3")
    TObjectPtr<UStaticMeshComponent> TrayFloor;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 3")
    TArray<TObjectPtr<UStaticMeshComponent>> TrayWalls;

    // Task 4
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 4")
    TObjectPtr<UStaticMeshComponent> RedSample;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 4")
    TObjectPtr<UStaticMeshComponent> BlueSample;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 4")
    TObjectPtr<UStaticMeshComponent> RedPad;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 4")
    TObjectPtr<UStaticMeshComponent> BluePad;

    // Task 5
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<UStaticMeshComponent> Console;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<UStaticMeshComponent> GreenButton;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<UStaticMeshComponent> RedButton;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<UStaticMeshComponent> GreenLight;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<UStaticMeshComponent> RedLight;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UPointLightComponent> WorkcellLight;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UPostProcessComponent> WorkcellExposure;

    UPROPERTY(EditInstanceOnly, Category = "BAS Workcell|Task")
    TObjectPtr<APandaRobotActor> PandaRobot;

    UPROPERTY(Transient)
    TObjectPtr<UStaticMeshComponent> HeldPart;

    TArray<UStaticMeshComponent*> TaskParts(EBASTask ForTask) const;
    TArray<UStaticMeshComponent*> ActiveGraspables() const;
    float FingerStopFor(const UStaticMeshComponent* Part) const;
    void PlaceTaskProps();
    void ApplyTaskVisibility(bool bPlaying);
    void ResetMovable(UStaticMeshComponent* Part, const FVector& Metres);
    void UpdateButtons();
    void SetButtonPressed(UStaticMeshComponent* Button, UStaticMeshComponent* Light, bool bPressed, float DepthCm);
    FVector ToMujoco(const UStaticMeshComponent* Part) const;
    bool IsReleased() const;

    void Grasp(UStaticMeshComponent* Part);
    void Release();
    void AdvanceScriptedTest(float DeltaSeconds);

    bool bGreenPressed = false;
    bool bRedPressed = false;
    FVector2D OtherObjectStartXY;
    int32 ScriptedPhase = -1;
    float ScriptedPhaseElapsed = 0.0f;
};
