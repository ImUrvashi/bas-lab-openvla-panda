#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BASWorkcellActor.generated.h"

class UPointLightComponent;
class UPostProcessComponent;
class USceneComponent;
class UStaticMeshComponent;
class APandaRobotActor;

/** The five station tasks; only the active task's props are visible. */
UENUM(BlueprintType)
enum class EBASTask : uint8
{
    CubeToPad      UMETA(DisplayName = "1 Sample cube to processing pad"),
    StowInDrawer   UMETA(DisplayName = "2 Stow tool in drawer and close it"),
    CloseValve     UMETA(DisplayName = "3 Close the coolant valve"),
    InsertFilter   UMETA(DisplayName = "4 Insert filter cartridge"),
    SealContainer  UMETA(DisplayName = "5 Vial into container, close lid"),
};

/** Mechanism handle currently gripped (drawer knob, valve lever, lid tab). */
UENUM()
enum class EBASHandle : uint8
{
    None,
    Drawer,
    Valve,
    Lid,
};

/** BAS workcell at the MuJoCo world origin: bench, task props, kinematic grasp/mechanisms and success checks. */
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

    /** Loose object start (cube, tool, cartridge, vial), MuJoCo metres (x forward, y left). */
    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task")
    FVector2D CubeStartXY = FVector2D(0.45, 0.0);

    /** Target or fixture centre: marker, drawer cabinet, valve, scrubber, container. */
    UPROPERTY(EditAnywhere, Category = "BAS Workcell|Task")
    FVector2D MarkerXY = FVector2D(0.60, 0.0);

    /** Show the active task's props at the layout above, with mechanisms in their start state. */
    UFUNCTION(CallInEditor, BlueprintCallable, Category = "BAS Workcell|Task")
    void ResetTask();

    /** Random layout for the active task, inside the reachable, visible bench area. */
    void RandomizeLayout(FRandomStream& Stream);

    FString GetTaskInstruction() const;

    /** Loose object (task 3: valve lever), MuJoCo metres. */
    FVector GetActiveObjectMujoco() const;

    /** Where the object must end up (marker, drawer floor, slot, container floor; task 3: lever). */
    FVector GetTargetMujoco() const;

    /** Privileged state for scripted experts: positions (MuJoCo metres) and mechanism values. */
    void GetTaskState(TMap<FString, double>& Out) const;

    UFUNCTION(BlueprintPure, Category = "BAS Workcell|Task")
    bool IsTaskSuccess() const;

    UFUNCTION(BlueprintPure, Category = "BAS Workcell|Task")
    bool IsCubePlaced() const;

    UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "BAS Workcell|State")
    bool bObjectHeld = false;

    UPROPERTY(VisibleInstanceOnly, Category = "BAS Workcell|State")
    EBASHandle HeldHandle = EBASHandle::None;

    UPROPERTY(VisibleInstanceOnly, Category = "BAS Workcell|State")
    float DrawerOpenM = 0.0f;

    UPROPERTY(VisibleInstanceOnly, Category = "BAS Workcell|State")
    float ValveDeg = 0.0f;

    UPROPERTY(VisibleInstanceOnly, Category = "BAS Workcell|State")
    float LidDeg = 90.0f;

    /** Tool resting in the drawer (task 2) or vial in the container (task 5). */
    UPROPERTY(VisibleInstanceOnly, Category = "BAS Workcell|State")
    bool bItemStowed = false;

    UPROPERTY(VisibleInstanceOnly, Category = "BAS Workcell|State")
    bool bCartridgeSeated = false;

    UPROPERTY(VisibleInstanceOnly, Category = "BAS Workcell|State")
    bool bLidLatched = false;

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

    // Task 2: cabinet with a sliding drawer, loose tool
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 2")
    TObjectPtr<USceneComponent> CabinetRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 2")
    TObjectPtr<USceneComponent> DrawerRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 2")
    TObjectPtr<UStaticMeshComponent> DrawerKnob;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 2")
    TObjectPtr<UStaticMeshComponent> Tool;

    // Task 3: pipe with a quarter-turn valve
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 3")
    TObjectPtr<USceneComponent> ValveRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 3")
    TObjectPtr<USceneComponent> ValveLeverRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 3")
    TObjectPtr<UStaticMeshComponent> ValveLever;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 3")
    TObjectPtr<UStaticMeshComponent> ValveLight;

    // Task 4: air scrubber with a cartridge slot, loose cartridge
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 4")
    TObjectPtr<USceneComponent> ScrubberRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 4")
    TObjectPtr<UStaticMeshComponent> ScrubberLight;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 4")
    TObjectPtr<UStaticMeshComponent> Cartridge;

    // Task 5: container with a hinged lid, loose vial
    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<USceneComponent> ContainerRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<USceneComponent> LidRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<UStaticMeshComponent> LidTab;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<UStaticMeshComponent> ContainerLight;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell|Task 5")
    TObjectPtr<UStaticMeshComponent> Vial;

    /** Decorative prop parts and their colours, applied in OnConstruction. */
    UPROPERTY()
    TArray<TObjectPtr<UStaticMeshComponent>> ColouredParts;

    UPROPERTY()
    TArray<FLinearColor> PartColours;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UPointLightComponent> WorkcellLight;

    UPROPERTY(VisibleAnywhere, Category = "BAS Workcell")
    TObjectPtr<UPostProcessComponent> WorkcellExposure;

    UPROPERTY(EditInstanceOnly, Category = "BAS Workcell|Task")
    TObjectPtr<APandaRobotActor> PandaRobot;

    UPROPERTY(Transient)
    TObjectPtr<UStaticMeshComponent> HeldPart;

    UStaticMeshComponent* ActiveObject() const;
    USceneComponent* ActiveFixture() const;
    UStaticMeshComponent* ActiveHandle() const;
    bool CanGraspObject() const;
    float FingerStopFor(const UStaticMeshComponent* Part) const;
    FVector ToMujoco(const FVector& World) const;
    FVector LocalMujoco(const USceneComponent* Frame, const FVector& World) const;
    void ApplyTaskVisibility();
    void PlaceTaskProps();
    void SetFree(UStaticMeshComponent* Part, bool bFree);
    void SnapInto(UStaticMeshComponent* Part, USceneComponent* Frame, const FVector& LocalMetres);
    void UpdateMechanisms();
    void TryGrasp();

    void Grasp(UStaticMeshComponent* Part);
    void Release();
    void OnObjectReleased(UStaticMeshComponent* Part);
    void GripHandle(EBASHandle Handle);
    void ReleaseHandle();
    void DriveHandle();
    void AdvanceScriptedTest(float DeltaSeconds);

    FVector GripStartMujoco = FVector::ZeroVector;
    float GripStartValue = 0.0f;
    float GripStartYaw = 0.0f;
    int32 ScriptedPhase = -1;
    float ScriptedPhaseElapsed = 0.0f;
};
