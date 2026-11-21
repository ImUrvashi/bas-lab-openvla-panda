#include "BASWorkcellActor.h"
#include "PandaRobotActor.h"

#include "Components/PointLightComponent.h"
#include "Components/PostProcessComponent.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "Kismet/GameplayStatics.h"
#include "UObject/UObjectGlobals.h"

namespace
{
    struct FScriptedPose
    {
        const TCHAR* Name;
        float Joints[7];
        float GripperOpenness;
        bool bRequiresCube;
    };

    // Fixed workcell poses from the repository's PandaKinematics IK solver.
    // The source "hand" sits 9 cm above the grasp centre. At Z=49.4 cm,
    // the grasp centre is at the cube's 40 cm resting height.
    const FScriptedPose ScriptedPoses[] = {
        {TEXT("Approach cube"), {0, -0.34041444f, 0, -1.99879844f, 0, 1.65838258f, -0.7853f}, 1.0f, false},
        {TEXT("Descend to cube"), {0, -0.33669018f, 0, -2.26606778f, 0, 1.92937684f, -0.7853f}, 1.0f, false},
        {TEXT("Close gripper"), {0, -0.33669018f, 0, -2.26606778f, 0, 1.92937684f, -0.7853f}, 0.5f, false},
        {TEXT("Lift cube"), {0, -0.34045004f, 0, -1.99885625f, 0, 1.65840766f, -0.7853f}, 0.5f, true},
        {TEXT("Move over marker"), {0, 0.14075931f, 0, -1.45550379f, 0, 1.59626457f, -0.7853f}, 0.5f, true},
        {TEXT("Lower to marker"), {0, 0.12363692f, 0, -1.74907139f, 0, 1.87270772f, -0.7853f}, 0.5f, true},
        {TEXT("Release cube"), {0, 0.12363692f, 0, -1.74907139f, 0, 1.87270772f, -0.7853f}, 1.0f, true},
    };

    // Prop geometry in MuJoCo metres. Bench top is at z 0.375. Fingers close along x,
    // so every graspable's narrow side faces x.
    constexpr double BenchZ = 0.375;
    constexpr double CanisterR = 0.022, CanisterH = 0.08;
    constexpr double DockR = 0.045, DockH = 0.006;
    const FVector CaseHalf(0.02, 0.03, 0.015);
    const FVector2D TrayHalf(0.06, 0.08);
    constexpr double TrayFloorT = 0.004, TrayWall = 0.006, TrayWallH = 0.035;
    constexpr double SampleR = 0.015, SampleH = 0.06;
    constexpr double PadHalf = 0.04;
    const FVector ConsoleHalf(0.05, 0.08, 0.02);
    constexpr double ButtonR = 0.0125, ButtonH = 0.012, ButtonSideY = 0.045, ButtonBackX = -0.01, LightFrontX = 0.03;
    constexpr double ButtonTop = BenchZ + 2 * 0.02 + ButtonH;
    constexpr float ButtonTravelCm = 0.8f, ButtonLatchCm = 0.6f, ButtonCatchCm = 1.6f;
    // The fingertips reach about 2 cm below the grasp centre between the pads.
    constexpr float FingertipBelowGraspCm = 2.0f;

    // MuJoCo metres (y left) -> Unreal centimetres (y right): the one place this flip happens.
    FVector MujocoToUnrealCm(const FVector& Metres)
    {
        return FVector(Metres.X, -Metres.Y, Metres.Z) * 100.0;
    }

    FVector At(const FVector2D& XY, double Z)
    {
        return MujocoToUnrealCm(FVector(XY.X, XY.Y, Z));
    }

    // Engine/BasicShapes/Cube and Cylinder are 100 cm across; sizes below are in metres.
    void ConfigureBox(UStaticMeshComponent* Part, UStaticMesh* CubeMesh,
        const FVector& PositionMetres, const FVector& HalfSizeMetres, bool bCollides)
    {
        Part->SetStaticMesh(CubeMesh);
        Part->SetRelativeLocation(PositionMetres * 100.0);
        Part->SetRelativeScale3D(HalfSizeMetres * 2.0);
        Part->SetCollisionEnabled(bCollides ? ECollisionEnabled::QueryAndPhysics : ECollisionEnabled::NoCollision);
    }

    void ConfigureCylinder(UStaticMeshComponent* Part, UStaticMesh* CylinderMesh, double Radius, double Height)
    {
        Part->SetStaticMesh(CylinderMesh);
        Part->SetRelativeScale3D(FVector(2 * Radius, 2 * Radius, Height));
        Part->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    }

    void SetColour(UStaticMeshComponent* Part, const FLinearColor& Value)
    {
        UMaterialInstanceDynamic* Material = Cast<UMaterialInstanceDynamic>(Part->GetMaterial(0));
        if (!Material)
        {
            UMaterialInterface* Base = LoadObject<UMaterialInterface>(
                nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
            if (!Base)
            {
                return;
            }
            Material = UMaterialInstanceDynamic::Create(Base, Part);
            Part->SetMaterial(0, Material);
        }
        Material->SetVectorParameterValue(TEXT("Color"), Value);
    }

    const FLinearColor GreenDim(0.02f, 0.10f, 0.03f), GreenLit(0.25f, 1.0f, 0.3f);
    const FLinearColor RedDim(0.12f, 0.02f, 0.02f), RedLit(1.0f, 0.2f, 0.1f);
}

ABASWorkcellActor::ABASWorkcellActor()
{
    PrimaryActorTick.bCanEverTick = true;
    WorkcellRoot = CreateDefaultSubobject<USceneComponent>(TEXT("WorkcellRoot"));
    RootComponent = WorkcellRoot;

    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
    UStaticMesh* CylinderMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    const auto Make = [this](const TCHAR* Name)
    {
        UStaticMeshComponent* Part = CreateDefaultSubobject<UStaticMeshComponent>(Name);
        Part->SetupAttachment(WorkcellRoot);
        return Part;
    };

    RobotMount = Make(TEXT("RobotMount"));
    ConfigureBox(RobotMount, CubeMesh, FVector(-0.05, 0, -0.515), FVector(0.25, 0.40, 0.515), true);
    MountPlate = Make(TEXT("MountPlate"));
    ConfigureBox(MountPlate, CubeMesh, FVector(0, 0, -0.015), FVector(0.20, 0.20, 0.015), false);
    BenchRack = Make(TEXT("BenchRack"));
    ConfigureBox(BenchRack, CubeMesh, FVector(0.55, 0, -0.3525), FVector(0.35, 0.40, 0.6775), true);
    BenchTop = Make(TEXT("BenchTop"));
    ConfigureBox(BenchTop, CubeMesh, FVector(0.55, 0, 0.35), FVector(0.35, 0.40, 0.025), true);

    TargetMarker = Make(TEXT("TargetMarker"));
    ConfigureBox(TargetMarker, CubeMesh, FVector(0.60, 0, 0.3765), FVector(0.05, 0.05, 0.0005), false);
    PickCube = Make(TEXT("PickCube"));
    PickCube->SetMobility(EComponentMobility::Movable);
    ConfigureBox(PickCube, CubeMesh, FVector(0.45, 0, 0.40), FVector(0.025, 0.025, 0.025), true);

    Canister = Make(TEXT("Canister"));
    ConfigureCylinder(Canister, CylinderMesh, CanisterR, CanisterH);
    Dock = Make(TEXT("AnalysisDock"));
    ConfigureCylinder(Dock, CylinderMesh, DockR, DockH);
    DockCentre = Make(TEXT("AnalysisDockCentre"));
    ConfigureCylinder(DockCentre, CylinderMesh, 0.025, 0.001);

    ToolCase = Make(TEXT("ToolCase"));
    ConfigureBox(ToolCase, CubeMesh, FVector::ZeroVector, CaseHalf, false);
    TrayFloor = Make(TEXT("StorageTrayFloor"));
    ConfigureBox(TrayFloor, CubeMesh, FVector::ZeroVector, FVector(TrayHalf.X, TrayHalf.Y, TrayFloorT / 2), false);
    const FVector WallHalves[] = {
        FVector(TrayWall / 2, TrayHalf.Y, TrayWallH / 2), FVector(TrayWall / 2, TrayHalf.Y, TrayWallH / 2),
        FVector(TrayHalf.X, TrayWall / 2, TrayWallH / 2), FVector(TrayHalf.X, TrayWall / 2, TrayWallH / 2)};
    for (int32 Index = 0; Index < 4; ++Index)
    {
        UStaticMeshComponent* Wall = Make(*FString::Printf(TEXT("StorageTrayWall%d"), Index));
        ConfigureBox(Wall, CubeMesh, FVector::ZeroVector, WallHalves[Index], false);
        TrayWalls.Add(Wall);
    }

    RedSample = Make(TEXT("RedSample"));
    ConfigureCylinder(RedSample, CylinderMesh, SampleR, SampleH);
    BlueSample = Make(TEXT("BlueSample"));
    ConfigureCylinder(BlueSample, CylinderMesh, SampleR, SampleH);
    RedPad = Make(TEXT("RedPad"));
    ConfigureBox(RedPad, CubeMesh, FVector::ZeroVector, FVector(PadHalf, PadHalf, 0.0005), false);
    BluePad = Make(TEXT("BluePad"));
    ConfigureBox(BluePad, CubeMesh, FVector::ZeroVector, FVector(PadHalf, PadHalf, 0.0005), false);

    Console = Make(TEXT("ControlConsole"));
    ConfigureBox(Console, CubeMesh, FVector::ZeroVector, ConsoleHalf, false);
    GreenButton = Make(TEXT("GreenButton"));
    ConfigureCylinder(GreenButton, CylinderMesh, ButtonR, ButtonH);
    RedButton = Make(TEXT("RedButton"));
    ConfigureCylinder(RedButton, CylinderMesh, ButtonR, ButtonH);
    GreenLight = Make(TEXT("GreenLight"));
    ConfigureCylinder(GreenLight, CylinderMesh, 0.008, 0.004);
    RedLight = Make(TEXT("RedLight"));
    ConfigureCylinder(RedLight, CylinderMesh, 0.008, 0.004);

    for (UStaticMeshComponent* Part : {PickCube.Get(), Canister.Get(), ToolCase.Get(), RedSample.Get(), BlueSample.Get()})
    {
        Part->SetMobility(EComponentMobility::Movable);
    }
    for (UStaticMeshComponent* Part : {GreenButton.Get(), RedButton.Get()})
    {
        Part->SetMobility(EComponentMobility::Movable);
    }

    WorkcellLight = CreateDefaultSubobject<UPointLightComponent>(TEXT("WorkcellLight"));
    WorkcellLight->SetupAttachment(WorkcellRoot);
    WorkcellLight->SetRelativeLocation(FVector(50, 0, 180));
    WorkcellLight->SetIntensity(2500.0f);
    WorkcellLight->SetAttenuationRadius(500.0f);

    WorkcellExposure = CreateDefaultSubobject<UPostProcessComponent>(TEXT("WorkcellExposure"));
    WorkcellExposure->SetupAttachment(WorkcellRoot);
    WorkcellExposure->bUnbound = true;
    WorkcellExposure->Settings.bOverride_AutoExposureBias = true;
    WorkcellExposure->Settings.AutoExposureBias = -1.0f;
}

TArray<UStaticMeshComponent*> ABASWorkcellActor::TaskParts(EBASTask ForTask) const
{
    switch (ForTask)
    {
    case EBASTask::CubeToPad:
        return {TargetMarker, PickCube};
    case EBASTask::CanisterToDock:
        return {Canister, Dock, DockCentre};
    case EBASTask::ToolCaseToTray:
    {
        TArray<UStaticMeshComponent*> Parts = {ToolCase, TrayFloor};
        for (UStaticMeshComponent* Wall : TrayWalls)
        {
            Parts.Add(Wall);
        }
        return Parts;
    }
    case EBASTask::SortSamples:
        return {RedSample, BlueSample, RedPad, BluePad};
    case EBASTask::PressButton:
        return {Console, GreenButton, RedButton, GreenLight, RedLight};
    }
    return {};
}

TArray<UStaticMeshComponent*> ABASWorkcellActor::ActiveGraspables() const
{
    switch (Task)
    {
    case EBASTask::CubeToPad: return {PickCube};
    case EBASTask::CanisterToDock: return {Canister};
    case EBASTask::ToolCaseToTray: return {ToolCase};
    case EBASTask::SortSamples: return {RedSample, BlueSample};
    default: return {};
    }
}

float ABASWorkcellActor::FingerStopFor(const UStaticMeshComponent* Part) const
{
    // Each finger stops at half the object's width along x (as MuJoCo contact would).
    if (Part == Canister) return 100.0f * CanisterR;
    if (Part == ToolCase) return 100.0f * CaseHalf.X;
    if (Part == RedSample || Part == BlueSample) return 100.0f * SampleR;
    return 2.5f;
}

void ABASWorkcellActor::BeginPlay()
{
    Super::BeginPlay();

    // Free bodies in zero gravity, as in MuJoCo; masses from bas_lab/mod_extra_objects.xml where defined.
    const TPair<UStaticMeshComponent*, float> Masses[] = {{PickCube.Get(), 0.08f}, {Canister.Get(), 0.06f},
        {ToolCase.Get(), 0.05f}, {RedSample.Get(), 0.03f}, {BlueSample.Get(), 0.03f}};
    for (const TPair<UStaticMeshComponent*, float>& Mass : Masses)
    {
        Mass.Key->SetMassOverrideInKg(NAME_None, Mass.Value, true);
        Mass.Key->SetLinearDamping(0.01f);
        Mass.Key->SetAngularDamping(0.01f);
    }

    if (!PandaRobot)
    {
        PandaRobot = Cast<APandaRobotActor>(UGameplayStatics::GetActorOfClass(this, APandaRobotActor::StaticClass()));
    }
    ResetTask();
    if (bRunScriptedPickPlaceOnBeginPlay)
    {
        if (Task != EBASTask::CubeToPad || !PandaRobot || !GetActorLocation().IsNearlyZero(1.0f) ||
            !PandaRobot->GetActorLocation().IsNearlyZero(1.0f) ||
            !GetActorRotation().IsNearlyZero() || !PandaRobot->GetActorRotation().IsNearlyZero())
        {
            ScriptedTestStatus = TEXT("Failed: needs task 1, workcell and Panda at world origin with zero rotation");
            UE_LOG(LogTemp, Warning, TEXT("BAS scripted test: %s"), *ScriptedTestStatus);
        }
        else
        {
            PandaRobot->ResetHome();
            ScriptedPhase = 0;
            ScriptedPhaseElapsed = 0.0f;
            ScriptedTestStatus = TEXT("Running: approach cube");
            UE_LOG(LogTemp, Display, TEXT("BAS scripted test: %s"), *ScriptedTestStatus);
        }
    }
}

void ABASWorkcellActor::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    if (!PandaRobot)
    {
        return;
    }

    if (HeldPart)
    {
        if (PandaRobot->FingerOpeningCm >= FingerStopFor(HeldPart) + 0.7f)
        {
            Release();
        }
    }
    else
    {
        // Grasp proxy: closing fingers within 6 cm of an object's centre pick up the nearest one.
        const FVector GraspPoint = PandaRobot->GetGraspCenterWorld();
        UStaticMeshComponent* Best = nullptr;
        float BestDist = 6.0f;
        for (UStaticMeshComponent* Part : ActiveGraspables())
        {
            const float Dist = FVector::Dist(Part->GetComponentLocation(), GraspPoint);
            if (PandaRobot->FingerOpeningCm <= FingerStopFor(Part) + 0.2f && Dist <= BestDist)
            {
                Best = Part;
                BestDist = Dist;
            }
        }
        if (Best)
        {
            Grasp(Best);
        }
    }
    if (Task == EBASTask::PressButton)
    {
        UpdateButtons();
    }
    if (ScriptedPhase >= 0)
    {
        AdvanceScriptedTest(DeltaSeconds);
    }
}

void ABASWorkcellActor::AdvanceScriptedTest(float DeltaSeconds)
{
    if (ScriptedPhase >= UE_ARRAY_COUNT(ScriptedPoses))
    {
        ScriptedPhase = -1;
        return;
    }
    const FScriptedPose& Pose = ScriptedPoses[ScriptedPhase];
    ScriptedPhaseElapsed += DeltaSeconds;
    if (ScriptedPhaseElapsed > 8.0f ||
        (Pose.bRequiresCube && ScriptedPhase < UE_ARRAY_COUNT(ScriptedPoses) - 1 && !bObjectHeld))
    {
        ScriptedTestStatus = FString::Printf(TEXT("Failed at %s: %s"), Pose.Name,
            bObjectHeld ? TEXT("timed out") : TEXT("cube was not held"));
        ScriptedPhase = -1;
        UE_LOG(LogTemp, Warning, TEXT("BAS scripted test: %s"), *ScriptedTestStatus);
        return;
    }

    TArray<float> Targets;
    Targets.Append(Pose.Joints, UE_ARRAY_COUNT(Pose.Joints));
    if (!PandaRobot->CommandJointTargets(Targets, Pose.GripperOpenness))
    {
        ScriptedTestStatus = FString::Printf(TEXT("Failed at %s: Panda rejected target"), Pose.Name);
        ScriptedPhase = -1;
        UE_LOG(LogTemp, Warning, TEXT("BAS scripted test: %s"), *ScriptedTestStatus);
        return;
    }
    bool bPoseReached = PandaRobot->ArmJointsRad.Num() == UE_ARRAY_COUNT(Pose.Joints);
    for (int32 Joint = 0; bPoseReached && Joint < UE_ARRAY_COUNT(Pose.Joints); ++Joint)
    {
        bPoseReached &= FMath::IsNearlyEqual(PandaRobot->ArmJointsRad[Joint], Pose.Joints[Joint], 0.005f);
    }
    bPoseReached &= FMath::IsNearlyEqual(PandaRobot->FingerOpeningCm,
        FMath::Max(4.0f * Pose.GripperOpenness, PandaRobot->GetFingerStopCm()), 0.05f);
    if (ScriptedPhase == 2)
    {
        bPoseReached &= bObjectHeld;
    }
    if (ScriptedPhase == UE_ARRAY_COUNT(ScriptedPoses) - 1)
    {
        bPoseReached &= !bObjectHeld;
    }
    if (!bPoseReached)
    {
        return;
    }

    ++ScriptedPhase;
    ScriptedPhaseElapsed = 0.0f;
    if (ScriptedPhase == UE_ARRAY_COUNT(ScriptedPoses))
    {
        ScriptedTestStatus = IsCubePlaced() ? TEXT("Complete: cube placed on yellow marker") :
            TEXT("Failed: cube released away from yellow marker");
        ScriptedPhase = -1;
    }
    else
    {
        ScriptedTestStatus = FString::Printf(TEXT("Running: %s"), ScriptedPoses[ScriptedPhase].Name);
    }
    UE_LOG(LogTemp, Display, TEXT("BAS scripted test: %s"), *ScriptedTestStatus);
}

void ABASWorkcellActor::Grasp(UStaticMeshComponent* Part)
{
    USceneComponent* Frame = PandaRobot ? PandaRobot->GetGraspFrame() : nullptr;
    if (!Frame || HeldPart)
    {
        return;
    }

    Part->SetSimulatePhysics(false);
    Part->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    if (Part->AttachToComponent(Frame, FAttachmentTransformRules::KeepWorldTransform))
    {
        if (Part == PickCube)
        {
            Part->SetWorldLocation(Frame->GetComponentLocation());
        }
        HeldPart = Part;
        bObjectHeld = true;
        // Fingers stop on the object, so openness reads as in MuJoCo (0.62 for the 5 cm cube).
        PandaRobot->SetFingerStopCm(FingerStopFor(Part));
    }
    else
    {
        Part->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Part->SetSimulatePhysics(true);
        Part->SetEnableGravity(false);
    }
}

void ABASWorkcellActor::Release()
{
    if (!HeldPart)
    {
        return;
    }
    UStaticMeshComponent* Part = HeldPart;
    Part->DetachFromComponent(FDetachmentTransformRules::KeepWorldTransform);
    Part->AttachToComponent(WorkcellRoot, FAttachmentTransformRules::KeepWorldTransform);
    Part->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    Part->SetSimulatePhysics(true);
    Part->SetEnableGravity(false);
    Part->SetPhysicsLinearVelocity(FVector::ZeroVector);
    Part->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
    HeldPart = nullptr;
    bObjectHeld = false;
    if (PandaRobot)
    {
        PandaRobot->SetFingerStopCm(0.0f);
    }
}

void ABASWorkcellActor::ResetMovable(UStaticMeshComponent* Part, const FVector& Metres)
{
    const bool bActive = ActiveGraspables().Contains(Part) && (HasActorBegunPlay() || IsActorBeginningPlay());
    Part->SetSimulatePhysics(false);
    Part->SetRelativeLocationAndRotation(MujocoToUnrealCm(Metres), FRotator::ZeroRotator);
    if (bActive)
    {
        Part->SetSimulatePhysics(true);
        Part->SetEnableGravity(false);
        Part->SetPhysicsLinearVelocity(FVector::ZeroVector);
        Part->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
    }
}

void ABASWorkcellActor::ApplyTaskVisibility(bool bPlaying)
{
    const TArray<UStaticMeshComponent*> Collidable = {
        PickCube.Get(), Canister.Get(), Dock.Get(), ToolCase.Get(), TrayFloor.Get(), TrayWalls[0].Get(),
        TrayWalls[1].Get(), TrayWalls[2].Get(), TrayWalls[3].Get(), RedSample.Get(), BlueSample.Get(), Console.Get()};
    for (EBASTask Each : {EBASTask::CubeToPad, EBASTask::CanisterToDock, EBASTask::ToolCaseToTray,
             EBASTask::SortSamples, EBASTask::PressButton})
    {
        const bool bActive = Each == Task;
        for (UStaticMeshComponent* Part : TaskParts(Each))
        {
            Part->SetVisibility(bActive);
            Part->SetCollisionEnabled(bActive && Collidable.Contains(Part) ?
                ECollisionEnabled::QueryAndPhysics : ECollisionEnabled::NoCollision);
            if (!bActive && bPlaying)
            {
                Part->SetSimulatePhysics(false);
            }
        }
    }
}

void ABASWorkcellActor::PlaceTaskProps()
{
    // Task 1
    TargetMarker->SetRelativeLocation(At(MarkerXY, 0.3765));
    ResetMovable(PickCube, FVector(CubeStartXY.X, CubeStartXY.Y, 0.40));

    // Task 2: canister upright on the bench, dock disc with a green centre.
    Dock->SetRelativeLocation(At(MarkerXY, BenchZ + DockH / 2));
    DockCentre->SetRelativeLocation(At(MarkerXY, BenchZ + DockH + 0.0005));
    ResetMovable(Canister, FVector(CubeStartXY.X, CubeStartXY.Y, BenchZ + CanisterH / 2));

    // Task 3: open tray, 4 walls around a thin floor.
    TrayFloor->SetRelativeLocation(At(MarkerXY, BenchZ + TrayFloorT / 2));
    const FVector2D WallOffsets[] = {
        FVector2D(TrayHalf.X - TrayWall / 2, 0), FVector2D(-TrayHalf.X + TrayWall / 2, 0),
        FVector2D(0, TrayHalf.Y - TrayWall / 2), FVector2D(0, -TrayHalf.Y + TrayWall / 2)};
    for (int32 Index = 0; Index < 4; ++Index)
    {
        TrayWalls[Index]->SetRelativeLocation(At(MarkerXY + WallOffsets[Index], BenchZ + TrayWallH / 2));
    }
    ResetMovable(ToolCase, FVector(CubeStartXY.X, CubeStartXY.Y, BenchZ + CaseHalf.Z));

    // Task 4: red sample/pad at CubeStartXY/MarkerXY, blue at the "Other" positions.
    RedPad->SetRelativeLocation(At(MarkerXY, 0.3765));
    BluePad->SetRelativeLocation(At(OtherTargetXY, 0.3765));
    ResetMovable(RedSample, FVector(CubeStartXY.X, CubeStartXY.Y, BenchZ + SampleH / 2));
    ResetMovable(BlueSample, FVector(OtherObjectXY.X, OtherObjectXY.Y, BenchZ + SampleH / 2));
    OtherObjectStartXY = TargetVariant == 0 ? OtherObjectXY : CubeStartXY;

    // Task 5: console at MarkerXY with two buttons and their indicator lights.
    Console->SetRelativeLocation(At(MarkerXY, BenchZ + ConsoleHalf.Z));
    const double GreenY = bSwapButtons ? -ButtonSideY : ButtonSideY;
    GreenLight->SetRelativeLocation(At(MarkerXY + FVector2D(LightFrontX, GreenY), BenchZ + 2 * ConsoleHalf.Z + 0.002));
    RedLight->SetRelativeLocation(At(MarkerXY + FVector2D(LightFrontX, -GreenY), BenchZ + 2 * ConsoleHalf.Z + 0.002));
    bGreenPressed = bRedPressed = false;
    SetButtonPressed(GreenButton, GreenLight, false, 0.0f);
    SetButtonPressed(RedButton, RedLight, false, 0.0f);
}

void ABASWorkcellActor::SetButtonPressed(UStaticMeshComponent* Button, UStaticMeshComponent* Light, bool bPressed, float DepthCm)
{
    const bool bGreen = Button == GreenButton;
    const double Y = (bGreen != bSwapButtons) ? ButtonSideY : -ButtonSideY;
    const FVector Rest = At(MarkerXY + FVector2D(ButtonBackX, Y), ButtonTop - ButtonH / 2);
    Button->SetRelativeLocation(Rest - FVector(0, 0, DepthCm));
    SetColour(Light, bGreen ? (bPressed ? GreenLit : GreenDim) : (bPressed ? RedLit : RedDim));
}

void ABASWorkcellActor::UpdateButtons()
{
    // Kinematic press: a fingertip inside a button's rim pushes it down; past 6 mm it latches and lights up.
    const FVector Tip = WorkcellRoot->GetComponentTransform().InverseTransformPosition(
        PandaRobot->GetGraspCenterWorld()) - FVector(0, 0, FingertipBelowGraspCm);
    const auto Depth = [this, &Tip](bool bGreen)
    {
        const double Y = (bGreen != bSwapButtons) ? ButtonSideY : -ButtonSideY;
        const FVector Top = At(MarkerXY + FVector2D(ButtonBackX, Y), ButtonTop);
        if (FVector2D(Tip.X - Top.X, Tip.Y - Top.Y).Size() > ButtonCatchCm)
        {
            return 0.0f;
        }
        return FMath::Clamp(static_cast<float>(Top.Z - Tip.Z), 0.0f, ButtonTravelCm);
    };
    const float GreenDepth = Depth(true), RedDepth = Depth(false);
    bGreenPressed |= GreenDepth >= ButtonLatchCm;
    bRedPressed |= RedDepth >= ButtonLatchCm;
    SetButtonPressed(GreenButton, GreenLight, bGreenPressed, GreenDepth);
    SetButtonPressed(RedButton, RedLight, bRedPressed, RedDepth);
}

void ABASWorkcellActor::ResetTask()
{
    if (HeldPart)
    {
        Release();
    }
    ApplyTaskVisibility(HasActorBegunPlay() || IsActorBeginningPlay());
    PlaceTaskProps();
}

void ABASWorkcellActor::RandomizeLayout(FRandomStream& Stream)
{
    const auto In = [&Stream](double XLo, double XHi, double YMax)
    {
        return FVector2D(Stream.FRandRange(XLo, XHi), Stream.FRandRange(-YMax, YMax));
    };
    switch (Task)
    {
    case EBASTask::CubeToPad:
        // Ranges from README / closed_loop_panda: cube x 0.40-0.49, y +-0.10; marker x 0.57-0.66, y +-0.20.
        do
        {
            CubeStartXY = FVector2D(Stream.FRandRange(0.40, 0.49), Stream.FRandRange(-0.10, 0.10));
            MarkerXY = FVector2D(Stream.FRandRange(0.57, 0.66), Stream.FRandRange(-0.20, 0.20));
        } while (FVector2D::Distance(CubeStartXY, MarkerXY) < 0.13);
        break;
    case EBASTask::CanisterToDock:
        do
        {
            CubeStartXY = In(0.40, 0.49, 0.10);
            MarkerXY = In(0.57, 0.66, 0.18);
        } while (FVector2D::Distance(CubeStartXY, MarkerXY) < 0.13);
        break;
    case EBASTask::ToolCaseToTray:
        do
        {
            CubeStartXY = In(0.40, 0.48, 0.10);
            MarkerXY = In(0.58, 0.63, 0.14);
        } while (FVector2D::Distance(CubeStartXY, MarkerXY) < 0.15);
        break;
    case EBASTask::SortSamples:
    {
        const auto TooClose = [this]()
        {
            for (const FVector2D& Sample : {CubeStartXY, OtherObjectXY})
            {
                for (const FVector2D& Pad : {MarkerXY, OtherTargetXY})
                {
                    if (FVector2D::Distance(Sample, Pad) < 0.12) return true;
                }
            }
            return FVector2D::Distance(CubeStartXY, OtherObjectXY) < 0.09 || FMath::Abs(MarkerXY.Y - OtherTargetXY.Y) < 0.12;
        };
        do
        {
            CubeStartXY = In(0.40, 0.49, 0.12);
            OtherObjectXY = In(0.40, 0.49, 0.12);
            MarkerXY = In(0.58, 0.65, 0.17);
            OtherTargetXY = In(0.58, 0.65, 0.17);
        } while (TooClose());
        TargetVariant = Stream.RandRange(0, 1);
        break;
    }
    case EBASTask::PressButton:
        MarkerXY = In(0.56, 0.62, 0.12);
        bSwapButtons = Stream.FRand() < 0.5f;
        TargetVariant = Stream.RandRange(0, 1);
        break;
    }
}

FString ABASWorkcellActor::GetTaskInstruction() const
{
    switch (Task)
    {
    case EBASTask::CubeToPad: return TEXT("Pick up the red cube and place it on the yellow marker.");
    case EBASTask::CanisterToDock: return TEXT("Move the sample canister to the analysis dock.");
    case EBASTask::ToolCaseToTray: return TEXT("Put the tool case in the storage tray.");
    case EBASTask::SortSamples:
        return TargetVariant == 0 ? TEXT("Put the red sample on the red pad.") : TEXT("Put the blue sample on the blue pad.");
    case EBASTask::PressButton:
        return TargetVariant == 0 ? TEXT("Press the green button on the control panel.") :
            TEXT("Press the red button on the control panel.");
    }
    return FString();
}

FVector ABASWorkcellActor::ToMujoco(const UStaticMeshComponent* Part) const
{
    const FVector Cm = WorkcellRoot->GetComponentTransform().InverseTransformPosition(Part->GetComponentLocation());
    return FVector(Cm.X, -Cm.Y, Cm.Z) / 100.0;
}

FVector ABASWorkcellActor::GetActiveObjectMujoco() const
{
    switch (Task)
    {
    case EBASTask::CanisterToDock: return ToMujoco(Canister);
    case EBASTask::ToolCaseToTray: return ToMujoco(ToolCase);
    case EBASTask::SortSamples: return ToMujoco(TargetVariant == 0 ? RedSample : BlueSample);
    case EBASTask::PressButton: return GetTargetMujoco();
    default: return ToMujoco(PickCube);
    }
}

FVector ABASWorkcellActor::GetTargetMujoco() const
{
    switch (Task)
    {
    case EBASTask::CanisterToDock: return FVector(MarkerXY.X, MarkerXY.Y, BenchZ + DockH);
    case EBASTask::ToolCaseToTray: return FVector(MarkerXY.X, MarkerXY.Y, BenchZ + TrayFloorT);
    case EBASTask::SortSamples:
    {
        const FVector2D Pad = TargetVariant == 0 ? MarkerXY : OtherTargetXY;
        return FVector(Pad.X, Pad.Y, 0.377);
    }
    case EBASTask::PressButton:
    {
        const bool bGreen = TargetVariant == 0;
        const double Y = (bGreen != bSwapButtons) ? ButtonSideY : -ButtonSideY;
        return FVector(MarkerXY.X + ButtonBackX, MarkerXY.Y + Y, ButtonTop);
    }
    default: return FVector(MarkerXY.X, MarkerXY.Y, 0.377);
    }
}

bool ABASWorkcellActor::IsReleased() const
{
    return !bObjectHeld && PandaRobot && PandaRobot->GetGripperOpenness() > 0.8f;
}

bool ABASWorkcellActor::IsTaskSuccess() const
{
    switch (Task)
    {
    case EBASTask::CubeToPad:
        return IsCubePlaced() && IsReleased();
    case EBASTask::CanisterToDock:
    {
        const FVector P = ToMujoco(Canister);
        return IsReleased() && FVector2D::Distance(FVector2D(P.X, P.Y), MarkerXY) <= 0.025 &&
            FMath::Abs(P.Z - (BenchZ + DockH + CanisterH / 2)) <= 0.02 &&
            Canister->GetUpVector().Z >= FMath::Cos(FMath::DegreesToRadians(15.0f));
    }
    case EBASTask::ToolCaseToTray:
    {
        const FVector P = ToMujoco(ToolCase);
        const double Slack = 0.01;
        return IsReleased() &&
            FMath::Abs(P.X - MarkerXY.X) <= TrayHalf.X - TrayWall - CaseHalf.X + Slack &&
            FMath::Abs(P.Y - MarkerXY.Y) <= TrayHalf.Y - TrayWall - CaseHalf.Y + Slack &&
            P.Z <= BenchZ + TrayFloorT + CaseHalf.Z + 0.02;
    }
    case EBASTask::SortSamples:
    {
        const FVector P = ToMujoco(TargetVariant == 0 ? RedSample : BlueSample);
        const FVector O = ToMujoco(TargetVariant == 0 ? BlueSample : RedSample);
        const FVector2D Pad = TargetVariant == 0 ? MarkerXY : OtherTargetXY;
        return IsReleased() && FVector2D::Distance(FVector2D(P.X, P.Y), Pad) <= 0.035 &&
            FMath::Abs(P.Z - (BenchZ + SampleH / 2)) <= 0.02 &&
            FVector2D::Distance(FVector2D(O.X, O.Y), OtherObjectStartXY) <= 0.03;
    }
    case EBASTask::PressButton:
    {
        // Right button latched, wrong one untouched, fingertips back up at least 2 cm.
        if (!PandaRobot)
        {
            return false;
        }
        const bool bTarget = TargetVariant == 0 ? bGreenPressed : bRedPressed;
        const bool bWrong = TargetVariant == 0 ? bRedPressed : bGreenPressed;
        const double TipZ = WorkcellRoot->GetComponentTransform().InverseTransformPosition(
            PandaRobot->GetGraspCenterWorld()).Z / 100.0 - FingertipBelowGraspCm / 100.0;
        return bTarget && !bWrong && TipZ >= ButtonTop + 0.02;
    }
    }
    return false;
}

bool ABASWorkcellActor::IsCubePlaced() const
{
    const FVector CubeLocation = PickCube->GetComponentLocation();
    const FVector TargetLocation = TargetMarker->GetComponentLocation();
    const FVector Delta = CubeLocation - TargetLocation;
    const float ExpectedCubeZ = BenchTop->GetComponentLocation().Z + 2.5f + 2.5f;
    return !bObjectHeld && FVector2D(Delta.X, Delta.Y).Size() <= 4.0f &&
        FMath::Abs(CubeLocation.Z - ExpectedCubeZ) <= 2.0f;
}

void ABASWorkcellActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    ApplyTaskVisibility(false);
    PlaceTaskProps();

    SetColour(RobotMount, FLinearColor(0.30f, 0.33f, 0.38f));
    SetColour(MountPlate, FLinearColor(0.46f, 0.49f, 0.54f));
    SetColour(BenchRack, FLinearColor(0.46f, 0.49f, 0.54f));
    SetColour(BenchTop, FLinearColor(0.68f, 0.71f, 0.75f));
    SetColour(TargetMarker, FLinearColor(1.0f, 0.85f, 0.0f));
    SetColour(PickCube, FLinearColor(0.9f, 0.12f, 0.08f));

    SetColour(Canister, FLinearColor(0.95f, 0.45f, 0.05f));      // orange sample canister
    SetColour(Dock, FLinearColor(0.12f, 0.13f, 0.15f));          // dark analysis dock
    SetColour(DockCentre, FLinearColor(0.1f, 0.8f, 0.3f));       // green docking spot
    SetColour(ToolCase, FLinearColor(0.08f, 0.18f, 0.55f));      // navy tool case
    SetColour(TrayFloor, FLinearColor(0.35f, 0.45f, 0.40f));     // grey-green storage tray
    for (UStaticMeshComponent* Wall : TrayWalls)
    {
        SetColour(Wall, FLinearColor(0.35f, 0.45f, 0.40f));
    }
    SetColour(RedSample, FLinearColor(0.9f, 0.1f, 0.08f));
    SetColour(BlueSample, FLinearColor(0.1f, 0.3f, 0.95f));
    SetColour(RedPad, FLinearColor(0.75f, 0.12f, 0.1f));
    SetColour(BluePad, FLinearColor(0.12f, 0.25f, 0.8f));
    SetColour(Console, FLinearColor(0.15f, 0.16f, 0.19f));       // dark control console
    SetColour(GreenButton, FLinearColor(0.1f, 0.7f, 0.15f));
    SetColour(RedButton, FLinearColor(0.85f, 0.1f, 0.08f));
    SetButtonPressed(GreenButton, GreenLight, bGreenPressed, 0.0f);
    SetButtonPressed(RedButton, RedLight, bRedPressed, 0.0f);
}
