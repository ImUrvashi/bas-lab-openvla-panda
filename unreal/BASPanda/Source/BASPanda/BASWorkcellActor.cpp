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

    // Prop geometry in MuJoCo metres, relative to each fixture's root on the bench top (z 0.375).
    // Fingers close along x, so every gripped part is 2.5 cm thick along x.
    constexpr double BenchZ = 0.375;
    constexpr float HandleStopCm = 1.25f;

    // Task 2: cabinet body; the drawer slides out of its +y face by up to 11 cm.
    const FVector CabinetHalf(0.08, 0.07, 0.045);
    constexpr double DrawerMaxOpen = 0.11, DrawerFloorTop = 0.016, DrawerInner = 0.056;
    const FVector ToolHalf(0.0125, 0.03, 0.012);

    // Task 3: lever 10.5 cm above the bench, along the pipe (y) when open, across it (x) when closed.
    constexpr double LeverZ = 0.105;

    // Task 4: low scrubber (5 cm, so the camera sees the item row over it) with a 4 cm deep slot;
    // a seated cartridge sticks out 4 cm.
    const FVector ScrubberHalf(0.05, 0.065, 0.025);
    const FVector CartridgeHalf(0.0125, 0.03, 0.04);
    constexpr double ScrubberTop = 2 * 0.025, SlotDepth = 0.04;
    constexpr double SeatedCartridgeZ = ScrubberTop - SlotDepth + 0.04;

    // Task 5: open-top container 10 x 10 x 6 cm; lid hinged on its robot-side (-x) top edge.
    constexpr double ContainerHalf = 0.05, ContainerH = 0.06, ContainerInner = 0.044;
    const FVector LidHinge(-0.05, 0.0, 0.06);
    constexpr double VialR = 0.015, VialH = 0.05;

    // One station bench: every prop has a home spot (MuJoCo metres), all inside reach and the policy camera view.
    enum EProp { PCube, PPad, PTool, PCabinet, PVial, PCartridge, PScrubber, PContainer, PValve, PropCount };
    // Loose items sit in one row at x 0.55, in front of the arm's home pose (which hides x < 0.52 near y = 0).
    const FVector2D HomeXY[PropCount] = {
        {0.555, -0.125}, // red cube
        {0.66, -0.17},  // yellow processing pad (front left)
        {0.55, -0.04},  // tool
        {0.44, -0.27},  // cabinet (back left); drawer opens towards +y
        {0.55, 0.04},   // sample vial
        {0.55, 0.12},   // filter cartridge
        {0.66, 0.02},   // air scrubber (front centre)
        {0.64, 0.21},   // sample container (front right), lid hinged on its robot side
        {0.45, 0.25},   // coolant valve (back right) on a pipe along y
    };

    // MuJoCo metres (y left) -> Unreal centimetres (y right): the one place this flip happens.
    FVector MujocoToUnrealCm(const FVector& Metres)
    {
        return FVector(Metres.X, -Metres.Y, Metres.Z) * 100.0;
    }

    FVector At(const FVector2D& XY, double Z)
    {
        return MujocoToUnrealCm(FVector(XY.X, XY.Y, Z));
    }

    // Engine/BasicShapes/Cube is 100 cm on each side. Sizes are half-extents in metres.
    void ConfigureBox(UStaticMeshComponent* Part, UStaticMesh* CubeMesh,
        const FVector& PositionMetres, const FVector& HalfSizeMetres, bool bCollides)
    {
        Part->SetStaticMesh(CubeMesh);
        Part->SetRelativeLocation(PositionMetres * 100.0);
        Part->SetRelativeScale3D(HalfSizeMetres * 2.0);
        Part->SetCollisionEnabled(bCollides ? ECollisionEnabled::QueryAndPhysics : ECollisionEnabled::NoCollision);
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

    const FLinearColor LitGreen(0.25f, 1.0f, 0.3f), DimGreen(0.03f, 0.12f, 0.05f);
    const FLinearColor LitRed(1.0f, 0.12f, 0.05f), DimAmber(0.35f, 0.22f, 0.02f);
}

ABASWorkcellActor::ABASWorkcellActor()
{
    PrimaryActorTick.bCanEverTick = true;
    WorkcellRoot = CreateDefaultSubobject<USceneComponent>(TEXT("WorkcellRoot"));
    RootComponent = WorkcellRoot;

    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
    UStaticMesh* CylinderMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));

    const auto Root = [this](USceneComponent* Parent, const TCHAR* Name)
    {
        USceneComponent* Part = CreateDefaultSubobject<USceneComponent>(Name);
        Part->SetupAttachment(Parent);
        Part->SetMobility(EComponentMobility::Movable);
        return Part;
    };
    // Visual prop part: position and half-size in metres relative to its parent fixture.
    const auto Shape = [this](USceneComponent* Parent, const TCHAR* Name, UStaticMesh* Mesh, const FVector& Scale,
        const FVector& Pos, const FLinearColor& Colour, const FRotator& Rotation = FRotator::ZeroRotator)
    {
        UStaticMeshComponent* Part = CreateDefaultSubobject<UStaticMeshComponent>(Name);
        Part->SetupAttachment(Parent);
        Part->SetMobility(EComponentMobility::Movable);
        Part->SetStaticMesh(Mesh);
        Part->SetRelativeLocationAndRotation(MujocoToUnrealCm(Pos), Rotation);
        Part->SetRelativeScale3D(Scale);
        Part->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        ColouredParts.Add(Part);
        PartColours.Add(Colour);
        return Part;
    };
    const auto Box = [&](USceneComponent* Parent, const TCHAR* Name, const FVector& Half, const FVector& Pos,
        const FLinearColor& Colour)
    {
        return Shape(Parent, Name, CubeMesh, Half * 2.0, Pos, Colour);
    };
    const auto Cyl = [&](USceneComponent* Parent, const TCHAR* Name, double R, double H, const FVector& Pos,
        const FLinearColor& Colour, const FRotator& Rotation = FRotator::ZeroRotator)
    {
        return Shape(Parent, Name, CylinderMesh, FVector(2 * R, 2 * R, H), Pos, Colour, Rotation);
    };

    RobotMount = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("RobotMount"));
    RobotMount->SetupAttachment(WorkcellRoot);
    ConfigureBox(RobotMount, CubeMesh, FVector(-0.05, 0, -0.515), FVector(0.25, 0.40, 0.515), true);
    MountPlate = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MountPlate"));
    MountPlate->SetupAttachment(WorkcellRoot);
    ConfigureBox(MountPlate, CubeMesh, FVector(0, 0, -0.015), FVector(0.20, 0.20, 0.015), false);
    BenchRack = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("BenchRack"));
    BenchRack->SetupAttachment(WorkcellRoot);
    ConfigureBox(BenchRack, CubeMesh, FVector(0.55, 0, -0.3525), FVector(0.35, 0.40, 0.6775), true);
    BenchTop = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("BenchTop"));
    BenchTop->SetupAttachment(WorkcellRoot);
    ConfigureBox(BenchTop, CubeMesh, FVector(0.55, 0, 0.35), FVector(0.35, 0.40, 0.025), true);

    // Task 1
    TargetMarker = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("TargetMarker"));
    TargetMarker->SetupAttachment(WorkcellRoot);
    ConfigureBox(TargetMarker, CubeMesh, FVector(0.60, 0, 0.3765), FVector(0.05, 0.05, 0.0005), false);
    PickCube = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PickCube"));
    PickCube->SetupAttachment(WorkcellRoot);
    PickCube->SetMobility(EComponentMobility::Movable);
    ConfigureBox(PickCube, CubeMesh, FVector(0.45, 0, 0.40), FVector(0.025, 0.025, 0.025), true);

    const FLinearColor Panel(0.55f, 0.60f, 0.68f), Dark(0.12f, 0.13f, 0.15f), Accent(0.95f, 0.45f, 0.05f);

    // Task 2: stowage cabinet, drawer (floor, walls, front, knob), loose tool.
    CabinetRoot = Root(WorkcellRoot, TEXT("CabinetRoot"));
    Box(CabinetRoot, TEXT("CabinetBody"), CabinetHalf, FVector(0, 0, CabinetHalf.Z), Panel);
    DrawerRoot = Root(CabinetRoot, TEXT("DrawerRoot"));
    Box(DrawerRoot, TEXT("DrawerFloor"), FVector(0.062, 0.062, 0.003), FVector(0, 0, 0.013), Dark);
    Box(DrawerRoot, TEXT("DrawerSideL"), FVector(0.003, 0.062, 0.02), FVector(0.059, 0, 0.036), Panel);
    Box(DrawerRoot, TEXT("DrawerSideR"), FVector(0.003, 0.062, 0.02), FVector(-0.059, 0, 0.036), Panel);
    Box(DrawerRoot, TEXT("DrawerBack"), FVector(0.062, 0.003, 0.02), FVector(0, -0.059, 0.036), Panel);
    Box(DrawerRoot, TEXT("DrawerFront"), FVector(0.075, 0.004, 0.042), FVector(0, 0.074, 0.045), Accent);
    DrawerKnob = Box(DrawerRoot, TEXT("DrawerKnob"), FVector(0.0125, 0.012, 0.012), FVector(0, 0.074, 0.099), Dark);
    Tool = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Tool"));
    Tool->SetupAttachment(WorkcellRoot);
    Tool->SetMobility(EComponentMobility::Movable);
    ConfigureBox(Tool, CubeMesh, FVector::ZeroVector, ToolHalf, false);

    // Task 3: coolant pipe on two supports, valve body, stem, quarter-turn lever, status light.
    ValveRoot = Root(WorkcellRoot, TEXT("ValveRoot"));
    Cyl(ValveRoot, TEXT("CoolantPipe"), 0.015, 0.24, FVector(0, 0, 0.05), FLinearColor(0.7f, 0.72f, 0.75f),
        FRotator(0, 0, 90));
    Box(ValveRoot, TEXT("PipeSupportL"), FVector(0.012, 0.012, 0.025), FVector(0, 0.09, 0.025), Dark);
    Box(ValveRoot, TEXT("PipeSupportR"), FVector(0.012, 0.012, 0.025), FVector(0, -0.09, 0.025), Dark);
    Box(ValveRoot, TEXT("ValveBody"), FVector(0.03, 0.03, 0.03), FVector(0, 0, 0.05), FLinearColor(0.2f, 0.35f, 0.6f));
    Cyl(ValveRoot, TEXT("ValveStem"), 0.006, 0.02, FVector(0, 0, 0.09), Dark);
    ValveLeverRoot = Root(ValveRoot, TEXT("ValveLeverRoot"));
    ValveLeverRoot->SetRelativeLocation(MujocoToUnrealCm(FVector(0, 0, LeverZ)));
    ValveLever = Box(ValveLeverRoot, TEXT("ValveLever"), FVector(0.0125, 0.05, 0.0075), FVector::ZeroVector,
        FLinearColor(0.85f, 0.1f, 0.08f));
    Cyl(ValveLeverRoot, TEXT("ValveHub"), 0.012, 0.02, FVector::ZeroVector, Dark);
    ValveLight = Cyl(ValveRoot, TEXT("ValveLight"), 0.008, 0.004, FVector(0.022, 0.022, 0.082), LitRed);

    // Task 4: air scrubber with a yellow-rimmed slot and seated light, loose filter cartridge.
    ScrubberRoot = Root(WorkcellRoot, TEXT("ScrubberRoot"));
    Box(ScrubberRoot, TEXT("ScrubberBody"), ScrubberHalf, FVector(0, 0, ScrubberHalf.Z), FLinearColor(0.75f, 0.8f, 0.9f));
    Box(ScrubberRoot, TEXT("ScrubberSlotRim"), FVector(0.022, 0.04, 0.0008), FVector(0, 0, ScrubberTop + 0.0005),
        FLinearColor(1.0f, 0.8f, 0.0f));
    Box(ScrubberRoot, TEXT("ScrubberSlot"), FVector(0.016, 0.034, 0.001), FVector(0, 0, ScrubberTop + 0.001), FLinearColor(0.02f, 0.02f, 0.02f));
    ScrubberLight = Cyl(ScrubberRoot, TEXT("ScrubberLight"), 0.008, 0.004, FVector(0.035, 0.05, ScrubberTop + 0.0015), DimGreen);
    Cartridge = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("FilterCartridge"));
    Cartridge->SetupAttachment(WorkcellRoot);
    Cartridge->SetMobility(EComponentMobility::Movable);
    ConfigureBox(Cartridge, CubeMesh, FVector::ZeroVector, CartridgeHalf, false);

    // Task 5: open container (floor + 4 walls), hinged lid with tab, latch light, loose vial.
    ContainerRoot = Root(WorkcellRoot, TEXT("ContainerRoot"));
    const FLinearColor Shell(0.55f, 0.45f, 0.75f);
    Box(ContainerRoot, TEXT("ContainerFloor"), FVector(ContainerHalf, ContainerHalf, 0.003), FVector(0, 0, 0.003), Shell);
    Box(ContainerRoot, TEXT("ContainerWallF"), FVector(0.003, ContainerHalf, ContainerH / 2), FVector(0.047, 0, ContainerH / 2), Shell);
    Box(ContainerRoot, TEXT("ContainerWallB"), FVector(0.003, ContainerHalf, ContainerH / 2), FVector(-0.047, 0, ContainerH / 2), Shell);
    Box(ContainerRoot, TEXT("ContainerWallL"), FVector(ContainerHalf, 0.003, ContainerH / 2), FVector(0, 0.047, ContainerH / 2), Shell);
    Box(ContainerRoot, TEXT("ContainerWallR"), FVector(ContainerHalf, 0.003, ContainerH / 2), FVector(0, -0.047, ContainerH / 2), Shell);
    LidRoot = Root(ContainerRoot, TEXT("LidHinge"));
    LidRoot->SetRelativeLocation(MujocoToUnrealCm(LidHinge));
    Box(LidRoot, TEXT("Lid"), FVector(ContainerHalf, ContainerHalf, 0.003), FVector(0.05, 0, 0.003), FLinearColor(0.75f, 0.68f, 0.9f));
    LidTab = Box(LidRoot, TEXT("LidTab"), FVector(0.0125, 0.02, 0.01), FVector(0.09, 0, 0.013), Dark);
    ContainerLight = Cyl(ContainerRoot, TEXT("LatchLight"), 0.007, 0.003, FVector(0.065, 0, 0.0015), DimAmber);
    Vial = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SampleVial"));
    Vial->SetupAttachment(WorkcellRoot);
    Vial->SetMobility(EComponentMobility::Movable);
    Vial->SetStaticMesh(CylinderMesh);
    Vial->SetRelativeScale3D(FVector(2 * VialR, 2 * VialR, VialH));

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

UStaticMeshComponent* ABASWorkcellActor::ActiveObject() const
{
    switch (Task)
    {
    case EBASTask::CubeToPad: return PickCube;
    case EBASTask::StowInDrawer: return Tool;
    case EBASTask::InsertFilter: return Cartridge;
    case EBASTask::SealContainer: return Vial;
    default: return nullptr;
    }
}


UStaticMeshComponent* ABASWorkcellActor::ActiveHandle() const
{
    switch (Task)
    {
    case EBASTask::StowInDrawer: return DrawerKnob;
    case EBASTask::CloseValve: return ValveLever;
    case EBASTask::SealContainer: return LidTab;
    default: return nullptr;
    }
}

bool ABASWorkcellActor::CanGrasp(const UStaticMeshComponent* Part) const
{
    // Stowed items stay put once their drawer or lid is (nearly) shut; a seated cartridge is locked.
    if (Part == Cartridge) return !bCartridgeSeated;
    if (Part == Vial) return !(bVialStowed && LidDeg < 30.0f);
    if (Part == Tool) return !(bToolStowed && DrawerOpenM < 0.05f);
    return true;
}

FVector2D ABASWorkcellActor::Home(int32 Prop) const
{
    return HomeXY[Prop] + (Jitter.IsValidIndex(Prop) ? Jitter[Prop] : FVector2D::ZeroVector);
}

float ABASWorkcellActor::FingerStopFor(const UStaticMeshComponent* Part) const
{
    // Each finger stops at half the gripped part's thickness along x (as MuJoCo contact would).
    if (Part == PickCube) return 2.5f;
    if (Part == Vial) return 100.0f * VialR;
    return HandleStopCm;
}

void ABASWorkcellActor::BeginPlay()
{
    Super::BeginPlay();

    // Free bodies in zero gravity, as in MuJoCo.
    const TPair<UStaticMeshComponent*, float> Masses[] = {
        {PickCube.Get(), 0.08f}, {Tool.Get(), 0.05f}, {Cartridge.Get(), 0.06f}, {Vial.Get(), 0.03f}};
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

    const float Opening = PandaRobot->FingerOpeningCm;
    if (HeldPart)
    {
        if (Opening >= FingerStopFor(HeldPart) + 0.7f)
        {
            Release();
        }
    }
    else if (HeldHandle != EBASHandle::None)
    {
        if (Opening >= HandleStopCm + 0.7f)
        {
            ReleaseHandle();
        }
        else
        {
            DriveHandle();
        }
    }
    else
    {
        TryGrasp();
    }
    UpdateMechanisms();
    if (ScriptedPhase >= 0)
    {
        AdvanceScriptedTest(DeltaSeconds);
    }
}

void ABASWorkcellActor::TryGrasp()
{
    // Grasp proxy: closing fingers take the nearest loose object (within 6 cm) or handle (within 5 cm),
    // whatever the task, so a wrong choice by the policy really happens.
    const FVector GraspPoint = PandaRobot->GetGraspCenterWorld();
    const float Opening = PandaRobot->FingerOpeningCm;
    UStaticMeshComponent* BestObject = nullptr;
    float BestDist = 6.0f;
    for (UStaticMeshComponent* Part : {PickCube.Get(), Tool.Get(), Cartridge.Get(), Vial.Get()})
    {
        const float Dist = FVector::Dist(Part->GetComponentLocation(), GraspPoint);
        if (CanGrasp(Part) && Opening <= FingerStopFor(Part) + 0.2f && Dist <= BestDist)
        {
            BestObject = Part;
            BestDist = Dist;
        }
    }
    EBASHandle BestHandle = EBASHandle::None;
    float HandleDist = FMath::Min(5.0f, BestDist);
    const TPair<UStaticMeshComponent*, EBASHandle> Handles[] = {
        {DrawerKnob.Get(), EBASHandle::Drawer}, {ValveLever.Get(), EBASHandle::Valve}, {LidTab.Get(), EBASHandle::Lid}};
    for (const TPair<UStaticMeshComponent*, EBASHandle>& Handle : Handles)
    {
        const float Dist = FVector::Dist(Handle.Key->GetComponentLocation(), GraspPoint);
        if (Opening <= HandleStopCm + 0.2f && Dist < HandleDist)
        {
            BestHandle = Handle.Value;
            HandleDist = Dist;
        }
    }
    if (BestHandle != EBASHandle::None)
    {
        GripHandle(BestHandle);
    }
    else if (BestObject)
    {
        Grasp(BestObject);
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

FVector ABASWorkcellActor::ToMujoco(const FVector& World) const
{
    const FVector Cm = WorkcellRoot->GetComponentTransform().InverseTransformPosition(World);
    return FVector(Cm.X, -Cm.Y, Cm.Z) / 100.0;
}

FVector ABASWorkcellActor::LocalMujoco(const USceneComponent* Frame, const FVector& World) const
{
    const FVector Cm = Frame->GetComponentTransform().InverseTransformPosition(World);
    return FVector(Cm.X, -Cm.Y, Cm.Z) / 100.0;
}

void ABASWorkcellActor::SetFree(UStaticMeshComponent* Part, bool bFree)
{
    Part->SetSimulatePhysics(false);
    Part->SetCollisionEnabled(bFree ? ECollisionEnabled::QueryAndPhysics : ECollisionEnabled::NoCollision);
    if (bFree)
    {
        Part->SetSimulatePhysics(true);
        Part->SetEnableGravity(false);
        Part->SetPhysicsLinearVelocity(FVector::ZeroVector);
        Part->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
    }
}

void ABASWorkcellActor::SnapInto(UStaticMeshComponent* Part, USceneComponent* Frame, const FVector& LocalMetres)
{
    SetFree(Part, false);
    Part->AttachToComponent(Frame, FAttachmentTransformRules::KeepWorldTransform);
    Part->SetRelativeLocationAndRotation(MujocoToUnrealCm(LocalMetres), FRotator::ZeroRotator);
}

void ABASWorkcellActor::Grasp(UStaticMeshComponent* Part)
{
    USceneComponent* Frame = PandaRobot ? PandaRobot->GetGraspFrame() : nullptr;
    if (!Frame || HeldPart)
    {
        return;
    }
    SetFree(Part, false);
    if (Part->AttachToComponent(Frame, FAttachmentTransformRules::KeepWorldTransform))
    {
        if (Part == PickCube)
        {
            Part->SetWorldLocation(Frame->GetComponentLocation());
        }
        HeldPart = Part;
        bObjectHeld = true;
        bToolStowed &= Part != Tool;
        bVialStowed &= Part != Vial;
        // Fingers stop on the object, so openness reads as in MuJoCo (0.62 for the 5 cm cube).
        PandaRobot->SetFingerStopCm(FingerStopFor(Part));
    }
    else
    {
        SetFree(Part, true);
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
    HeldPart = nullptr;
    bObjectHeld = false;
    if (PandaRobot)
    {
        PandaRobot->SetFingerStopCm(0.0f);
    }
    OnObjectReleased(Part);
}

void ABASWorkcellActor::OnObjectReleased(UStaticMeshComponent* Part)
{
    // Released inside its receptacle: settle onto the floor / into the slot and move with the fixture.
    const FVector World = Part->GetComponentLocation();
    if (Part == Tool)
    {
        const FVector L = LocalMujoco(DrawerRoot, World);
        const double MaxX = DrawerInner - ToolHalf.X + 0.01, MaxY = DrawerInner - ToolHalf.Y + 0.01;
        const bool bOutsideCabinet = L.Y + DrawerOpenM - ToolHalf.Y >= CabinetHalf.Y - 0.01;
        if (FMath::Abs(L.X) <= MaxX && FMath::Abs(L.Y) <= MaxY && bOutsideCabinet &&
            L.Z <= DrawerFloorTop + ToolHalf.Z + 0.03)
        {
            SnapInto(Tool, DrawerRoot, FVector(FMath::Clamp(L.X, -MaxX + 0.01, MaxX - 0.01),
                FMath::Clamp(L.Y, -MaxY + 0.01, MaxY - 0.01), DrawerFloorTop + ToolHalf.Z));
            bToolStowed = true;
            return;
        }
    }
    else if (Part == Cartridge)
    {
        const FVector L = LocalMujoco(ScrubberRoot, World);
        if (FVector2D(L.X, L.Y).Size() <= 0.012 && L.Z <= 2 * ScrubberHalf.Z + CartridgeHalf.Z - 0.02)
        {
            SnapInto(Cartridge, ScrubberRoot, FVector(0, 0, SeatedCartridgeZ));
            bCartridgeSeated = true;
            return;
        }
    }
    else if (Part == Vial)
    {
        const FVector L = LocalMujoco(ContainerRoot, World);
        const double Max = ContainerInner - VialR + 0.008;
        if (FMath::Abs(L.X) <= Max && FMath::Abs(L.Y) <= Max && L.Z <= ContainerH + VialH / 2 + 0.02)
        {
            SnapInto(Vial, ContainerRoot, FVector(FMath::Clamp(L.X, -Max + 0.008, Max - 0.008),
                FMath::Clamp(L.Y, -Max + 0.008, Max - 0.008), 0.006 + VialH / 2));
            bVialStowed = true;
            return;
        }
    }
    SetFree(Part, true);
}

void ABASWorkcellActor::GripHandle(EBASHandle Handle)
{
    USceneComponent* Frame = PandaRobot->GetGraspFrame();
    HeldHandle = Handle;
    GripStartYaw = Frame->GetComponentRotation().Yaw;
    if (Handle == EBASHandle::Lid)
    {
        GripStartMujoco = LocalMujoco(ContainerRoot, Frame->GetComponentLocation()) - LidHinge;
        GripStartValue = LidDeg;
        bLidLatched = false;
    }
    else
    {
        GripStartMujoco = ToMujoco(Frame->GetComponentLocation());
        GripStartValue = Handle == EBASHandle::Drawer ? DrawerOpenM : ValveDeg;
    }
    PandaRobot->SetFingerStopCm(HandleStopCm);
}

void ABASWorkcellActor::ReleaseHandle()
{
    // Snap into the end states: drawer shut, valve fully closed, lid latched.
    switch (HeldHandle)
    {
    case EBASHandle::Drawer:
        DrawerOpenM = DrawerOpenM <= 0.008f ? 0.0f : DrawerOpenM;
        break;
    case EBASHandle::Valve:
        ValveDeg = FMath::Abs(ValveDeg) >= 85.0f ? FMath::Sign(ValveDeg) * 90.0f : ValveDeg;
        break;
    case EBASHandle::Lid:
        bLidLatched = LidDeg <= 5.0f;
        LidDeg = bLidLatched ? 0.0f : LidDeg;
        break;
    default:
        break;
    }
    HeldHandle = EBASHandle::None;
    PandaRobot->SetFingerStopCm(0.0f);
}

void ABASWorkcellActor::DriveHandle()
{
    // The gripped mechanism follows the hand along its one degree of freedom; pulling off-track slips.
    USceneComponent* Frame = PandaRobot->GetGraspFrame();
    if (HeldHandle == EBASHandle::Lid)
    {
        const FVector H = LocalMujoco(ContainerRoot, Frame->GetComponentLocation()) - LidHinge;
        const double Radius = FVector2D(H.X, H.Z).Size(), StartRadius = FVector2D(GripStartMujoco.X, GripStartMujoco.Z).Size();
        if (FMath::Abs(H.Y - GripStartMujoco.Y) > 0.04 || FMath::Abs(Radius - StartRadius) > 0.04)
        {
            return ReleaseHandle();
        }
        const double Swing = FMath::RadiansToDegrees(FMath::Atan2(H.Z, H.X) - FMath::Atan2(GripStartMujoco.Z, GripStartMujoco.X));
        LidDeg = FMath::Clamp(static_cast<float>(GripStartValue + Swing), 0.0f, 90.0f);
        return;
    }
    const FVector Move = ToMujoco(Frame->GetComponentLocation()) - GripStartMujoco;
    if (HeldHandle == EBASHandle::Drawer)
    {
        if (FMath::Abs(Move.X) > 0.04 || FMath::Abs(Move.Z) > 0.04)
        {
            return ReleaseHandle();
        }
        DrawerOpenM = FMath::Clamp(static_cast<float>(GripStartValue + Move.Y), 0.0f, static_cast<float>(DrawerMaxOpen));
    }
    else if (HeldHandle == EBASHandle::Valve)
    {
        if (FVector2D(Move.X, Move.Y).Size() > 0.03)
        {
            return ReleaseHandle();
        }
        const float Turn = FMath::FindDeltaAngleDegrees(GripStartYaw, Frame->GetComponentRotation().Yaw);
        ValveDeg = FMath::Clamp(GripStartValue + Turn, -90.0f, 90.0f);
    }
}

void ABASWorkcellActor::UpdateMechanisms()
{
    DrawerRoot->SetRelativeLocation(MujocoToUnrealCm(FVector(0, DrawerOpenM, 0)));
    ValveLeverRoot->SetRelativeRotation(FRotator(0, ValveDeg, 0));
    LidRoot->SetRelativeRotation(FRotator(LidDeg, 0, 0));
    SetColour(ValveLight, FMath::Abs(ValveDeg) >= 89.0f ? LitGreen : LitRed);
    SetColour(ScrubberLight, bCartridgeSeated ? LitGreen : DimGreen);
    SetColour(ContainerLight, bLidLatched ? LitGreen : DimAmber);
}


void ABASWorkcellActor::PlaceTaskProps()
{
    const bool bPlaying = HasActorBegunPlay() || IsActorBeginningPlay();
    const auto ResetObject = [this, bPlaying](UStaticMeshComponent* Part, int32 Prop, double HalfHeight)
    {
        if (Part->GetAttachParent() != WorkcellRoot)
        {
            Part->AttachToComponent(WorkcellRoot, FAttachmentTransformRules::KeepWorldTransform);
        }
        SetFree(Part, false);
        Part->SetVisibility(true);
        Part->SetRelativeLocationAndRotation(At(Home(Prop), BenchZ + HalfHeight), FRotator::ZeroRotator);
        if (bPlaying)
        {
            SetFree(Part, true);
        }
    };
    ResetObject(PickCube, PCube, 0.025);
    ResetObject(Tool, PTool, ToolHalf.Z);
    ResetObject(Cartridge, PCartridge, CartridgeHalf.Z);
    ResetObject(Vial, PVial, VialH / 2);

    TargetMarker->SetVisibility(true);
    TargetMarker->SetRelativeLocation(At(Home(PPad), 0.3765));
    const TPair<USceneComponent*, int32> Fixtures[] = {
        {CabinetRoot.Get(), PCabinet}, {ValveRoot.Get(), PValve}, {ScrubberRoot.Get(), PScrubber}, {ContainerRoot.Get(), PContainer}};
    for (const TPair<USceneComponent*, int32>& Fixture : Fixtures)
    {
        Fixture.Key->SetVisibility(true, true);
        Fixture.Key->SetRelativeLocation(At(Home(Fixture.Value), BenchZ));
    }
    DrawerOpenM = 0.0f;
    ValveDeg = 0.0f;
    LidDeg = 90.0f;
    bToolStowed = bVialStowed = bCartridgeSeated = bLidLatched = false;
    UpdateMechanisms();
}

void ABASWorkcellActor::ResetTask()
{
    if (HeldPart)
    {
        Release();
    }
    if (HeldHandle != EBASHandle::None && PandaRobot)
    {
        ReleaseHandle();
    }
    HeldHandle = EBASHandle::None;
    PlaceTaskProps();
}

void ABASWorkcellActor::RandomizeLayout(FRandomStream& Stream)
{
    Jitter.SetNum(PropCount);
    for (FVector2D& Shift : Jitter)
    {
        Shift = FVector2D(Stream.FRandRange(-LayoutJitterM, LayoutJitterM), Stream.FRandRange(-LayoutJitterM, LayoutJitterM));
    }
}

FString ABASWorkcellActor::GetTaskInstruction() const
{
    switch (Task)
    {
    case EBASTask::CubeToPad: return TEXT("Pick up the red cube and place it on the yellow marker.");
    case EBASTask::StowInDrawer: return TEXT("Put the tool in the drawer and close it.");
    case EBASTask::CloseValve: return TEXT("Close the coolant valve.");
    case EBASTask::InsertFilter: return TEXT("Insert the filter cartridge into the air scrubber.");
    case EBASTask::SealContainer: return TEXT("Put the sample vial in the container and close the lid.");
    }
    return FString();
}

FVector ABASWorkcellActor::GetActiveObjectMujoco() const
{
    if (UStaticMeshComponent* Object = ActiveObject())
    {
        return ToMujoco(Object->GetComponentLocation());
    }
    return ToMujoco(ValveLever->GetComponentLocation());
}

FVector ABASWorkcellActor::GetTargetMujoco() const
{
    switch (Task)
    {
    case EBASTask::StowInDrawer:
    {
        // Middle of the exposed drawer floor when fully open.
        const FVector2D Cabinet = Home(PCabinet);
        return FVector(Cabinet.X, Cabinet.Y + CabinetHalf.Y + DrawerMaxOpen / 2, BenchZ + DrawerFloorTop);
    }
    case EBASTask::CloseValve: return ToMujoco(ValveLever->GetComponentLocation());
    case EBASTask::InsertFilter: return FVector(Home(PScrubber).X, Home(PScrubber).Y, BenchZ + 2 * ScrubberHalf.Z);
    case EBASTask::SealContainer: return FVector(Home(PContainer).X, Home(PContainer).Y, BenchZ + 0.006);
    default: return FVector(Home(PPad).X, Home(PPad).Y, 0.377);
    }
}

void ABASWorkcellActor::GetTaskState(TMap<FString, double>& Out) const
{
    const auto Put = [&Out](const TCHAR* Name, const FVector& V)
    {
        Out.Add(FString(Name) + TEXT("_x"), V.X);
        Out.Add(FString(Name) + TEXT("_y"), V.Y);
        Out.Add(FString(Name) + TEXT("_z"), V.Z);
    };
    Put(TEXT("object"), GetActiveObjectMujoco());
    Put(TEXT("target"), GetTargetMujoco());
    if (const UStaticMeshComponent* Handle = ActiveHandle())
    {
        Put(TEXT("handle"), ToMujoco(Handle->GetComponentLocation()));
    }
    Out.Add(TEXT("held_object"), bObjectHeld ? 1.0 : 0.0);
    Out.Add(TEXT("held_handle"), HeldHandle != EBASHandle::None ? 1.0 : 0.0);
    Out.Add(TEXT("drawer_open"), DrawerOpenM);
    Out.Add(TEXT("valve_deg"), ValveDeg);
    Out.Add(TEXT("lid_deg"), LidDeg);
    Out.Add(TEXT("tool_stowed"), bToolStowed ? 1.0 : 0.0);
    Out.Add(TEXT("vial_stowed"), bVialStowed ? 1.0 : 0.0);
    Out.Add(TEXT("seated"), bCartridgeSeated ? 1.0 : 0.0);
    Out.Add(TEXT("latched"), bLidLatched ? 1.0 : 0.0);
}

bool ABASWorkcellActor::IsTaskSuccess() const
{
    const bool bHandsFree = !bObjectHeld && HeldHandle == EBASHandle::None && PandaRobot &&
        PandaRobot->GetGripperOpenness() > 0.8f;
    switch (Task)
    {
    case EBASTask::CubeToPad: return bHandsFree && IsCubePlaced();
    case EBASTask::StowInDrawer: return bHandsFree && bToolStowed && DrawerOpenM <= 0.001f;
    case EBASTask::CloseValve: return bHandsFree && FMath::Abs(ValveDeg) >= 89.0f;
    case EBASTask::InsertFilter: return bHandsFree && bCartridgeSeated;
    case EBASTask::SealContainer: return bHandsFree && bVialStowed && bLidLatched;
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
    PlaceTaskProps();

    SetColour(RobotMount, FLinearColor(0.30f, 0.33f, 0.38f));
    SetColour(MountPlate, FLinearColor(0.46f, 0.49f, 0.54f));
    SetColour(BenchRack, FLinearColor(0.46f, 0.49f, 0.54f));
    SetColour(BenchTop, FLinearColor(0.68f, 0.71f, 0.75f));
    SetColour(TargetMarker, FLinearColor(1.0f, 0.85f, 0.0f));
    SetColour(PickCube, FLinearColor(0.9f, 0.12f, 0.08f));
    SetColour(Tool, FLinearColor(0.1f, 0.65f, 0.2f));        // green tool
    SetColour(Cartridge, FLinearColor(0.1f, 0.6f, 0.6f));    // teal filter cartridge
    SetColour(Vial, FLinearColor(0.95f, 0.35f, 0.6f));       // pink sample vial
    for (int32 Index = 0; Index < ColouredParts.Num() && Index < PartColours.Num(); ++Index)
    {
        SetColour(ColouredParts[Index], PartColours[Index]);
    }
    UpdateMechanisms();
}
