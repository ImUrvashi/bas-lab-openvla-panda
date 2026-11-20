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

    // Engine/BasicShapes/Cube is 100 cm on each side. Source sizes below are
    // half-extents in metres from bas_lab/mod_workstation.xml.
    void ConfigureBox(UStaticMeshComponent* Part, UStaticMesh* CubeMesh,
        const FVector& PositionMetres, const FVector& HalfSizeMetres,
        bool bCollides)
    {
        Part->SetStaticMesh(CubeMesh);
        Part->SetRelativeLocation(PositionMetres * 100.0);
        Part->SetRelativeScale3D(HalfSizeMetres * 2.0);
        Part->SetCollisionEnabled(bCollides ? ECollisionEnabled::QueryAndPhysics : ECollisionEnabled::NoCollision);
    }
}

ABASWorkcellActor::ABASWorkcellActor()
{
    PrimaryActorTick.bCanEverTick = true;
    WorkcellRoot = CreateDefaultSubobject<USceneComponent>(TEXT("WorkcellRoot"));
    RootComponent = WorkcellRoot;

    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));

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

    TargetMarker = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("TargetMarker"));
    TargetMarker->SetupAttachment(WorkcellRoot);
    ConfigureBox(TargetMarker, CubeMesh, FVector(0.60, 0, 0.3765), FVector(0.05, 0.05, 0.0005), false);

    PickCube = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("PickCube"));
    PickCube->SetupAttachment(WorkcellRoot);
    PickCube->SetMobility(EComponentMobility::Movable);
    ConfigureBox(PickCube, CubeMesh, FVector(0.45, 0, 0.40), FVector(0.025, 0.025, 0.025), true);

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

void ABASWorkcellActor::BeginPlay()
{
    Super::BeginPlay();

    // MuJoCo's pick_cube is an 80 g free body in zero gravity. The bench's
    // adhesion actuator and finger contacts still need a separate implementation.
    PickCube->SetSimulatePhysics(true);
    PickCube->SetEnableGravity(false);
    PickCube->SetMassOverrideInKg(NAME_None, 0.08f, true);
    PickCube->SetLinearDamping(0.01f);
    PickCube->SetAngularDamping(0.01f);

    if (!PandaRobot)
    {
        PandaRobot = Cast<APandaRobotActor>(UGameplayStatics::GetActorOfClass(this, APandaRobotActor::StaticClass()));
    }
    if (bRunScriptedPickPlaceOnBeginPlay)
    {
        if (!PandaRobot || !GetActorLocation().IsNearlyZero(1.0f) ||
            !PandaRobot->GetActorLocation().IsNearlyZero(1.0f) ||
            !GetActorRotation().IsNearlyZero() || !PandaRobot->GetActorRotation().IsNearlyZero())
        {
            ScriptedTestStatus = TEXT("Failed: place workcell and Panda at world origin with zero rotation");
            UE_LOG(LogTemp, Warning, TEXT("BAS scripted test: %s"), *ScriptedTestStatus);
        }
        else
        {
            PandaRobot->ResetHome();
            ResetCube();
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

    if (bCubeHeld)
    {
        if (PandaRobot->FingerOpeningCm >= 3.2f)
        {
            ReleaseCube();
        }
    }
    else if (PandaRobot->FingerOpeningCm <= 2.7f &&
        FVector::Dist(PickCube->GetComponentLocation(), PandaRobot->GetGraspCenterWorld()) <= 6.0f)
    {
        GraspCube();
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
        (Pose.bRequiresCube && ScriptedPhase < UE_ARRAY_COUNT(ScriptedPoses) - 1 && !bCubeHeld))
    {
        ScriptedTestStatus = FString::Printf(TEXT("Failed at %s: %s"), Pose.Name,
            bCubeHeld ? TEXT("timed out") : TEXT("cube was not held"));
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
        4.0f * Pose.GripperOpenness, 0.05f);
    if (ScriptedPhase == 2)
    {
        bPoseReached &= bCubeHeld;
    }
    if (ScriptedPhase == UE_ARRAY_COUNT(ScriptedPoses) - 1)
    {
        bPoseReached &= !bCubeHeld;
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

void ABASWorkcellActor::GraspCube()
{
    USceneComponent* Frame = PandaRobot ? PandaRobot->GetGraspFrame() : nullptr;
    if (!Frame || bCubeHeld)
    {
        return;
    }

    PickCube->SetSimulatePhysics(false);
    PickCube->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    if (PickCube->AttachToComponent(Frame, FAttachmentTransformRules::KeepWorldTransform))
    {
        PickCube->SetWorldLocation(Frame->GetComponentLocation());
        bCubeHeld = true;
    }
    else
    {
        PickCube->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        PickCube->SetSimulatePhysics(true);
        PickCube->SetEnableGravity(false);
    }
}

void ABASWorkcellActor::ReleaseCube()
{
    if (!bCubeHeld)
    {
        return;
    }

    PickCube->DetachFromComponent(FDetachmentTransformRules::KeepWorldTransform);
    PickCube->AttachToComponent(WorkcellRoot, FAttachmentTransformRules::KeepWorldTransform);
    PickCube->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    PickCube->SetSimulatePhysics(true);
    PickCube->SetEnableGravity(false);
    PickCube->SetPhysicsLinearVelocity(FVector::ZeroVector);
    PickCube->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
    bCubeHeld = false;
}

void ABASWorkcellActor::ResetCube()
{
    if (bCubeHeld)
    {
        ReleaseCube();
    }
    const bool bResumePhysics = PickCube->IsSimulatingPhysics();
    if (bResumePhysics)
    {
        PickCube->SetSimulatePhysics(false);
    }
    PickCube->SetRelativeLocationAndRotation(FVector(45.0, 0.0, 40.0), FRotator::ZeroRotator);
    if (bResumePhysics)
    {
        PickCube->SetSimulatePhysics(true);
        PickCube->SetEnableGravity(false);
        PickCube->SetPhysicsLinearVelocity(FVector::ZeroVector);
        PickCube->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
    }
}

bool ABASWorkcellActor::IsCubePlaced() const
{
    const FVector CubeLocation = PickCube->GetComponentLocation();
    const FVector TargetLocation = TargetMarker->GetComponentLocation();
    const FVector Delta = CubeLocation - TargetLocation;
    const float ExpectedCubeZ = BenchTop->GetComponentLocation().Z + 2.5f + 2.5f;
    return !bCubeHeld && FVector2D(Delta.X, Delta.Y).Size() <= 4.0f &&
        FMath::Abs(CubeLocation.Z - ExpectedCubeZ) <= 2.0f;
}

void ABASWorkcellActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    UMaterialInterface* BaseMaterial = LoadObject<UMaterialInterface>(
        nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
    if (!BaseMaterial)
    {
        return;
    }

    const auto Colour = [this, BaseMaterial](UStaticMeshComponent* Part, const FLinearColor& Value)
    {
        UMaterialInstanceDynamic* Material = UMaterialInstanceDynamic::Create(BaseMaterial, this);
        Material->SetVectorParameterValue(TEXT("Color"), Value);
        Part->SetMaterial(0, Material);
    };

    Colour(RobotMount, FLinearColor(0.30f, 0.33f, 0.38f));
    Colour(MountPlate, FLinearColor(0.46f, 0.49f, 0.54f));
    Colour(BenchRack, FLinearColor(0.46f, 0.49f, 0.54f));
    Colour(BenchTop, FLinearColor(0.68f, 0.71f, 0.75f));
    Colour(TargetMarker, FLinearColor(1.0f, 0.85f, 0.0f));
    Colour(PickCube, FLinearColor(0.9f, 0.12f, 0.08f));
}
