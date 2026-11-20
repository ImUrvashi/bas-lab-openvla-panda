#include "PandaRobotActor.h"

#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Misc/MessageDialog.h"
#include "UObject/UObjectGlobals.h"

namespace
{
    struct FLinkSpec
    {
        const TCHAR* Name;
        int32 ParentIndex;
        FVector LocationCm;
        FQuat FixedRotation;
    };

    // Values from unreal/exported/Panda/panda_manifest.json. Meshes are local
    // to each body's pivot and already use Unreal centimetres / X,-Y,Z axes.
    const FLinkSpec Links[] = {
        {TEXT("link0"),        -1, FVector(0, 0, 0),          FQuat::Identity},
        {TEXT("link1"),         0, FVector(0, 0, 33.3),       FQuat::Identity},
        {TEXT("link2"),         1, FVector(0, 0, 0),          FQuat(0.70710678, 0, 0, 0.70710678)},
        {TEXT("link3"),         2, FVector(0, 31.6, 0),       FQuat(-0.70710678, 0, 0, 0.70710678)},
        {TEXT("link4"),         3, FVector(8.25, 0, 0),       FQuat(-0.70710678, 0, 0, 0.70710678)},
        {TEXT("link5"),         4, FVector(-8.25, -38.4, 0),  FQuat(0.70710678, 0, 0, 0.70710678)},
        {TEXT("link6"),         5, FVector(0, 0, 0),          FQuat(-0.70710678, 0, 0, 0.70710678)},
        {TEXT("link7"),         6, FVector(8.8, 0, 0),        FQuat(-0.70710678, 0, 0, 0.70710678)},
        {TEXT("hand"),          7, FVector(0, 0, 10.7),       FQuat(0, 0, 0.38268343, 0.92387953)},
        {TEXT("left_finger"),   8, FVector(0, 0, 5.84),       FQuat::Identity},
        {TEXT("right_finger"),  8, FVector(0, 0, 5.84),       FQuat(0, 0, -1, 0)},
    };

    constexpr int32 NumLinks = UE_ARRAY_COUNT(Links);
    constexpr int32 ArmStartIndex = 1;
    constexpr int32 NumArmJoints = 7;
    constexpr int32 LeftFingerIndex = 9;
    constexpr int32 RightFingerIndex = 10;
    const float JointLower[NumArmJoints] = {-2.8973f, -1.7628f, -2.8973f, -3.0718f, -2.8973f, -0.0175f, -2.8973f};
    const float JointUpper[NumArmJoints] = {2.8973f, 1.7628f, 2.8973f, -0.0698f, 2.8973f, 3.7525f, 2.8973f};
}

APandaRobotActor::APandaRobotActor()
{
    PrimaryActorTick.bCanEverTick = true;
    PandaRoot = CreateDefaultSubobject<USceneComponent>(TEXT("PandaRoot"));
    RootComponent = PandaRoot;

    MotionFrames.Reserve(NumLinks);
    for (int32 Index = 0; Index < NumLinks; ++Index)
    {
        const FLinkSpec& Spec = Links[Index];
        USceneComponent* FixedFrame = CreateDefaultSubobject<USceneComponent>(
            FName(FString::Printf(TEXT("%s_Fixed"), Spec.Name)));
        FixedFrame->SetupAttachment(Spec.ParentIndex < 0 ? PandaRoot.Get() : MotionFrames[Spec.ParentIndex].Get());
        FixedFrame->SetRelativeLocationAndRotation(Spec.LocationCm, Spec.FixedRotation);

        USceneComponent* MotionFrame = CreateDefaultSubobject<USceneComponent>(
            FName(FString::Printf(TEXT("%s_Motion"), Spec.Name)));
        MotionFrame->SetupAttachment(FixedFrame);
        MotionFrames.Add(MotionFrame);

        UStaticMeshComponent* Visual = CreateDefaultSubobject<UStaticMeshComponent>(
            FName(FString::Printf(TEXT("%s_Visual"), Spec.Name)));
        Visual->SetupAttachment(MotionFrame);
        Visual->SetMobility(EComponentMobility::Movable);
        Visual->SetCollisionEnabled(ECollisionEnabled::NoCollision);

        const FString AssetPath = FString::Printf(TEXT("/Game/%s.%s"), Spec.Name, Spec.Name);
        if (UStaticMesh* Mesh = LoadObject<UStaticMesh>(nullptr, *AssetPath))
        {
            Visual->SetStaticMesh(Mesh);
        }
    }

    GraspFrame = CreateDefaultSubobject<USceneComponent>(TEXT("GraspFrame"));
    GraspFrame->SetupAttachment(MotionFrames[8]);
    GraspFrame->SetRelativeLocation(FVector(0.0, 0.0, 9.0));

    ArmJointsRad = {0.0f, 0.0f, 0.0f, -1.57079f, 0.0f, 1.57079f, -0.7853f};
    FingerOpeningCm = 4.0f;
    ApplyJointPose();
}

void APandaRobotActor::BeginPlay()
{
    Super::BeginPlay();
    if (bRunMotionTestOnBeginPlay && ArmJointsRad.Num() == NumArmJoints)
    {
        MotionTestStartJoints = ArmJointsRad;
        MotionTestElapsed = 0.0f;
        bMotionTestActive = true;
        MotionTestStatus = TEXT("Running two-second joint 1 sweep");
        UE_LOG(LogTemp, Display, TEXT("Panda motion test started"));
    }
}

void APandaRobotActor::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    if (bMotionTestActive)
    {
        MotionTestElapsed += DeltaSeconds;
        if (MotionTestElapsed >= 2.0f)
        {
            ArmJointsRad = MotionTestStartJoints;
            ApplyJointPose();
            bMotionTestActive = false;
            MotionTestStatus = TEXT("Complete: starting pose restored");
            UE_LOG(LogTemp, Display, TEXT("Panda motion test complete; starting pose restored"));
            return;
        }
        const float Sweep = 0.1f * FMath::Sin(PI * MotionTestElapsed / 2.0f);
        ArmJointsRad[0] = FMath::Clamp(MotionTestStartJoints[0] + Sweep, JointLower[0], JointUpper[0]);
        ApplyJointPose();
        return;
    }
    if (!bCommandActive || CommandTargetsRad.Num() != NumArmJoints)
    {
        return;
    }
    bool bAtTarget = true;
    for (int32 Joint = 0; Joint < NumArmJoints; ++Joint)
    {
        ArmJointsRad[Joint] = FMath::FInterpConstantTo(ArmJointsRad[Joint], CommandTargetsRad[Joint], DeltaSeconds,
            CommandJointSpeeds.IsValidIndex(Joint) ? CommandJointSpeeds[Joint] : 0.5f);
        bAtTarget &= FMath::IsNearlyEqual(ArmJointsRad[Joint], CommandTargetsRad[Joint], 0.001f);
    }
    CommandFingerTargetCm = FMath::Max(CommandFingerTargetCm, FingerStopCm);
    FingerOpeningCm = FMath::FInterpConstantTo(FingerOpeningCm, CommandFingerTargetCm, DeltaSeconds, CommandFingerSpeed);
    bAtTarget &= FMath::IsNearlyEqual(FingerOpeningCm, CommandFingerTargetCm, 0.001f);
    ApplyJointPose();
    bCommandActive = !bAtTarget;
}

bool APandaRobotActor::CommandJointTargets(const TArray<float>& JointTargetsRad, float GripperOpenness, float DurationSec)
{
    if (!HasActorBegunPlay() || bMotionTestActive || ArmJointsRad.Num() != NumArmJoints ||
        JointTargetsRad.Num() != NumArmJoints || !FMath::IsFinite(GripperOpenness))
    {
        return false;
    }
    CommandTargetsRad.SetNum(NumArmJoints);
    for (int32 Joint = 0; Joint < NumArmJoints; ++Joint)
    {
        if (!FMath::IsFinite(JointTargetsRad[Joint]))
        {
            return false;
        }
        const float SafeTarget = FMath::Clamp(JointTargetsRad[Joint], JointLower[Joint], JointUpper[Joint]);
        CommandTargetsRad[Joint] = FMath::Clamp(SafeTarget, ArmJointsRad[Joint] - 0.15f, ArmJointsRad[Joint] + 0.15f);
    }
    CommandFingerTargetCm = FMath::Max(4.0f * FMath::Clamp(GripperOpenness, 0.0f, 1.0f), FingerStopCm);

    // Timed step: every joint arrives together after DurationSec. Otherwise the 0.5 rad/s preview speed.
    CommandJointSpeeds.SetNum(NumArmJoints);
    for (int32 Joint = 0; Joint < NumArmJoints; ++Joint)
    {
        CommandJointSpeeds[Joint] = DurationSec > 0.0f ?
            FMath::Max(FMath::Abs(CommandTargetsRad[Joint] - ArmJointsRad[Joint]) / DurationSec, 1e-3f) : 0.5f;
    }
    CommandFingerSpeed = DurationSec > 0.0f ?
        FMath::Max(FMath::Abs(CommandFingerTargetCm - FingerOpeningCm) / DurationSec, 1e-3f) : 4.0f;
    bCommandActive = true;
    return true;
}

void APandaRobotActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);

    // The editor may load the native actor before the user imports the OBJ
    // assets. Retry missing bindings whenever the actor is reconstructed.
    TArray<UStaticMeshComponent*> Components;
    GetComponents<UStaticMeshComponent>(Components);
    for (UStaticMeshComponent* Component : Components)
    {
        if (Component->GetStaticMesh() != nullptr)
        {
            continue;
        }
        const FString ComponentName = Component->GetName();
        for (const FLinkSpec& Spec : Links)
        {
            if (ComponentName == FString::Printf(TEXT("%s_Visual"), Spec.Name))
            {
                const FString AssetPath = FString::Printf(TEXT("/Game/%s.%s"), Spec.Name, Spec.Name);
                Component->SetStaticMesh(LoadObject<UStaticMesh>(nullptr, *AssetPath));
                break;
            }
        }
    }
    ApplyJointPose();
}

void APandaRobotActor::InspectImportedLinks()
{
#if WITH_EDITOR
    TArray<UStaticMeshComponent*> Components;
    GetComponents<UStaticMeshComponent>(Components);
    FString Report = TEXT("Panda visual link check (centimetres)\n");
    for (const FLinkSpec& Spec : Links)
    {
        UStaticMeshComponent* Visual = nullptr;
        const FString ExpectedName = FString::Printf(TEXT("%s_Visual"), Spec.Name);
        for (UStaticMeshComponent* Candidate : Components)
        {
            if (Candidate->GetName() == ExpectedName)
            {
                Visual = Candidate;
                break;
            }
        }
        if (!Visual || !Visual->GetStaticMesh())
        {
            Report += FString::Printf(TEXT("%s: MISSING mesh/component\n"), Spec.Name);
            continue;
        }
        const FBox Local = Visual->GetStaticMesh()->GetBounds().GetBox();
        const FBox World = Visual->CalcBounds(Visual->GetComponentTransform()).GetBox();
        Report += FString::Printf(
            TEXT("%s: local Z %.1f..%.1f, world Z %.1f..%.1f, visible %s\n"),
            Spec.Name, Local.Min.Z, Local.Max.Z, World.Min.Z, World.Max.Z,
            Visual->IsVisible() ? TEXT("yes") : TEXT("no"));
    }
    UE_LOG(LogTemp, Display, TEXT("%s"), *Report);
    FMessageDialog::Open(EAppMsgType::Ok, FText::FromString(Report));
#endif
}

void APandaRobotActor::ApplyJointPose()
{
    if (MotionFrames.Num() != NumLinks || ArmJointsRad.Num() != NumArmJoints)
    {
        return;
    }

    // The coordinate reflection turns each MJCF +Z hinge into an Unreal -Z
    // hinge. Angles remain the original MuJoCo values in radians.
    for (int32 Joint = 0; Joint < NumArmJoints; ++Joint)
    {
        MotionFrames[ArmStartIndex + Joint]->SetRelativeRotation(
            FQuat(FVector(0, 0, -1), ArmJointsRad[Joint]));
    }
    const float Opening = FMath::Clamp(FingerOpeningCm, 0.0f, 4.0f);
    const FVector FingerOffset(0, -Opening, 0);
    MotionFrames[LeftFingerIndex]->SetRelativeLocation(FingerOffset);
    MotionFrames[RightFingerIndex]->SetRelativeLocation(FingerOffset);
}

void APandaRobotActor::SetFingerStopCm(float StopCm)
{
    FingerStopCm = FMath::Clamp(StopCm, 0.0f, 4.0f);
    // Apply at once: at low frame rates the fingers can pass the stop within a single tick.
    CommandFingerTargetCm = FMath::Max(CommandFingerTargetCm, FingerStopCm);
    if (FingerOpeningCm < FingerStopCm)
    {
        FingerOpeningCm = FingerStopCm;
        ApplyJointPose();
    }
}

void APandaRobotActor::ResetHome()
{
    bCommandActive = false;
    FingerStopCm = 0.0f;
    ArmJointsRad = {0.0f, 0.0f, 0.0f, -1.57079f, 0.0f, 1.57079f, -0.7853f};
    FingerOpeningCm = 4.0f;
    ApplyJointPose();
}

FVector APandaRobotActor::GetGraspCenterWorld() const
{
    return GraspFrame ? GraspFrame->GetComponentLocation() : GetActorLocation();
}
