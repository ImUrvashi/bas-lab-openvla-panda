#include "BASStationModuleActor.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Components/PointLightComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Components/SceneComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "UObject/UObjectGlobals.h"

namespace
{
    struct FStationBox
    {
        FVector PositionCm;
        FQuat Rotation;
        FVector Scale;
        uint8 MaterialIndex;
    };

#include "StationModuleBoxes.inl"

    // Same order as MATERIALS in unreal/tools/export_bas_station.py.
    const FLinearColor StationMaterialColors[] = {
        FLinearColor(0.46f, 0.49f, 0.54f), // rack_body
        FLinearColor(0.70f, 0.72f, 0.76f), // locker_a
        FLinearColor(0.62f, 0.66f, 0.71f), // locker_b
        FLinearColor(0.60f, 0.67f, 0.64f), // locker_c
        FLinearColor(0.12f, 0.13f, 0.16f), // dark
        FLinearColor(0.10f, 0.55f, 0.60f), // accent_teal
        FLinearColor(0.15f, 0.35f, 0.75f), // accent_blue
        FLinearColor(0.08f, 0.38f, 0.55f), // screen
        FLinearColor(0.04f, 0.18f, 0.30f), // screen_dim
        FLinearColor(0.48f, 0.50f, 0.56f), // frame_mat
        FLinearColor(0.10f, 1.00f, 0.30f), // led_green
        FLinearColor(0.20f, 0.60f, 1.00f), // led_blue
        FLinearColor(1.00f, 1.00f, 1.00f), // led_white
    };
    static_assert(UE_ARRAY_COUNT(StationMaterialColors) == 13);

    void Colour(UPrimitiveComponent* Part, UMaterialInterface* Base, UObject* Owner,
        const FLinearColor& Value)
    {
        UMaterialInstanceDynamic* Material = UMaterialInstanceDynamic::Create(Base, Owner);
        Material->SetVectorParameterValue(TEXT("Color"), Value);
        Part->SetMaterial(0, Material);
    }
}

ABASStationModuleActor::ABASStationModuleActor()
{
    PrimaryActorTick.bCanEverTick = false;
    ModuleRoot = CreateDefaultSubobject<USceneComponent>(TEXT("ModuleRoot"));
    RootComponent = ModuleRoot;

    ShellPanels = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("ShellPanels"));
    ShellPanels->SetupAttachment(ModuleRoot);
    UStaticMesh* Cube = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
    UStaticMesh* Cylinder = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    ShellPanels->SetStaticMesh(Cube);
    // The source shell is decorative. Contact comes from the workcell only.
    ShellPanels->SetCollisionEnabled(ECollisionEnabled::NoCollision);

    for (int32 Index = 0; Index < UE_ARRAY_COUNT(StationMaterialColors); ++Index)
    {
        const FName Name(*FString::Printf(TEXT("StationBoxes_%02d"), Index));
        UInstancedStaticMeshComponent* Component = CreateDefaultSubobject<UInstancedStaticMeshComponent>(Name);
        Component->SetupAttachment(ModuleRoot);
        Component->SetStaticMesh(Cube);
        Component->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        StationBoxes.Add(Component);
    }

    EndCaps = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("EndCaps"));
    EndCaps->SetupAttachment(ModuleRoot);
    EndCaps->SetStaticMesh(Cylinder);
    EndCaps->SetCollisionEnabled(ECollisionEnabled::NoCollision);

    HatchRings = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("HatchRings"));
    HatchRings->SetupAttachment(ModuleRoot);
    HatchRings->SetStaticMesh(Cylinder);
    HatchRings->SetCollisionEnabled(ECollisionEnabled::NoCollision);

    HatchDoors = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("HatchDoors"));
    HatchDoors->SetupAttachment(ModuleRoot);
    HatchDoors->SetStaticMesh(Cylinder);
    HatchDoors->SetCollisionEnabled(ECollisionEnabled::NoCollision);

    ForwardCabinLight = CreateDefaultSubobject<UPointLightComponent>(TEXT("ForwardCabinLight"));
    ForwardCabinLight->SetupAttachment(ModuleRoot);
    ForwardCabinLight->SetRelativeLocation(FVector(55, -170, 175));
    ForwardCabinLight->SetIntensity(3000.0f);
    ForwardCabinLight->SetAttenuationRadius(450.0f);

    AftCabinLight = CreateDefaultSubobject<UPointLightComponent>(TEXT("AftCabinLight"));
    AftCabinLight->SetupAttachment(ModuleRoot);
    AftCabinLight->SetRelativeLocation(FVector(55, 170, 175));
    AftCabinLight->SetIntensity(3000.0f);
    AftCabinLight->SetAttenuationRadius(450.0f);
}

void ABASStationModuleActor::OnConstruction(const FTransform& Transform)
{
    Super::OnConstruction(Transform);
    ShellPanels->ClearInstances();
    for (UInstancedStaticMeshComponent* Component : StationBoxes)
    {
        Component->ClearInstances();
    }
    EndCaps->ClearInstances();
    HatchRings->ClearInstances();
    HatchDoors->ClearInstances();

    // mod_bas_module.xml: 24 sides, axis along Y, centre X=0.55 m,
    // Z=0.85 m, panel centres at radius 1.925 m, length 7.0 m.
    constexpr int32 PanelCount = 24;
    constexpr double RadiusCm = 192.5;
    constexpr double SegmentWidthCm = 52.02;
    for (int32 Index = 0; Index < PanelCount; ++Index)
    {
        const double Angle = 2.0 * PI * Index / PanelCount;
        const FVector Radial(FMath::Cos(Angle), 0.0, FMath::Sin(Angle));
        // A side opening makes the robot visible in the editor; disabling the
        // cutaway restores all 24 panels without changing their coordinates.
        if (bCutawayForEditing && Radial.X > 0.50)
        {
            continue;
        }
        const FVector Centre = FVector(55, 0, 85) + RadiusCm * Radial;
        const FQuat Orientation = FQuat::FindBetweenVectors(FVector::UpVector, Radial);
        const FVector CubeScale(SegmentWidthCm / 100.0, 7.0, 0.05);
        ShellPanels->AddInstance(FTransform(Orientation, Centre, CubeScale));
    }

    // The source includes 77 racks and their doors, screens, handles and LEDs.
    // Removing the +X side allows inspection while preserving the -X wall in
    // the side policy camera's background.
    for (const FStationBox& Box : GStationBoxes)
    {
        if (bCutawayForEditing && Box.PositionCm.X > 140.0)
        {
            continue;
        }
        StationBoxes[Box.MaterialIndex]->AddInstance(
            FTransform(Box.Rotation, Box.PositionCm, Box.Scale));
    }

    const FQuat DiscRotation(FVector::ForwardVector, PI / 2.0);
    for (const float Side : {-1.0f, 1.0f})
    {
        EndCaps->AddInstance(FTransform(DiscRotation, FVector(55, Side * 353, 85),
            FVector(3.9, 3.9, 0.06)));
        HatchRings->AddInstance(FTransform(DiscRotation, FVector(55, Side * 347, 85),
            FVector(1.24, 1.24, 0.06)));
        HatchDoors->AddInstance(FTransform(DiscRotation, FVector(55, Side * 343, 85),
            FVector(1.0, 1.0, 0.04)));
    }

    if (UMaterialInterface* Base = LoadObject<UMaterialInterface>(
        nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial")))
    {
        Colour(ShellPanels, Base, this, FLinearColor(0.58f, 0.60f, 0.65f));
        for (int32 Index = 0; Index < StationBoxes.Num(); ++Index)
        {
            Colour(StationBoxes[Index], Base, this, StationMaterialColors[Index]);
        }
        Colour(EndCaps, Base, this, FLinearColor(0.56f, 0.58f, 0.63f));
        Colour(HatchRings, Base, this, StationMaterialColors[9]);
        Colour(HatchDoors, Base, this, StationMaterialColors[2]);
    }
}
