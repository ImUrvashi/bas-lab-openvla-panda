#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BASStationModuleActor.generated.h"

class UInstancedStaticMeshComponent;
class UPointLightComponent;
class USceneComponent;

/** Source-sized BAS lab shell, rack panels, and end walls. */
UCLASS(Blueprintable)
class BASPANDA_API ABASStationModuleActor : public AActor
{
    GENERATED_BODY()

public:
    ABASStationModuleActor();
    virtual void OnConstruction(const FTransform& Transform) override;

    /** Open one side of the shell while arranging the workcell and camera. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BAS Station")
    bool bCutawayForEditing = true;

private:
    UPROPERTY(VisibleAnywhere, Category = "BAS Station")
    TObjectPtr<USceneComponent> ModuleRoot;

    UPROPERTY(VisibleAnywhere, Category = "BAS Station")
    TObjectPtr<UInstancedStaticMeshComponent> ShellPanels;

    UPROPERTY(VisibleAnywhere, Category = "BAS Station")
    TArray<TObjectPtr<UInstancedStaticMeshComponent>> StationBoxes;

    UPROPERTY(VisibleAnywhere, Category = "BAS Station")
    TObjectPtr<UInstancedStaticMeshComponent> EndCaps;

    UPROPERTY(VisibleAnywhere, Category = "BAS Station")
    TObjectPtr<UInstancedStaticMeshComponent> HatchRings;

    UPROPERTY(VisibleAnywhere, Category = "BAS Station")
    TObjectPtr<UInstancedStaticMeshComponent> HatchDoors;

    UPROPERTY(VisibleAnywhere, Category = "BAS Station")
    TObjectPtr<UPointLightComponent> ForwardCabinLight;

    UPROPERTY(VisibleAnywhere, Category = "BAS Station")
    TObjectPtr<UPointLightComponent> AftCabinLight;
};
