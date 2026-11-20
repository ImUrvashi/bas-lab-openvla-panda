#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "BASPolicyBridgeActor.generated.h"

/** One-shot observation/HTTP bridge with opt-in, bounded joint targets. */
UCLASS(Blueprintable)
class BASPANDA_API ABASPolicyBridgeActor : public AActor
{
    GENERATED_BODY()

public:
    ABASPolicyBridgeActor();
    virtual void BeginPlay() override;

    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    FString PolicyUrl = TEXT("http://127.0.0.1:8000/act");

    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    FString Instruction = TEXT("pick up the red cube");

    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    bool bSendOnBeginPlay = true;

    /** Apply an explicit 8D joint_targets field from the mock or IK gateway, never raw 7D actions. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    bool bApplyJointTargets = false;

    UPROPERTY(VisibleAnywhere, Category = "BAS Policy Test")
    FString LastResult = TEXT("Not sent");

    UFUNCTION(CallInEditor, Category = "BAS Policy Test")
    void SendTestObservation();

private:
    void HandleResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bSucceeded);
};
