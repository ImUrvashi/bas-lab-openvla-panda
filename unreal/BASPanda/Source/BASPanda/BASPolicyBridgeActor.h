#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "BASPolicyBridgeActor.generated.h"

class ABASWorkcellActor;
class APandaRobotActor;

/** Observation/HTTP bridge: one-shot test, or a closed loop that runs whole episodes. */
UCLASS(Blueprintable)
class BASPANDA_API ABASPolicyBridgeActor : public AActor
{
    GENERATED_BODY()

public:
    ABASPolicyBridgeActor();
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;

    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    FString PolicyUrl = TEXT("http://127.0.0.1:8000/act");

    /** The sentence the side_v12 model was trained on. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    FString Instruction = TEXT("Pick up the red cube and place it on the yellow marker.");

    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    bool bSendOnBeginPlay = false;

    /** Apply an explicit 8D joint_targets field from the mock or IK gateway, never raw 7D actions. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    bool bApplyJointTargets = false;

    UPROPERTY(VisibleAnywhere, Category = "BAS Policy Test")
    FString LastResult = TEXT("Not sent");

    UFUNCTION(CallInEditor, Category = "BAS Policy Test")
    void SendTestObservation();

    /** Run episodes on Play/Simulate: capture -> gateway -> IK -> one 0.1 s arm step, repeated. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop")
    bool bRunClosedLoopOnBeginPlay = false;

    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop", meta = (ClampMin = "1"))
    int32 NumEpisodes = 1;

    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop", meta = (ClampMin = "1"))
    int32 MaxSteps = 300;

    /** Simulated time per policy step; the world waits for each action, so latency only slows playback. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop", meta = (ClampMin = "0.02"))
    float StepDurationSec = 0.1f;

    /** Off: use the workcell's CubeStartXY/MarkerXY. On: random layouts from the training ranges. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop")
    bool bRandomizeLayout = false;

    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop")
    int32 LayoutSeed = 777;

    UPROPERTY(VisibleAnywhere, Category = "BAS Policy Loop")
    FString LoopStatus = TEXT("Not run");

    UFUNCTION(BlueprintCallable, Category = "BAS Policy Loop")
    void StartClosedLoop();

    UFUNCTION(BlueprintCallable, Category = "BAS Policy Loop")
    void StopClosedLoop(const FString& Reason);

private:
    bool SendObservation();
    void HandleResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bSucceeded);
    void StartEpisode();
    void EndEpisode(bool bSuccess, const FString& Why);
    bool FindActors();

    UPROPERTY(Transient)
    TObjectPtr<APandaRobotActor> Panda;

    UPROPERTY(Transient)
    TObjectPtr<ABASWorkcellActor> Workcell;

    FRandomStream LayoutStream;
    bool bLoopActive = false;
    bool bRequestInFlight = false;
    int32 Episode = 0;
    int32 Step = 0;
    int32 SuccessStreak = 0;
    int32 Successes = 0;
    double RequestStartTime = 0.0;
    double LatencySum = 0.0;
    int32 TotalRequests = 0;
};
