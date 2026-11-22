#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "BASWorkcellActor.h"
#include "BASPolicyBridgeActor.generated.h"

class ABASWorkcellActor;
class APandaRobotActor;
class SWidget;

/** Observation/HTTP bridge: one-shot test, or a closed loop that runs whole episodes. */
UCLASS(Blueprintable)
class BASPANDA_API ABASPolicyBridgeActor : public AActor
{
    GENERATED_BODY()

public:
    ABASPolicyBridgeActor();
    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type Reason) override;
    virtual void Tick(float DeltaSeconds) override;

    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    FString PolicyUrl = TEXT("http://127.0.0.1:8000/act");

    /** Used only when Use Task Instruction is off; task 1's sentence is the one side_v12 was trained on. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    FString Instruction = TEXT("Pick up the red cube and place it on the yellow marker.");

    /** Send the workcell's sentence for its current task (and task 4/5 target). */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Test")
    bool bUseTaskInstruction = true;

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

    /** Task to run; the loop sets it on the workcell at start (and picks that task's sentence and success check). */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop")
    EBASTask Task = EBASTask::CubeToPad;

    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop", meta = (ClampMin = "1"))
    int32 NumEpisodes = 1;

    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop", meta = (ClampMin = "1"))
    int32 MaxSteps = 600;

    /** Simulated time per policy step; the world waits for each action, so latency only slows playback. */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop", meta = (ClampMin = "0.02"))
    float StepDurationSec = 0.1f;

    /** On: ask for the next action while the arm is still moving and stretch each move over the measured round trip,
        so the arm moves continuously instead of stop-go (the camera frame is then one step behind). */
    UPROPERTY(EditAnywhere, Category = "BAS Policy Loop")
    bool bSmoothMotion = true;

    /** On: every episode shifts each prop around its home spot (workcell Layout Jitter M). */
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

    /** Play (not Simulate): on-screen box; a typed request is matched to one of the five tasks and run once. */
    UPROPERTY(EditAnywhere, Category = "BAS Ask")
    bool bShowCommandBox = true;

    /** Gateway that forwards to OpenVLA (policy_ik_gateway.py --upstream-url ...). */
    UPROPERTY(EditAnywhere, Category = "BAS Ask")
    FString ModelUrl = TEXT("http://127.0.0.1:8000/act");

    /** Gateway running the scripted experts (policy_ik_gateway.py --policy expert --port 8002). */
    UPROPERTY(EditAnywhere, Category = "BAS Ask")
    FString ScriptedUrl = TEXT("http://127.0.0.1:8002/act");

    UPROPERTY(EditAnywhere, Category = "BAS Ask")
    bool bUseModel = true;

    /** Drawer and lid tasks: send the current sub-step sentence (models trained with --substeps). */
    UPROPERTY(EditAnywhere, Category = "BAS Ask")
    bool bSubstepInstructions = true;

    /** Ask the gateway's Llama 3 front end (/interpret) which task a request means; keywords if it does not answer. */
    UPROPERTY(EditAnywhere, Category = "BAS Ask")
    bool bUseLanguageFrontEnd = true;

    /** Keyword match of a free-text request to a task; false if nothing matches. */
    static bool MatchTask(const FString& Request, EBASTask& OutTask);

    UFUNCTION(BlueprintCallable, Category = "BAS Ask")
    void RunRequest(const FString& Request);

private:
    void BuildCommandBox();
    void RunByKeywords(const FString& Request, const FString& Note);
    void RunTask(EBASTask NewTask, const FString& Note);
    FString AskStatusText() const;
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
    double LatencyEma = 0.0;
    int32 TotalRequests = 0;
    FString LastPhase;
    FString AskStatus;
    bool bInterpreting = false;
    bool bReturningHome = false;
    bool bAskStarted = false;
    TSharedPtr<SWidget> CommandBox;
};
