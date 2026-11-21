#include "BASPolicyBridgeActor.h"

#include "BASPolicyCameraActor.h"
#include "BASWorkcellActor.h"
#include "PandaRobotActor.h"
#include "Dom/JsonObject.h"
#include "HttpModule.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/Base64.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace
{
    // closed_loop_panda.py: success once the task check holds for 10 policy steps.
    constexpr int32 SuccessHoldSteps = 10;

    TArray<TSharedPtr<FJsonValue>> JsonNumbers(std::initializer_list<double> Values)
    {
        TArray<TSharedPtr<FJsonValue>> Out;
        for (const double Value : Values)
        {
            Out.Add(MakeShared<FJsonValueNumber>(Value));
        }
        return Out;
    }
}

ABASPolicyBridgeActor::ABASPolicyBridgeActor()
{
    PrimaryActorTick.bCanEverTick = true;
}

void ABASPolicyBridgeActor::BeginPlay()
{
    Super::BeginPlay();
    if (bRunClosedLoopOnBeginPlay)
    {
        StartClosedLoop();
    }
    else if (bSendOnBeginPlay)
    {
        SendTestObservation();
    }
}

bool ABASPolicyBridgeActor::FindActors()
{
    if (!Panda)
    {
        Panda = Cast<APandaRobotActor>(UGameplayStatics::GetActorOfClass(this, APandaRobotActor::StaticClass()));
    }
    if (!Workcell)
    {
        Workcell = Cast<ABASWorkcellActor>(UGameplayStatics::GetActorOfClass(this, ABASWorkcellActor::StaticClass()));
    }
    return Panda && Workcell;
}

void ABASPolicyBridgeActor::SendTestObservation()
{
    FindActors();
    SendObservation();
}

void ABASPolicyBridgeActor::StartClosedLoop()
{
    if (!FindActors())
    {
        LoopStatus = TEXT("Failed: level needs PandaRobotActor and BASWorkcellActor");
        UE_LOG(LogTemp, Error, TEXT("BAS loop: %s"), *LoopStatus);
        return;
    }
    Workcell->Task = Task;
    LayoutStream.Initialize(LayoutSeed);
    Episode = 0;
    Successes = 0;
    LatencySum = 0.0;
    TotalRequests = 0;
    bLoopActive = true;
    StartEpisode();
}

void ABASPolicyBridgeActor::StopClosedLoop(const FString& Reason)
{
    bLoopActive = false;
    LoopStatus = Reason;
    UE_LOG(LogTemp, Display, TEXT("BAS loop: %s"), *LoopStatus);
}

void ABASPolicyBridgeActor::StartEpisode()
{
    if (bRandomizeLayout)
    {
        Workcell->RandomizeLayout(LayoutStream);
    }
    Workcell->ResetTask();
    Panda->ResetHome();
    Step = 0;
    SuccessStreak = 0;
    LastPhase.Reset();
    LoopStatus = FString::Printf(TEXT("Episode %d/%d running"), Episode + 1, NumEpisodes);
    const FVector Object = Workcell->GetActiveObjectMujoco(), Target = Workcell->GetTargetMujoco();
    UE_LOG(LogTemp, Display, TEXT("BAS loop: episode %d start, task %d \"%s\", object (%.3f, %.3f) target (%.3f, %.3f)"),
        Episode + 1, static_cast<int32>(Workcell->Task) + 1, *Workcell->GetTaskInstruction(),
        Object.X, Object.Y, Target.X, Target.Y);
}

void ABASPolicyBridgeActor::EndEpisode(bool bSuccess, const FString& Why)
{
    Successes += bSuccess ? 1 : 0;
    UE_LOG(LogTemp, Display, TEXT("BAS loop: episode %d %s (%s) after %d steps"),
        Episode + 1, bSuccess ? TEXT("SUCCESS") : TEXT("FAILED"), *Why, Step);
    ++Episode;
    if (Episode < NumEpisodes)
    {
        StartEpisode();
        return;
    }
    StopClosedLoop(FString::Printf(TEXT("Done: %d/%d successful"), Successes, NumEpisodes));
    UE_LOG(LogTemp, Display, TEXT("BAS loop: mean request latency %.0f ms"),
        1000.0 * LatencySum / FMath::Max(1, TotalRequests));
}

void ABASPolicyBridgeActor::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    // Next step only once the previous action has been fully executed and answered.
    if (!bLoopActive || bRequestInFlight || !Panda || Panda->IsCommandActive())
    {
        return;
    }
    if (Step > 0)
    {
        SuccessStreak = Workcell->IsTaskSuccess() ? SuccessStreak + 1 : 0;
        // A scripted expert reports phases: let it finish its retreat (recorded demos end at DONE).
        const bool bExpertFinished = LastPhase.IsEmpty() || LastPhase == TEXT("DONE");
        if (SuccessStreak >= SuccessHoldSteps && bExpertFinished)
        {
            EndEpisode(true, TEXT("task check passed"));
            return;
        }
    }
    if (Step >= MaxSteps)
    {
        EndEpisode(false, TEXT("max steps"));
        return;
    }
    if (!SendObservation())
    {
        StopClosedLoop(TEXT("Stopped: ") + LastResult);
    }
}

bool ABASPolicyBridgeActor::SendObservation()
{
    ABASPolicyCameraActor* Camera = Cast<ABASPolicyCameraActor>(
        UGameplayStatics::GetActorOfClass(this, ABASPolicyCameraActor::StaticClass()));
    if (!Camera)
    {
        LastResult = TEXT("No BASPolicyCameraActor in this level");
        UE_LOG(LogTemp, Error, TEXT("BAS policy: %s"), *LastResult);
        return false;
    }

    TArray<uint8> RGB;
    FString Error;
    if (!Camera->CaptureRGB(RGB, Error))
    {
        LastResult = Error;
        UE_LOG(LogTemp, Error, TEXT("BAS policy: %s"), *LastResult);
        return false;
    }

    TSharedRef<FJsonObject> Payload = MakeShared<FJsonObject>();
    Payload->SetStringField(TEXT("instruction"),
        bUseTaskInstruction && Workcell ? Workcell->GetTaskInstruction() : Instruction);
    Payload->SetStringField(TEXT("image_b64"), FBase64::Encode(RGB.GetData(), RGB.Num()));
    Payload->SetArrayField(TEXT("shape"), JsonNumbers({240, 320, 3}));
    if (Panda && Panda->ArmJointsRad.Num() == 7)
    {
        TArray<TSharedPtr<FJsonValue>> Joints;
        for (const float Angle : Panda->ArmJointsRad)
        {
            Joints.Add(MakeShared<FJsonValueNumber>(Angle));
        }
        Payload->SetArrayField(TEXT("joints_rad"), Joints);
        Payload->SetNumberField(TEXT("gripper_opening"), FMath::Clamp(Panda->GetGripperOpenness(), 0.0f, 1.0f));
    }
    // Privileged task state for the scripted expert and for logging; OpenVLA never sees it.
    if (Workcell)
    {
        // cube_pos/marker_xy keep task 1's field names; for other tasks they are the active object and target.
        const FVector Object = Workcell->GetActiveObjectMujoco(), Target = Workcell->GetTargetMujoco();
        Payload->SetNumberField(TEXT("task"), static_cast<int32>(Workcell->Task) + 1);
        Payload->SetArrayField(TEXT("cube_pos"), JsonNumbers({Object.X, Object.Y, Object.Z}));
        Payload->SetArrayField(TEXT("marker_xy"), JsonNumbers({Target.X, Target.Y}));
        Payload->SetArrayField(TEXT("target_pos"), JsonNumbers({Target.X, Target.Y, Target.Z}));
        TMap<FString, double> State;
        Workcell->GetTaskState(State);
        TSharedRef<FJsonObject> StateJson = MakeShared<FJsonObject>();
        for (const TPair<FString, double>& Entry : State)
        {
            StateJson->SetNumberField(Entry.Key, Entry.Value);
        }
        Payload->SetObjectField(TEXT("state"), StateJson);
    }
    Payload->SetNumberField(TEXT("episode"), Episode);
    Payload->SetNumberField(TEXT("step"), Step);

    FString JsonBody;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonBody);
    FJsonSerializer::Serialize(Payload, Writer);

    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
    Request->SetURL(PolicyUrl);
    Request->SetVerb(TEXT("POST"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Request->SetContentAsString(JsonBody);
    Request->SetTimeout(30.0f);
    Request->OnProcessRequestComplete().BindUObject(this, &ABASPolicyBridgeActor::HandleResponse);
    LastResult = FString::Printf(TEXT("Sending %d RGB bytes"), RGB.Num());
    if (!Request->ProcessRequest())
    {
        LastResult = TEXT("Could not start HTTP request");
        UE_LOG(LogTemp, Error, TEXT("BAS policy: %s"), *LastResult);
        return false;
    }
    bRequestInFlight = true;
    RequestStartTime = FPlatformTime::Seconds();
    return true;
}

void ABASPolicyBridgeActor::HandleResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bSucceeded)
{
    bRequestInFlight = false;
    LatencySum += FPlatformTime::Seconds() - RequestStartTime;
    ++TotalRequests;
    const bool bLoop = bLoopActive;
    auto Fail = [this, bLoop](const FString& Message)
    {
        LastResult = Message;
        UE_LOG(LogTemp, Error, TEXT("BAS policy: %s"), *LastResult);
        if (bLoop)
        {
            StopClosedLoop(TEXT("Stopped: ") + Message);
        }
    };

    if (!bSucceeded || !Response.IsValid())
    {
        return Fail(TEXT("HTTP connection failed; is the gateway running on port 8000?"));
    }
    TSharedPtr<FJsonObject> Body;
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Response->GetContentAsString());
    if (Response->GetResponseCode() != 200 || !FJsonSerializer::Deserialize(Reader, Body) || !Body.IsValid())
    {
        return Fail(FString::Printf(TEXT("HTTP %d: %s"), Response->GetResponseCode(), *Response->GetContentAsString()));
    }
    const TArray<TSharedPtr<FJsonValue>>* Action = nullptr;
    if (!Body->TryGetArrayField(TEXT("action"), Action) || !Action || Action->Num() != 7)
    {
        return Fail(TEXT("Policy response has no seven-value action"));
    }
    if (!bLoop && !bApplyJointTargets)
    {
        LastResult = FString::Printf(TEXT("OK: received seven values; gripper=%.2f (not applied)"), (*Action)[6]->AsNumber());
        UE_LOG(LogTemp, Display, TEXT("BAS policy: %s"), *LastResult);
        return;
    }

    const TArray<TSharedPtr<FJsonValue>>* Targets = nullptr;
    if (!Body->TryGetArrayField(TEXT("joint_targets"), Targets) || !Targets || Targets->Num() != 8)
    {
        return Fail(TEXT("No eight-value joint_targets in response; no motion applied"));
    }
    TArray<float> JointTargets;
    for (int32 Index = 0; Index < 8; ++Index)
    {
        if (!(*Targets)[Index].IsValid() || (*Targets)[Index]->Type != EJson::Number ||
            !FMath::IsFinite((*Targets)[Index]->AsNumber()))
        {
            return Fail(TEXT("Invalid joint_targets; no motion applied"));
        }
        if (Index < 7)
        {
            JointTargets.Add(static_cast<float>((*Targets)[Index]->AsNumber()));
        }
    }
    FindActors();
    if (!Panda || !Panda->CommandJointTargets(JointTargets, (*Targets)[7]->AsNumber(), bLoop ? StepDurationSec : 0.0f))
    {
        return Fail(TEXT("Panda could not accept joint targets; run Simulate with motion test disabled"));
    }
    if (bLoop)
    {
        ++Step;
        LastPhase.Reset();
        Body->TryGetStringField(TEXT("phase"), LastPhase);
        LastResult = FString::Printf(TEXT("Step %d %s"), Step, *LastPhase);
        if (Step % 25 == 0)
        {
            UE_LOG(LogTemp, Display, TEXT("BAS loop: episode %d %s"), Episode + 1, *LastResult);
        }
        return;
    }
    LastResult = TEXT("OK: applied bounded joint target from policy gateway");
    UE_LOG(LogTemp, Display, TEXT("BAS policy: %s"), *LastResult);
}
