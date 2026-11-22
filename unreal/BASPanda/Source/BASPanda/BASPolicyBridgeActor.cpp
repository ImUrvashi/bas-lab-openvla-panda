#include "BASPolicyBridgeActor.h"

#include "BASPolicyCameraActor.h"
#include "BASWorkcellActor.h"
#include "PandaRobotActor.h"
#include "Dom/JsonObject.h"
#include "Engine/GameViewportClient.h"
#include "GameFramework/PlayerController.h"
#include "HttpModule.h"
#include "Kismet/GameplayStatics.h"
#include "Misc/Base64.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Text/STextBlock.h"

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
    else if (bShowCommandBox)
    {
        BuildCommandBox();
    }
}

void ABASPolicyBridgeActor::EndPlay(const EEndPlayReason::Type Reason)
{
    UGameViewportClient* Viewport = GetWorld() ? GetWorld()->GetGameViewport() : nullptr;
    if (CommandBox.IsValid() && Viewport)
    {
        Viewport->RemoveViewportWidgetContent(CommandBox.ToSharedRef());
    }
    CommandBox.Reset();
    Super::EndPlay(Reason);
}

bool ABASPolicyBridgeActor::MatchTask(const FString& Request, EBASTask& OutTask)
{
    const FString Text = Request.ToLower();
    auto Has = [&Text](std::initializer_list<const TCHAR*> Words)
    {
        for (const TCHAR* Word : Words)
        {
            if (Text.Contains(Word))
            {
                return true;
            }
        }
        return false;
    };
    // Most specific objects first: "close the lid" is task 5, "close the drawer" task 2.
    if (Has({TEXT("vial"), TEXT("container"), TEXT("lid"), TEXT("seal")})) { OutTask = EBASTask::SealContainer; return true; }
    if (Has({TEXT("drawer"), TEXT("tool"), TEXT("stow")})) { OutTask = EBASTask::StowInDrawer; return true; }
    if (Has({TEXT("valve"), TEXT("coolant")})) { OutTask = EBASTask::CloseValve; return true; }
    if (Has({TEXT("filter"), TEXT("cartridge"), TEXT("scrubber")})) { OutTask = EBASTask::InsertFilter; return true; }
    if (Has({TEXT("cube"), TEXT("marker"), TEXT("pad")})) { OutTask = EBASTask::CubeToPad; return true; }
    return false;
}

void ABASPolicyBridgeActor::RunRequest(const FString& Request)
{
    if (bLoopActive)
    {
        StopClosedLoop(TEXT("Stopped"));
        AskStatus = TEXT("Stopped. Type the next request.");
        bAskStarted = false;
        return;
    }
    if (bInterpreting || Request.TrimStartAndEnd().IsEmpty())
    {
        return;
    }
    if (!bUseLanguageFrontEnd)
    {
        return RunByKeywords(Request, TEXT("Keywords:"));
    }
    // Llama 3 on the GPU server reads the request; its task number picks the trained sentence for OpenVLA.
    TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("text"), Request);
    FString JsonBody;
    FJsonSerializer::Serialize(Body, TJsonWriterFactory<>::Create(&JsonBody));
    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Http = FHttpModule::Get().CreateRequest();
    Http->SetURL((bUseModel ? ModelUrl : ScriptedUrl).Replace(TEXT("/act"), TEXT("/interpret")));
    Http->SetVerb(TEXT("POST"));
    Http->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Http->SetContentAsString(JsonBody);
    Http->SetTimeout(30.0f);
    TWeakObjectPtr<ABASPolicyBridgeActor> Self(this);
    Http->OnProcessRequestComplete().BindLambda(
        [Self, Request](FHttpRequestPtr, FHttpResponsePtr Response, bool bSucceeded)
        {
            if (!Self.IsValid())
            {
                return;
            }
            Self->bInterpreting = false;
            TSharedPtr<FJsonObject> Out;
            if (!bSucceeded || !Response.IsValid() || Response->GetResponseCode() != 200 ||
                !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Response->GetContentAsString()), Out) ||
                !Out.IsValid())
            {
                return Self->RunByKeywords(Request, TEXT("Llama 3 unavailable, keywords:"));
            }
            const int32 TaskNumber = static_cast<int32>(Out->GetNumberField(TEXT("task")));
            const FString Reply = TEXT("Llama 3: ") + Out->GetStringField(TEXT("reply"));
            UE_LOG(LogTemp, Display, TEXT("BAS ask: \"%s\" -> task %d (%s)"), *Request, TaskNumber, *Reply);
            if (TaskNumber >= 1 && TaskNumber <= 5)
            {
                return Self->RunTask(static_cast<EBASTask>(TaskNumber - 1), Reply);
            }
            Self->AskStatus = Reply;
            Self->bAskStarted = false;
        });
    if (Http->ProcessRequest())
    {
        bInterpreting = true;
        AskStatus = TEXT("Asking Llama 3 ...");
        bAskStarted = false;
    }
    else
    {
        RunByKeywords(Request, TEXT("Llama 3 unavailable, keywords:"));
    }
}

void ABASPolicyBridgeActor::RunByKeywords(const FString& Request, const FString& Note)
{
    EBASTask NewTask;
    if (!MatchTask(Request, NewTask))
    {
        AskStatus = TEXT("Not understood. Try: place the red cube on the marker / stow the tool in the drawer / "
                         "close the coolant valve / insert the filter / put the vial in the container");
        bAskStarted = false;
        return;
    }
    RunTask(NewTask, Note);
}

void ABASPolicyBridgeActor::RunTask(EBASTask NewTask, const FString& Note)
{
    Task = NewTask;
    NumEpisodes = 1;
    bUseTaskInstruction = true;
    PolicyUrl = bUseModel ? ModelUrl : ScriptedUrl;
    StartClosedLoop();
    bAskStarted = bLoopActive;
    AskStatus = Workcell ? FString::Printf(TEXT("%s  ->  task %d: %s"), *Note, static_cast<int32>(Task) + 1,
                                           *Workcell->GetTaskInstruction())
                         : LoopStatus;
}

FString ABASPolicyBridgeActor::AskStatusText() const
{
    const TCHAR* Mode = bUseModel ? TEXT("OpenVLA") : TEXT("scripted");
    if (bLoopActive)
    {
        const FString Now = Workcell && bSubstepInstructions ? Workcell->GetTaskInstruction(true) : FString();
        return FString::Printf(TEXT("%s   [%s, step %d/%d]%s%s"), *AskStatus, Mode, Step, MaxSteps,
                               Now.IsEmpty() ? TEXT("") : TEXT("   now: "), *Now);
    }
    if (!bAskStarted)
    {
        return AskStatus.IsEmpty() ? FString::Printf(TEXT("Ready (%s). Type a request and press Enter."), Mode) : AskStatus;
    }
    const bool bDone = LoopStatus.StartsWith(TEXT("Done: 1/1"));
    return FString::Printf(TEXT("%s   ->  %s after %d steps"), *AskStatus,
        bDone ? TEXT("DONE") : (LoopStatus.StartsWith(TEXT("Done")) ? TEXT("not finished") : *LoopStatus), Step);
}

void ABASPolicyBridgeActor::BuildCommandBox()
{
    UWorld* World = GetWorld();
    UGameViewportClient* Viewport = World ? World->GetGameViewport() : nullptr;
    if (!Viewport || !FindActors())
    {
        return;     // Simulate has no game viewport; the box is for Play
    }
    TWeakObjectPtr<ABASPolicyBridgeActor> Self(this);
    TSharedPtr<SEditableTextBox> Input;
    TSharedRef<SWidget> Box =
        SNew(SBox).HAlign(HAlign_Center).VAlign(VAlign_Bottom).Padding(FMargin(0, 0, 0, 40))
        [
            SNew(SBorder).Padding(12.0f).BorderBackgroundColor(FLinearColor(0.0f, 0.0f, 0.0f, 0.7f))
            [
                SNew(SVerticalBox)
                + SVerticalBox::Slot().AutoHeight()
                [
                    SNew(SHorizontalBox)
                    + SHorizontalBox::Slot().FillWidth(1.0f).Padding(0, 0, 8, 0)
                    [
                        SAssignNew(Input, SEditableTextBox)
                        .MinDesiredWidth(560.0f)
                        .HintText(FText::FromString(TEXT("Tell the robot what to do, e.g. close the coolant valve")))
                        .OnTextCommitted_Lambda([Self](const FText& Text, ETextCommit::Type How)
                        {
                            if (How == ETextCommit::OnEnter && Self.IsValid())
                            {
                                Self->RunRequest(Text.ToString());
                            }
                        })
                    ]
                    + SHorizontalBox::Slot().AutoWidth().Padding(0, 0, 8, 0)
                    [
                        SNew(SButton)
                        .Text_Lambda([Self]()
                        {
                            return FText::FromString(Self.IsValid() && Self->bLoopActive ? TEXT("Stop") : TEXT("Run"));
                        })
                        .OnClicked_Lambda([Self, Weak = TWeakPtr<SEditableTextBox>(Input)]()
                        {
                            if (Self.IsValid() && Weak.IsValid())
                            {
                                Self->RunRequest(Weak.Pin()->GetText().ToString());
                            }
                            return FReply::Handled();
                        })
                    ]
                    + SHorizontalBox::Slot().AutoWidth()
                    [
                        SNew(SButton)
                        .Text_Lambda([Self]()
                        {
                            return FText::FromString(Self.IsValid() && Self->bUseModel ? TEXT("Mode: OpenVLA") : TEXT("Mode: Scripted"));
                        })
                        .OnClicked_Lambda([Self]()
                        {
                            if (Self.IsValid() && !Self->bLoopActive)
                            {
                                Self->bUseModel = !Self->bUseModel;
                            }
                            return FReply::Handled();
                        })
                    ]
                ]
                + SVerticalBox::Slot().AutoHeight().Padding(0, 8, 0, 0)
                [
                    SNew(STextBlock)
                    .ColorAndOpacity(FLinearColor::White)
                    .Text_Lambda([Self]() { return FText::FromString(Self.IsValid() ? Self->AskStatusText() : FString()); })
                ]
            ]
        ];
    CommandBox = Box;
    Viewport->AddViewportWidgetContent(Box, 10);

    if (APlayerController* Player = World->GetFirstPlayerController())
    {
        // Watch the bench through the policy camera; mouse free for the box.
        if (AActor* Camera = UGameplayStatics::GetActorOfClass(this, ABASPolicyCameraActor::StaticClass()))
        {
            Player->SetViewTarget(Camera);
        }
        Player->SetShowMouseCursor(true);
        FInputModeGameAndUI Mode;
        Mode.SetWidgetToFocus(Input);
        Player->SetInputMode(Mode);
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
    bReturningHome = CommandBox.IsValid();     // after a typed request, glide back to the rest pose
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
    if (bReturningHome && !bLoopActive && Panda && !Panda->IsCommandActive())
    {
        static const TArray<float> Home = {0.0f, 0.0f, 0.0f, -1.57079f, 0.0f, 1.57079f, -0.7853f};
        float Off = 0.0f;
        for (int32 J = 0; J < Home.Num() && J < Panda->ArmJointsRad.Num(); ++J)
        {
            Off = FMath::Max(Off, FMath::Abs(Home[J] - Panda->ArmJointsRad[J]));
        }
        // the Panda limits each command to 0.15 rad per joint, so this takes a few steps
        bReturningHome = Off > 0.005f && Panda->CommandJointTargets(Home, 1.0f, 0.25f);
        return;
    }
    // Next step once the previous action is answered; stop-go mode also waits until it has been executed.
    if (!bLoopActive || bRequestInFlight || !Panda || (!bSmoothMotion && Panda->IsCommandActive()))
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
        bUseTaskInstruction && Workcell ? Workcell->GetTaskInstruction(bSubstepInstructions) : Instruction);
    Payload->SetStringField(TEXT("image_b64"), FBase64::Encode(RGB.GetData(), RGB.Num()));
    Payload->SetArrayField(TEXT("shape"), JsonNumbers({240, 320, 3}));
    if (Panda && Panda->ArmJointsRad.Num() == 7)
    {
        TArray<TSharedPtr<FJsonValue>> Joints;
        // smooth mode: the arm is still moving, so actions are relative to where this move ends
        for (const float Angle : bSmoothMotion ? Panda->GetCommandTargetsRad() : Panda->ArmJointsRad)
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
    const double RoundTrip = FPlatformTime::Seconds() - RequestStartTime;
    LatencySum += RoundTrip;
    LatencyEma = LatencyEma > 0.0 ? 0.8 * LatencyEma + 0.2 * RoundTrip : RoundTrip;
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
    if (Panda)
    {
        Panda->SpeedEaseSec = bLoop && bSmoothMotion ? 0.12f : 0.0f;
    }
    if (!Panda || !Panda->CommandJointTargets(JointTargets, (*Targets)[7]->AsNumber(),
        !bLoop ? 0.0f : bSmoothMotion ? FMath::Clamp(1.2f * static_cast<float>(LatencyEma), StepDurationSec, 2.0f) : StepDurationSec))
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
